from typing import NamedTuple
import torch
import torch.nn as nn
import numpy as np
from utils.get_data_col import read_points3D_text
from utils.sh_utils import RGB2SH
import torch.optim as optim

class BoundingBox(NamedTuple):
    lo: np.array
    hi: np.array


class Scene():
    def __init__(self,
                 bbox : BoundingBox,
                 torch_device = torch.device('cuda'),
                 init_method="random",
                 init_method="random",
                 points_txt = 'none',
                 max_sh_degree=4):
        self.device = torch_device
        self.bbox   = bbox

        if init_method == "random":
            self.points, self.opacities, self.scales, self.rots, self.colors = \
                self.get_random_points(self.get_num_points_for_bbox(density=0.5))
        elif init_method == 'from-dataset':
            output,self.points, self.colors = read_points3D_text(points_txt)
            num_pts = len(output)
            self.opacities = np.ones((num_pts, 1))
            x, z, self.scales, self.rots, y = \
                self.get_random_points(num_pts)

        self.points    = nn.Parameter(torch.tensor(self.points, device=self.device, dtype=torch.float32, requires_grad=True))
        self.opacities = nn.Parameter(torch.tensor(self.opacities, device=self.device, dtype=torch.float32, requires_grad=True))
        self.scales    = nn.Parameter(torch.tensor(self.scales, device=self.device, dtype=torch.float32, requires_grad=True))
        self.rots      = nn.Parameter(torch.tensor(self.rots, device=self.device, dtype=torch.float32, requires_grad=True))
        
        # Colors
        self.sh_clrs         = RGB2SH(torch.tensor(self.colors, device=self.device, dtype=torch.float32))
        self.features        = torch.zeros((self.sh_clrs.shape[0], max_sh_degree**2, 3), device=self.device, dtype=torch.float32)
        self.features[:,0,:] = self.sh_clrs
        self.features        = nn.Parameter(self.features).requires_grad_(True)

        self.viewspace_grad_accum = torch.zeros((self.points.shape[0],1), device=self.device)
        self.grad_denominator = torch.zeros((self.points.shape[0],1), device=self.device)
        #self.max_radii = torch.zeros((self.points.shape[0],1), device=self.device)

    # Density is points per unit area
    def get_num_points_for_bbox(self, density=1):
        bbox_range = self.bbox.hi - self.bbox.lo
        num_pts = np.round(bbox_range * density).astype('int')
        num_pts = np.prod(num_pts)

        return num_pts

    def get_random_points(self, num_pts, seed=None):
        rng = np.random.default_rng(seed)

        bbox_range = self.bbox.hi - self.bbox.lo
        
        points = rng.uniform(0.0, 1.0, (num_pts, 3))
        points = points * bbox_range + self.bbox.lo

        opacities = rng.uniform(0.0, 1.0, (num_pts, 1))

        scl_mean = np.array([0.05, 0.05, 0.05])
        scl_cov  = np.array([0.01, 0.01, 0.01])
        
        scales = rng.normal(scl_mean, scl_cov, size=(num_pts, 3))

        rots = rng.random((num_pts, 4))
        rots /= np.linalg.norm(rots, axis=1, keepdims=True)

        colors = np.random.uniform(0.0, 1.0, (num_pts, 3))

        return points, opacities, scales, rots, colors
    
    def init_optimizer(self, lrs):
        self.lrs = lrs
        # Set up a different learning rate for each parameter
        params = [{'params': [self.points], 'lr': lrs['position'], 'name':'point'},
                {'params': [self.opacities], 'lr': lrs['opacity'], 'name':'opacity'},
                {'params': [self.scales], 'lr': lrs['scale'], 'name':'scale'},
                {'params': [self.rots], 'lr': lrs['rotation'], 'name':'rotation'},
                {'params': [self.features], 'lr': lrs['color'], 'name':'color'},
                ]
        
        self.optimizer = optim.Adam(params, 0.0, eps=1e-15)

    def update_parameters(self, update_dict, pruning = False):
        #print("Point tensor print: " + str(update_dict['point']))
        self.points    = update_dict['point']
        self.opacities = update_dict['opacity']
        self.scales    = update_dict['scale']
        self.rots      = update_dict['rotation']
        self.features    = update_dict['color']

        if not pruning:
            self.viewspace_grad_accum = torch.zeros((self.points.shape[0],1), device=self.device)
            self.grad_denominator = torch.zeros((self.points.shape[0],1), device=self.device)
            #self.max_radii = torch.zeros((self.points.shape[0],1), device=self.device)

        # # Maybe necessary? (TODO: give it updated lrs)
        # self.init_optimizer(self.lrs)

    def add_to_optimizer(self, new_dict):
        update_dict= {}
        for group in self.optimizer.param_groups:
            new_params = new_dict[group['name']]
            current_state = self.optimizer.state.get(group['params'][0], None)
            if current_state is not None:
                current_state['exp_avg'] = torch.cat((current_state["exp_avg"],torch.zeros_like(new_params)), dim = 0)
                current_state['exp_avg_sq'] = torch.cat((current_state["exp_avg_sq"],torch.zeros_like(new_params)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0],new_params), dim=0).requires_grad_(True))
                self.optimizer.state[group["params"][0]] = current_state
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0],new_params), dim=0).requires_grad_(True))
                
            update_dict[group['name']] = group["params"][0]
            #Check to be sure this isn't a dict
            assert type(group["params"][0]) == nn.Parameter

        self.update_parameters(update_dict)


    def prune_from_optimizer(self, mask):
        keep = ~mask
        prune_dict = {}
        for group in self.optimizer.param_groups:
            current_state = self.optimizer.state.get(group['params'][0], None)
            if current_state is not None:
                current_state['exp_avg'] = current_state['exp_avg'][keep]
                current_state['exp_avg_sq'] = current_state['exp_avg_sq'][keep]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(group["params"][0][keep].requires_grad_(True))
                self.optimizer.state[group["params"][0]] = current_state
            else:
                group["params"][0] = nn.Parameter(group["params"][0][keep].requires_grad_(True))
                
            prune_dict[group['name']] = group["params"][0]

        self.update_parameters(prune_dict, pruning=True)
        self.viewspace_grad_accum = self.viewspace_grad_accum[keep]
        self.grad_denominator = self.grad_denominator[keep]
        #self.max_radii = self.max_radii[keep]
    
    def regularization_loss(self):
        # TODO
        reg_loss = torch.mean(self.scales.clamp(min=0.01, max=1.0) ** 2)  # Penalize extreme scales
        reg_loss += torch.mean(self.opacities.clamp(min=0.0, max=1.0) ** 2)  # Clamp opacities
        return reg_loss

    def log_parameters(self):
        print(f"Points mean: {self.points.mean(dim=0)}, std: {self.points.std(dim=0)}")
        print(f"Scales mean: {self.scales.mean(dim=0)}, std: {self.scales.std(dim=0)}")
        print(f"Opacities mean: {self.opacities.mean()}, std: {self.opacities.std()}")
        print(f"Colors mean: {self.features.mean(dim=0)}, std: {self.features.std(dim=0)}")

    def prune_and_densify(self,  minimum_opacity = 0.05, max_size = 100, grad_threshold = 0.0002):
        viewspace_grads = self.viewspace_grad_accum/self.grad_denominator

        bbox_range = self.bbox.hi - self.bbox.lo
        #Densification
        #self.clone_gaussians(viewspace_grads, grad_threshold, extent=bbox_range)
        self.split_gaussians(viewspace_grads, grad_threshold, extent=bbox_range)
        
        #Pruning
        #self.prune_gaussians(extent=bbox_range)
        
        torch.cuda.empty_cache()

    def add_densification_data(self, viewspace_points, visible_filter):
        self.viewspace_grad_accum[visible_filter] += torch.norm(
            viewspace_points.grad[visible_filter,:2], dim=-1, keepdim = True
            )
        self.grad_denominator[visible_filter] += 1

    def split_gaussians(self, grads, threshold, extent, percent_dense = 0.01, N=2):
        #might need to pad grads to be of shape gaussian_points?

        #Make mask of points that need to be split
        
        mask = torch.logical_and(torch.where(torch.norm(grads) >= threshold, True, False), 
                                 torch.max(self.scales, dim = 1).values > extent[0] * percent_dense)

        
        stds = torch.exp(self.scales[mask]).repeat(N,1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean = means, std = stds)

        rotation = self.build_rotation(self.rots[mask]).repeat(N,1,1)

        #Generate new gaussian values
        new_points = torch.bmm(rotation, samples.unsqueeze(-1)).squeeze(-1) + self.points[mask].repeat(N,1)
        new_opacities = self.opacities[mask].repeat(N,1)
        new_scales = self.scales[mask].repeat(N,1)/(0.8 * N)
        new_rots = self.rots[mask].repeat(N,1)
        new_colors = self.features[mask].repeat(N,1,1)

        new_dictionary = {'point': new_points, 'opacity': new_opacities, 'scale': new_scales, 'rotation': new_rots, 'color': new_colors}

        self.add_to_optimizer(new_dictionary)
        #Now, prune original gaussians
        prune_mask = torch.cat((mask, torch.zeros(N * mask.sum(), device="cuda", dtype=bool)))
        self.prune_from_optimizer(prune_mask)
               
        print(f'Split {mask.sum()} gaussians')


    def build_rotation(self, r):
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

        q = r / norm[:, None]

        R = torch.zeros((q.size(0), 3, 3), device='cuda')

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        return R

    def clone_gaussians(self, grads, threshold, extent, percent_dense = 0.01, N=2):
        #might need to pad grads to be of shape gaussian_points?
        #print("Cond 1: " + str(torch.where(torch.norm(grads) >= threshold, True, False).size()))
        #print("Cond 2: " + str((torch.max(self.scales, dim = 1).values <= extent[0] * extent[1] * percent_dense).size()))
        mask = torch.logical_and(torch.where(torch.norm(grads) >= threshold, True, False), torch.max(self.scales, dim = 1).values <= extent[0] * percent_dense)

        #Generate new gaussian values
        new_points = self.points[mask]
        new_opacities = self.opacities[mask]
        new_scales = self.scales[mask] * 0.8
        new_rots = self.rots[mask]
        new_colors = self.features[mask]
        
        new_dictionary = {'point': new_points, 'opacity': new_opacities, 'scale': new_scales, 'rotation': new_rots, 'color': new_colors}
        self.add_to_optimizer(new_dictionary)
                
        torch.cuda.empty_cache()

        

    def prune_gaussians(self, extent, min_opacity = 0.05, percent_dense = 0.1, max_size = 20):
        prune_mask = (self.opacities < min_opacity).squeeze()
        size_mask_1 =  torch.max(self.scales, dim = 1).values > extent[0] * percent_dense
        #size_mask_2 = self.max_radii > max_size
        #final_mask = torch.logical_or(torch.logical_or(prune_mask, size_mask_1), size_mask_2)
        final_mask = torch.logical_or(prune_mask, size_mask_1)

        self.prune_from_optimizer(final_mask)
        torch.cuda.empty_cache()


    def prune_points(self, mask):
        prune_dict = {}
        for group in self.optimizer.param_groups:
            group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
            prune_dict[group['name']] = group["params"][0]

        self.update_parameters(prune_dict, pruning=True)
        self.viewspace_grad_accum = self.viewspace_grad_accum[mask]
        self.grad_denominator = self.grad_denominator[mask]
