from typing import NamedTuple
import torch
import torch.nn as nn
import numpy as np
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
                 max_sh_degree=4):
        self.device = torch_device
        self.bbox   = bbox

        if init_method == "random":
            self.points, self.opacities, self.scales, self.rots, self.colors = \
                self.get_random_points(density=1)
        elif init_method == 'from-dataset':
            pass

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

    # Density is points per unit area
    def get_random_points(self, density=1, seed=None):
        bbox_range = self.bbox.hi - self.bbox.lo
        num_pts = np.round(bbox_range * density).astype('int')
        num_pts = np.prod(num_pts)

        rng = np.random.default_rng(seed)

        points = rng.uniform(0.0, 1.0, (num_pts, 3))
        points = points * bbox_range + self.bbox.lo

        opacities = rng.uniform(0.0, 1.0, (num_pts, 1))

        scl_mean = np.array([0.01, 0.01, 0.01])
        scl_cov  = np.array([0.1, 0.1, 0.1])
        
        scales = rng.normal(scl_mean, scl_cov, size=(num_pts, 3))

        rots = rng.random((num_pts, 4))
        rots /= np.linalg.norm(rots, axis=1, keepdims=True)

        colors = np.random.uniform(0.0, 1.0, (num_pts, 3))

        return points, opacities, scales, rots, colors
    
    def init_optimizer(self, lrs):
        # Set up a different learning rate for each parameter
        params = [{'params': [self.points], 'lr': lrs['position'], 'name':'point'},
                {'params': [self.opacities], 'lr': lrs['opacity'], 'name':'opacity'},
                {'params': [self.scales], 'lr': lrs['scale'], 'name':'scale'},
                {'params': [self.rots], 'lr': lrs['rotation'], 'name':'rotation'},
                {'params': [self.features], 'lr': lrs['color'], 'name':'color'},
                ]
        
        self.optimizer = optim.Adam(params, 0.0)

    def update_parameters(self, update_dict):
        self.points    = update_dict['point']
        self.opacities = update_dict['opacity']
        self.scales    = update_dict['scale']
        self.rots      = update_dict['rotation']
        self.colors    = update_dict['color']

        self.viewspace_grad_accum = torch.zeros((self.points.shape[0],1), device=self.device)
        self.grad_denominator = torch.zeros((self.points.shape[0],1), device=self.device)


    def add_to_optimizer(self, new_dict):
        update_dict= {}
        for group in self.optimizer.param_groups:
            new_params = new_dict[group['name']]
            group["params"][0] = nn.Parameter(torch.cat((group["params"][0],new_params), dim=0).requires_grad_(True))
            update_dict[group['name']] = group["params"][0]

        return update_dict


    def remove_from_optimizer(self, new_points, new_opacities, new_scales, new_rots, new_colors):
        pass
    
    def regularization_loss(self):
        # TODO
        reg_loss = torch.mean(self.scales.clamp(min=0.01, max=1.0) ** 2)  # Penalize extreme scales
        reg_loss += torch.mean(self.opacities.clamp(min=0.0, max=1.0) ** 2)  # Clamp opacities
        return reg_loss

    def log_parameters(self):
        print(f"Points mean: {self.points.mean(dim=0)}, std: {self.points.std(dim=0)}")
        print(f"Scales mean: {self.scales.mean(dim=0)}, std: {self.scales.std(dim=0)}")
        print(f"Opacities mean: {self.opacities.mean()}, std: {self.opacities.std()}")
        print(f"Colors mean: {self.colors.mean(dim=0)}, std: {self.colors.std(dim=0)}")

    def prune_and_densify(self,  minimum_opacity = 0.05, max_size = 100, grad_threshold = 0.0002):
        

        viewspace_grads = self.viewspace_grad_accum/self.grad_denominator

        #Densification
        #self.split_gaussians(viewspace_grads, grad_threshold)
        self.clone_gaussians(viewspace_grads, grad_threshold)

        #Pruning
        #self.prune_gaussians(viewspace_grads)

    def add_densification_data(self, viewspace_points, visible_filter):
        self.viewspace_grad_accum[visible_filter] += torch.norm(viewspace_points.grad[visible_filter,:2], dim=-1, keepdim = True)
        self.grad_denominator[visible_filter] += 1


    def split_gaussians(self, grads, threshold, percent_dense = 0.01, N=2):
        #might need to pad grads to be of shape gaussian_points?

        #Make mask of points that need to be split
        bbox_range = self.bbox.hi - self.bbox.lo
        mask = torch.logical_and(torch.where(grads > threshold, True, False), torch.max(self.scales, dim = 1).values > bbox_range * percent_dense)

        
        stds = torch.e(self.scales)[mask].repeat(N,1)
        means = torch.zeros((stds.size, 3), device=self.device)
        samples = torch.norm(means = means, stds = stds)

        rotation = self.build_rotation(self.rots).repeat(N,1)

        #Generate new gaussian values
        new_points = torch.bmm(rotation, samples) + self.points.repeat(N,1)
        new_opacities = self.opacities.repeat(N,1)
        new_scales = torch.log(self.scales.repeat(N,1)/(0.8 * N))
        new_rots = self.rots.repeat(N,1)
        new_colors = self.colors.repeat(N,1)
        

        #Now, prune original gaussians

    def build_rotation(self, rots):
        pass

    def clone_gaussians(self, grads, threshold, percent_dense = 0.01, N=2):
        #might need to pad grads to be of shape gaussian_points?
        bbox_range = self.bbox.hi - self.bbox.lo
        mask = torch.logical_and(torch.where(grads > threshold, True, False), torch.max(self.scales, dim = 1).values <= bbox_range[0] * percent_dense)

        #Generate new gaussian values
        new_points = self.points.repeat(N,1)
        new_opacities = self.opacities.repeat(N,1)
        new_scales = torch.log(self.scales.repeat(N,1)/(0.8 * N))
        new_rots = self.rots.repeat(N,1)
        new_colors = self.colors.repeat(N,1)

        new_dictionary = {'point': new_points, 'opacity': new_opacities, 'scale': new_scales, 'rotation': new_rots, 'color': new_colors}
        self.add_to_optimizer(new_dictionary)


    def prune_gaussians(self, opacity, max_radiis, max_size):
        pass