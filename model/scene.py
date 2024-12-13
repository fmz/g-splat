from typing import NamedTuple
import torch
import numpy as np

class BoundingBox(NamedTuple):
    lo: np.array
    hi: np.array


class Scene():
    def __init__(self,
                 bbox : BoundingBox,
                 torch_device = torch.device('cuda'),
                 init_method="random"):
        self.device = torch_device
        self.bbox   = bbox

        if init_method == "random":
            self.points, self.opacities, self.scales, self.rots, self.colors = \
                self.get_random_points(density=1)
        elif init_method == 'from-dataset':
            pass

        self.points    = torch.tensor(self.points, device=self.device, dtype=torch.float32, requires_grad=True)
        self.opacities = torch.tensor(self.opacities, device=self.device, dtype=torch.float32, requires_grad=True)
        self.scales    = torch.tensor(self.scales, device=self.device, dtype=torch.float32, requires_grad=True)
        self.rots      = torch.tensor(self.rots, device=self.device, dtype=torch.float32, requires_grad=True)
        self.colors    = torch.tensor(self.colors, device=self.device, dtype=torch.float32, requires_grad=True)

        self.position_grads = torch.zeros(self.points.shape, device=self.device)


    # Density is points per unit area
    def get_random_points(self, density=1, seed=None):
        bbox_range = self.bbox.hi - self.bbox.lo
        num_pts = np.round(bbox_range * density).astype('int')
        num_pts = np.prod(num_pts)

        rng = np.random.default_rng(seed)

        points = rng.uniform(0.0, 1.0, (num_pts, 3))
        points = points * bbox_range + self.bbox.lo

        opacities = rng.uniform(0.0, 1.0, (num_pts, 1))

        scl_mean = np.array([0.1, 0.1, 0.1])
        scl_cov  = np.array([0.1, 0.1, 0.1])
        
        scales = rng.normal(scl_mean, scl_cov, size=(num_pts, 3))

        rots = rng.random((num_pts, 4))
        rots /= np.linalg.norm(rots, axis=1, keepdims=True)

        colors = np.random.uniform(0.0, 1.0, (num_pts, 3))


        return points, opacities, scales, rots, colors

    def get_optimizable_params(self):
        return [self.points, self.opacities, self.scales, self.rots, self.colors]
    
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

    def prune_and_densify(minimum_opacity = 0.05, max_size = 100, ):
        self.prune

    def prune_gaussians():
        pass

    def densify_gaussians():
        pass