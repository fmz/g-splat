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
            

        self.points    = torch.tensor(self.points, device=self.device, dtype=torch.float32)
        self.opacities = torch.tensor(self.opacities, device=self.device, dtype=torch.float32)
        self.scales    = torch.tensor(self.scales, device=self.device, dtype=torch.float32)
        self.rots      = torch.tensor(self.rots, device=self.device, dtype=torch.float32)
        self.colors    = torch.tensor(self.colors, device=self.device, dtype=torch.float32)

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

        rots = np.zeros((num_pts, 4))
        rots[:, 0] = 1

        colors = np.random.uniform(0.0, 1.0, (num_pts, 3))


        return points, opacities, scales, rots, colors
