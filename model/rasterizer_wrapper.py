from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from model.camera import Camera
from model.scene import Scene

import torch
import torch.nn as nn

class GausRast(nn.Module):
    def __init__(self, torch_device = torch.device("cuda")):
        super(GausRast, self).__init__()
        self.device = torch_device
        self.bg_clr = torch.tensor([0.0,0.0,0.0], device=self.device, dtype=torch.float32)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.exp = torch.exp

    def forward(self, scene : Scene, camera : Camera):

        viewspace_points = torch.zeros_like(scene.points, device=self.device, requires_grad = True)
        raster_settings = GaussianRasterizationSettings(
            image_height   = int(camera.h),
            image_width    = int(camera.w),
            tanfovx        = float(camera.tanfovx),
            tanfovy        = float(camera.tanfovy),
            bg             = self.bg_clr,
            scale_modifier = float(1.0),
            viewmatrix     = camera.view,
            projmatrix     = camera.proj,
            sh_degree      = 0,   # Figure this out
            campos         = camera.pos,
            prefiltered    = False,
            debug          = True,
            antialiasing   = True
        )

        rasterizer = GaussianRasterizer(raster_settings)

        scales    = self.exp(scene.scales)
        colors    = self.sigmoid(scene.colors)
        opacities = self.sigmoid(scene.opacities)

        rgb, radii, depth = rasterizer(
            means3D=scene.points,
            means2D=viewspace_points,
            shs=None,
            colors_precomp=colors,
            opacities=opacities,
            scales=scales,
            rotations=scene.rots,
        )

        # # A couple test gaussians:
        # g_poss      = torch.tensor([[0.0, 0.0, 4.0],
        #                             [0.0, 2.0, 2.0],
        #                             [0.0, -4.0, 0.0],
        #                             [6.0, 0.0, -2.0],
        #                             [-8.0, 0.0, -4.0]], device=self.device)
        # g_opacities = torch.tensor([[1.0],
        #                             [1.0],
        #                             [1.0],
        #                             [1.0],
        #                             [1.0]], device=self.device)
        # g_scales    = torch.tensor([[1.0, 1.0, 1.0],
        #                             [1.0, 1.0, 1.0],
        #                             [1.0, 1.0, 1.0],
        #                             [1.0, 1.0, 1.0],
        #                             [1.0, 1.0, 1.0]], device=self.device)
        # g_rots      = torch.tensor([[1.0, 0.0, 0.0, 0.0],
        #                             [1.0, 0.0, 0.0, 0.0],
        #                             [1.0, 0.0, 0.0, 0.0],
        #                             [1.0, 0.0, 0.0, 0.0],
        #                             [1.0, 0.0, 0.0, 0.0]], device=self.device)
        # g_colors    = torch.tensor([[1.0, 0.0, 0.0],
        #                             [0.0, 1.0, 0.0],
        #                             [0.0, 0.0, 1.0],
        #                             [1.0, 1.0, 0.0],
        #                             [1.0, 0.0, 1.0]], device=self.device)

        # rgb, radii, depth_image = rasterizer(
        #     means3D        = g_poss,
        #     means2D        = viewspace_points,
        #     shs            = None,
        #     colors_precomp = g_colors,
        #     opacities      = g_opacities,
        #     scales         = g_scales,
        #     rotations      = g_rots
        # )

        visible_filter = (radii > 0).nonzero
        return rgb, viewspace_points, visible_filter