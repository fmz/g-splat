from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from model.camera import Camera
from model.scene import Scene

import matplotlib.pyplot as plt
import torch

class GausRast():
    def __init__(self, torch_device = torch.device("cuda")):
        self.device = torch_device

        self.bg_clr = torch.tensor([1.0,1.0,1.0], device=self.device, dtype=torch.float32)

    def forward(self, scene : Scene, camera : Camera):

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

        rgb, radii, depth = rasterizer(
            means3D=scene.points,
            means2D=None,
            shs=None,
            colors_precomp=scene.colors,
            opacities=scene.opacities,
            scales=scene.scales,
            rotations=scene.rots,
        )

        disp_img = rgb.cpu()
        disp_img = disp_img.permute((1,2,0))
        plt.imshow(disp_img)
        plt.show()

        return rgb