import os
import torch
import numpy as np
import glm
import matplotlib.pyplot as plt

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def build_quaternion(r):
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

def make_homogeneous(unhom_pts):
    ones = torch.ones(unhom_pts.shape[0], 1)
    hom_pts = torch.concatenate((unhom_pts, ones), axis=1)
    return hom_pts

def test_renderer():
    device = torch.device('cuda')

    # Scene params
    image_height = 720
    image_width = 1280

    # Camera params
    aspect_ratio = image_width / image_height
    cam_height_angle = 100.0 * np.pi / 180

    tan_height_angle_by_2 = np.tan(cam_height_angle/2)
    tan_width_angle_by_2  = aspect_ratio * tan_height_angle_by_2

    # Camera pose
    cam_up       = glm.vec3(0.0, 1.0, 0.0)
    cam_pos      = glm.vec3(0.0, 0.0, 10.0)
    cam_focus_on = glm.vec3(0.0, 0.0, 0.0)

    cam_near = 0.01
    cam_far  = 100.0
    
    cam_top = tan_height_angle_by_2 * cam_near
    cam_bottom = -cam_top
    cam_right = tan_width_angle_by_2 * cam_near
    cam_left = -cam_right

    view = glm.lookAt(cam_pos, cam_focus_on, cam_up)
    # hack:
    view[3, 2] *= -1
    view_torch = torch.tensor(view.to_tuple(), device=device)

    proj_glm = glm.perspective(cam_height_angle, aspect_ratio, cam_near, cam_far)

    vp_glm = proj_glm * view
    vp_torch = torch.tensor(glm.transpose(vp_glm).to_tuple(), device=device)

    raster_settings = GaussianRasterizationSettings(
        image_height   = int(image_height),
        image_width    = int(image_width),
        tanfovx        = float(tan_width_angle_by_2),
        tanfovy        = float(tan_height_angle_by_2),
        bg             = torch.tensor([0.0,0.0,0.0], device=device),
        scale_modifier = 1.0,
        viewmatrix     = view_torch,
        projmatrix     = vp_torch,
        sh_degree      = 0,   # Figure this out
        campos         = torch.tensor(cam_pos.to_tuple(), device=device),
        prefiltered    = False,
        debug          = True,
        antialiasing   = True
    )

    rasterizer = GaussianRasterizer(raster_settings)

    # A couple test gaussians:
    g_poss      = torch.tensor([[0.0, 0.0, 0.0],
                                [0.0, 2.0, 0.0],
                                [0.0, -4.0, 0.0],
                                [6.0, 0.0, 0.0],
                                [-8.0, 0.0, 0.0]], device=device)
    g_opacities = torch.tensor([[1.0],
                                [1.0],
                                [1.0],
                                [1.0],
                                [1.0]], device=device)
    g_scales    = torch.tensor([[1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0]], device=device)
    g_rots      = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0, 0.0]], device=device)
    g_colors    = torch.tensor([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [1.0, 1.0, 0.0],
                                [1.0, 0.0, 1.0]], device=device)

    rendered_image, radii, depth_image = rasterizer(
        means3D        = g_poss,
        means2D        = None,
        shs            = None,
        colors_precomp = g_colors,
        opacities      = g_opacities,
        scales         = g_scales,
        rotations      = g_rots
    )

    disp_img = rendered_image.cpu()
    disp_img = disp_img.permute((1,2,0))
    plt.imshow(disp_img)
    plt.show()


if __name__ == '__main__':
    test_renderer()