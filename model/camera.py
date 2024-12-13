import numpy as np
import torch
import glm
    
class Camera():
    def __init__(self, 
                 out_img_size, 
                 fov,
                 torch_device = torch.device("cuda"), 
                 up = [0.0, 1.0, 0.0], 
                 pos = [-20.0, -20.0, -20.0]):
        self.device = torch_device
        # Output params
        self.h = out_img_size[0]
        self.w = out_img_size[1]

        # Camera params
        self.aspect_ratio = self.w / self.h
        self.fov = fov
        
        self.up    = glm.vec3(up)
        self.pos   = glm.vec3(pos)
        self.focus = glm.vec3(0.0, 0.0, 0.0)

        self.cam_height_angle = self.fov * np.pi / 180

        self.view, self.proj = self.compute_camera_matrix()

        # Pos actually needs to be on the GPU
        self.pos = torch.tensor(self.pos.to_tuple(), device=self.device, dtype=torch.float32)

        self.tanfovy = np.tan(self.cam_height_angle/2)
        self.tanfovx = self.aspect_ratio * self.tanfovy

    def compute_camera_matrix(self):
        # Construct the camera pose
        cam_near = 0.01
        cam_far  = 100.0

        view = glm.lookAt(self.pos, self.focus, self.up)
        # hack: We look down the +Z direction
        view[3, 2] *= -1
        
        torch_view = torch.tensor(view.to_tuple(), device=self.device, dtype=torch.float32)

        proj_glm = glm.perspective(self.cam_height_angle, self.aspect_ratio, cam_near, cam_far)

        # Note: The rasterizer expects proj to be proj * view
        vp_glm = proj_glm * view
        torch_proj = torch.tensor(vp_glm.to_tuple(), device=self.device, dtype=torch.float32)

        return torch_view, torch_proj
