import numpy as np
import torch
import glm
    
class Camera():
    def __init__(self, out_img_size = (0,0), torch_device = torch.device("cuda")):
        self.device = torch_device
        # Output params
        self.h = int(out_img_size[0])
        self.w = int(out_img_size[1])

        if (self.h == 0):
            self.aspect_ratio = 0
        else:
            self.aspect_ratio = self.w / self.h
            
        self.cam_near = 0.01
        self.cam_far  = 100.0

        self.pos     = None
        self.tanfovx = None
        self.tanfovy = None
        self.view    = None
        self.proj    = None

        self.up     = None
        self.focus  = None
        
        self.cam_height_angle = None

    def setup_cam(self,
                  fov,
                  up,
                  pos,
                  focus):

        # Camera params        
        self.up    = glm.vec3(up)
        pos        = glm.vec3(pos)
        self.focus = glm.vec3(focus)

        self.cam_height_angle = fov * np.pi / 180

        self.tanfovy = np.tan(self.cam_height_angle/2)
        self.tanfovx = self.aspect_ratio * self.tanfovy

        self.view, self.proj = self.compute_camera_matrix(self.up, pos, self.focus)

        # Pos actually needs to be on the GPU
        self.pos = torch.tensor(pos.to_tuple(), device=self.device, dtype=torch.float32)

    def compute_camera_matrix(self, up : glm.vec3, pos : glm.vec3, focus : glm.vec3):
        # Construct the camera pose
        # Hack: we need to look down the +Z dimension (hence the '-'es)
        view = glm.lookAt(-pos, focus, up)
        view[3] *= -1
        view[3,3] = 1
        torch_view = torch.tensor(view.to_tuple(), device=self.device, dtype=torch.float32)

        proj_glm = glm.perspective(self.cam_height_angle, self.aspect_ratio, self.cam_near, self.cam_far)

        # Note: The rasterizer expects proj to be proj * view
        vp_glm = proj_glm * view
        torch_proj = torch.tensor(vp_glm.to_tuple(), device=self.device, dtype=torch.float32)

        return torch_view, torch_proj
    
    def set_position(self, new_pos):
        pos = glm.vec3(new_pos)
        self.view, self.proj = self.compute_camera_matrix(self.up, pos, self.focus)
        self.pos = torch.tensor(self.pos.to_tuple(), device=self.device, dtype=torch.float32)

    def setup_cam_from_view_and_proj(self, Rt : np.array, K : np.array):
        # Find the camera origin
        zeros_and_1 = np.array([0, 0, 0, 1])
        zeros_and_1 = np.expand_dims(zeros_and_1, axis=0)
        view_4x4 = np.concatenate((Rt, zeros_and_1), axis=0)
        view_inv = np.linalg.inv(view_4x4)
        cam_pos = view_inv @ np.array([0,0,0,1], dtype='float32')
        self.pos = torch.tensor(cam_pos[:3], device=self.device, dtype=torch.float32)
        view_4x4[2,:] *= -1 # HACK
        # View matrix (needs to be column major)
        self.view = torch.tensor(view_4x4, device=self.device, dtype=torch.float32).T

        # Projection params
        self.w = int(K[0,2]*2)
        self.h = int(K[1,2]*2)

        self.aspect_ratio = self.w/self.h

        # Recover fov (from Szeliski: tan(theta/2) = W/2f))
        f_w = K[0,0]
        f_h = K[1,1]
        self.tanfovx = self.w / (2 * f_w)
        self.tanfovy = self.h / (2 * f_h)

        #self.cam_height_angle = np.atan(self.tanfovy) * 2
        self.cam_height_angle = np.arctan(self.tanfovy) * 2
  
        proj_glm = glm.perspective(self.cam_height_angle, self.aspect_ratio, self.cam_near, self.cam_far)

        torch_proj = torch.tensor(proj_glm.to_tuple(), device=self.device, dtype=torch.float32)

        self.proj = torch_proj @ self.view
    
    def cam_from_viewer(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        print('initialzing camera is done')
        self.w = width
        self.h = height
        self.tanfovx = fovx
        self.tanfovy = fovy
        self.view = world_view_transform
        self.proj = full_proj_transform




    def log_camera_info(self):
        print(f"Position: {self.pos.cpu().numpy()}")
        print(f"Focus: {self.focus.cpu().numpy()}")
        print(f"Up: {self.up.cpu().numpy()}")
        print(f"Aspect Ratio: {self.aspect_ratio}")