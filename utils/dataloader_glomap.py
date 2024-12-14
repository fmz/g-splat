 if colmap_inputs== 'true':
            images_info = get_colmap_images_info(img_txt)
            camera_info = get_colmap_camera_info(camera_txt)
            k_matrix = build_k_matrix(camera_info)
            point_info = read_points3D_text(points_txt)
            extrinsic_matricies, rotation_matricies, translation_vectors = build_extrinsic_per_image(image_info)
            self.num_images = len(images_info)

            self.cam_r = rotation_matricies
            self.cam_K = k_matrix
            self.cam_t = translation_vectors
            
            for i in range(self.num_images):
                R_inv = torch.inverse(self.cam_R[i])
                t_inv = -self.cam_t[i]
                self.cam_pos[i] = (R_inv @ t_inv).squeeze()
            self.cameras = []
            for i in range(self.num_images):
                cam = Camera()
                cam.setup_cam_from_view_and_proj(
                    torch.cat((self.cam_R[i], self.cam_t[i]), dim=1).cpu().detach().numpy(),
                    self.cam_K[i].cpu().detach().numpy()
                )
                self.cameras.append(cam)
from PIL import Image
import numpy as np
import torch

import os
import glob

from model.camera import Camera

class Dataset():
    # The dataset format is detailed in https://vision.middlebury.edu/mview/data/
    # The most important part is:

    # name_par.txt: camera parameters. There is one line for each image.
    # The format for each line is:
    # "imgname.png k11 k12 k13 k21 k22 k23 k31 k32 k33 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3".
    # The projection matrix for that image is K*[R t]. The image origin is top-left, with x increasing horizontally, y vertically.

    def __init__(colmap_inputs ='false', img_txt,camera_txt,points_txt):
       :
        self.device = torch_device

        images_info = get_colmap_images_info(img_txt)
        camera_info = get_colmap_camera_info(camera_txt)
        k_matrix = build_k_matrix(camera_info)
        point_info = read_points3D_text(points_txt)
        extrinsic_matricies, rotation_matricies, translation_vectors = build_extrinsic_per_image(image_info)
        self.num_images = len(images_info)

        self.cam_r = rotation_matricies
        self.cam_K = k_matrix
        self.cam_t = translation_vectors
        
        self.cameras = []
        for i in range(self.num_images):
            cam = Camera()
            cam.setup_cam_from_view_and_proj(
                torch.cat((self.cam_R[i], self.cam_t[i]), dim=1).cpu().detach().numpy(),
                self.cam_K[i].cpu().detach().numpy()
            )
            self.cameras.append(cam)
            

           