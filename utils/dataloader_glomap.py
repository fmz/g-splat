from PIL import Image
import numpy as np
import torch

import os
import glob

from model.camera import Camera

class Dataset_Colmap():
    # The dataset format is detailed in https://vision.middlebury.edu/mview/data/
    # The most important part is:

    # name_par.txt: camera parameters. There is one line for each image.
    # The format for each line is:
    # "imgname.png k11 k12 k13 k21 k22 k23 k31 k32 k33 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3".
    # The projection matrix for that image is K*[R t]. The image origin is top-left, with x increasing horizontally, y vertically.

    def __init__( img_txt,camera_txt, image_path):
       :
        self.device = torch_device
        self.cam_r = []
        self.cam_k = []
        self.cam_t = []
        images_info = get_colmap_images_info(img_txt)
        camera_info = get_colmap_camera_info(camera_txt)
        self.img_shape = (camera_info["width"], camera_info["height"])
        k_matrix = build_k_matrix(camera_info)
        extrinsic_matricies, rotation_matricies, translation_vectors = build_extrinsic_per_image(image_info)
        self.num_images = len(images_info)
        for i in len(num_images):

            self.cam_r[i] = torch.tensor(rotation_matricies[i], device = self.device , dtype=torch.float32)
            self.cam_K[i] = torch.tensor(k_matrix, device = self.device , dtype=torch.float32)
            self.cam_t[i] = torch.tensor(translation_vector[i], device = self.device , dtype=torch.float32)
            self.images[i] = image_path+"/"+images_info[i]['image_name']

        self.cameras = []
        for i in range(self.num_images):
            cam = Camera()
            cam.setup_cam_from_view_and_proj(
                torch.cat((self.cam_R[i], self.cam_t[i]), dim=1).cpu().detach().numpy(),
                self.cam_K[i].cpu().detach().numpy()
            )
            self.cameras.append(cam)
            

           