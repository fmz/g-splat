from PIL import Image
import numpy as np
import torch

import os
import glob

from model.camera import Camera
from utils.get_data_col import get_colmap_camera_info,get_colmap_images_info,build_k_matrix,build_extrinsic_per_image

class Dataset_Colmap():
    # The dataset format is detailed in https://vision.middlebury.edu/mview/data/
    # The most important part is:

    # name_par.txt: camera parameters. There is one line for each image.
    # The format for each line is:
    # "imgname.png k11 k12 k13 k21 k22 k23 k31 k32 k33 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3".
    # The projection matrix for that image is K*[R t]. The image origin is top-left, with x increasing horizontally, y vertically.

    def __init__(self, img_txt,camera_txt,image_path,  torch_device=torch.device('cuda')):
        self.device = torch_device
        images_info = get_colmap_images_info(img_txt)
        camera_info = get_colmap_camera_info(camera_txt)
        k_matrix = build_k_matrix(camera_info)
        extrinsic_matricies, rotation_matricies, translation_vectors,images = build_extrinsic_per_image(images_info)
        self.num_images = len(images_info)

        # self.cam_K   = torch.zeros((self.num_images, 3, 3), device=self.device, dtype=torch.float32)
        # self.cam_R   = torch.zeros((self.num_images, 3, 3), device=self.device, dtype=torch.float32)
        # self.cam_t   = torch.zeros((self.num_images, 3, 1), device=self.device, dtype=torch.float32)
        # self.cam_mat = torch.zeros((self.num_images, 3, 4), device=self.device, dtype=torch.float32)
        # self.cam_pos = torch.zeros((self.num_images, 3), device=self.device, dtype=torch.float32)
        self.images = []
        
        self.cam_R = torch.tensor(rotation_matricies, device = self.device , dtype=torch.float32)
        self.cam_K = torch.tensor(k_matrix, device = self.device , dtype=torch.float32)
        self.cam_t = torch.tensor(translation_vectors, device = self.device , dtype=torch.float32)
        for image_name in images:
            #print(image_name)
            if not image_name.endswith(".jpg"):
                image_name = image_name+".jpg"
            img_path = image_path+"/"+image_name
            image = Image.open(img_path)
                # The current dataset has images that have alpha in them
            image = np.array(image, dtype='float32')
            image = image[:,:,:3]
            image /= 255.0
            # Convert to CHW
            image = np.permute_dims(image, (2,0,1))
            image = torch.tensor(image, device=self.device, dtype=torch.float32)
            image = image / 255.0
            image = image.to(self.device)

            self.images.append(image)
        self.img_shape = self.images[0].shape
        width = camera_info['width']
        height = camera_info['height']
        self.cameras = []
        for i in range(self.num_images):
            cam = Camera()
            cam.glo_map_setup_cam_from_view_and_proj(
                torch.cat((self.cam_R[i], self.cam_t[i]), dim=1).cpu().detach().numpy(),
                self.cam_K.cpu().detach().numpy(), width, height
            )
            self.cameras.append(cam)
            

           