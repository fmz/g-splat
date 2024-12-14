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

    def __init__(self, data_dir, torch_device=torch.device('cuda')):
        self.device = torch_device

        if os.path.isdir(data_dir):
            print(f'Loading data from {data_dir}...')
            self.data_dir = data_dir
        else:
            raise ValueError(f'The given directiory {data_dir} does not exist!')
        
        par_file = glob.glob(self.data_dir+"/*_par.txt")
        if len(par_file) != 1:
            raise ValueError(f'The given directiory {data_dir} does not have a par file!')

        self.par_file = par_file[0]

        self.images = []

        # Read the images
        with open(self.par_file, encoding='utf8') as file:
            lines = file.readlines()
            self.num_images = int(lines[0].strip())

            self.cam_K   = torch.zeros((self.num_images, 3, 3), device=self.device, dtype=torch.float32)
            self.cam_R   = torch.zeros((self.num_images, 3, 3), device=self.device, dtype=torch.float32)
            self.cam_t   = torch.zeros((self.num_images, 3, 1), device=self.device, dtype=torch.float32)
            self.cam_mat = torch.zeros((self.num_images, 3, 4), device=self.device, dtype=torch.float32)
            self.cam_pos = torch.zeros((self.num_images, 3), device=self.device, dtype=torch.float32)

            for i, line in enumerate(lines[1:]):
                args = line.split()
                img_path = os.path.join(self.data_dir, args[0])
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

                if i == 1:
                    self.img_shape = self.images[-1].shape

                self.cam_K[i] = torch.tensor(np.array(args[1:10], dtype=np.float32).reshape(3, 3), device=self.device)
                R = np.array(args[10:19], dtype=np.float32).reshape(3, 3)
                R = R[:, [1,2,0]]
                self.cam_R[i] = torch.tensor(R, device=self.device)
                self.cam_t[i] = torch.tensor(np.array(args[19:], dtype=np.float32).reshape(3, 1), device=self.device)
                self.cam_mat[i] = self.cam_K[i] @ torch.cat((self.cam_R[i], self.cam_t[i]), dim=1)

                R_inv = torch.inverse(self.cam_R[i])
                t_inv = -self.cam_t[i]
                self.cam_pos[i] = (R_inv @ t_inv).squeeze()
        
        print(f"Loaded {self.num_images} images and camera parameters.")

        # Create a camera for each image
        self.cameras = []
        for i in range(self.num_images):
            cam = Camera()
            cam.setup_cam_from_view_and_proj(
                torch.cat((self.cam_R[i], self.cam_t[i]), dim=1).cpu().detach().numpy(),
                self.cam_K[i].cpu().detach().numpy()
            )
            self.cameras.append(cam)