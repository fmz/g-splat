from PIL import Image
import numpy as np
import torch

import os
import glob

from model.camera import Camera

class DatasetDummy():
    def __init__(self, img_filename, torch_device=torch.device('cuda')):
        self.device = torch_device

        self.num_images = 1

        # Load and preprocess

        img = Image.open(img_filename).convert("RGB")
        newsize = (1920, 1080)
        img = img.resize(newsize)

        '''
        For newer pytorch:
        from torchvision.transforms import v2


        img_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(1080, 1920)),
            v2.ToDtype(torch.float32, scale=True),
        ])
        img = img_transform(img)
        '''
        # Uncommment the above and comment out the below for older pytorch
        img = np.array(img, dtype='float32')

        img = np.transpose(img, (2,0,1))
        img = torch.tensor(img, device=self.device, dtype=torch.float32)
        img = img / 255.0

        self.images = [img,]
        #############

        self.cameras = [Camera(newsize[::-1]),]
        self.cameras[0].setup_cam(60, up=[0.0, 1.0, 0.0], pos=[0.0, 0.0, 5.0], focus=[0.0, 0.0, 0.0])
