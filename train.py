import os
import torch
import pandas as pd
from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
#from torch.utils.data import Dataset, DataLoader
from fused_ssim import fused_ssim

from utils.dataloader import Dataset
from model.scene import Scene, BoundingBox
from model.camera import Camera
from model.rasterizer_wrapper import GausRast


device = torch.device('cuda')

data = Dataset("data/cube")

    

#Initialize gaussians based on point cloud/randomly

#Until converged
converged = False
while not converged:

    #Get random training camera (V) + image (I)
    camera = Camera(data.img_shape, 65)

    
    train_image = pass

    #Use rasterizer + current gaussians to generate output image w/ same camera (I')
    bbox  = BoundingBox(lo=np.array([-15.0, -15.0, -15.0]), hi=np.array([15.0, 15.0, 15.0]))
    scene = Scene(bbox)

    rasterizer = GausRast()
    
    test_image = rasterizer.forward(scene, camera)

    #Find Loss
    ssim_value = fused_ssim(train_image, test_image)

    #Backprop

    #Every x iterations, refine gaussians

        #Remove Big Gaussians

        #If view space position gradients above threshold

            #SplitGaussian if over-reconstructed

            #CloneGaussian if under-reconstructed
