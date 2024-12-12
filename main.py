from utils.dataloader import Dataset
from model.scene import Scene, BoundingBox
from model.camera import Camera
from model.rasterizer_wrapper import GausRast

import torch

import numpy as np

def g_splat():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    data = Dataset("data/cube")

    camera = Camera(data.img_shape, 65)

    bbox  = BoundingBox(lo=np.array([-15.0, -15.0, -15.0]), hi=np.array([15.0, 15.0, 15.0]))
    scene = Scene(bbox)

    rasterizer = GausRast()

    rasterizer.forward(scene, camera)
    

if __name__ == "__main__":
    g_splat()
