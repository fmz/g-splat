import os
import torch
import pandas as pd
from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import v2
from fused_ssim import fused_ssim

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=(1080, 1920)),
    v2.ToDtype(torch.float32, scale=True),
])

img_file1 = "data/yosemite1.jpg"
img_file2 = "data/yosemite2.jpg"

if not torch.cuda.is_available():
    raise Exception("CUDA is required for fused-ssim")
 
device = torch.device("cuda")
print(f"Running model on device: {device}")

img1 = Image.open(img_file1).convert("RGB")
img1 = transform(img1)
img2 = Image.open(img_file2).convert("RGB")
img2 = transform(img2)

img1 = img1.unsqueeze(0)
img2 = img2.unsqueeze(0)

#dataset = CustomImageDataset(img_dir=img_dir, transform=transform)
#dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

img1 = img1.to(device)
img2 = img2.to(device)

ssim_value = fused_ssim(img1, img2)

print(ssim_value)