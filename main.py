from utils.dataloader import Dataset
from utils.dataloader_glomap import Dataset_Colmap
from model.scene import Scene, BoundingBox
from model.camera import Camera
from model.rasterizer_wrapper import GausRast
from fused_ssim import fused_ssim

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from random import randint

learning_rates = {
    'opacity': 0.05,
    'scale': 0.005,
    'rotation': 0.001,
    'position': 0.0001,
    'color': 0.0025

}

hparams = {
    'lrs': learning_rates,
    'num_epochs': 100,
    'regularization_weight': 0.01,
    'densification_interval': 1,
    'opacity_reset_interval': 30,
    'densify_until_epoch':50,
    'dssim_scale':0.2
}

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def g_splat():
    global hparams

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    #data = Dataset("data/cube")

    # image_txt = "data/db/drjohnson/images/images.txt"
    # camera_txt = "data/db/drjohnson/images/cameras.txt"
    # point_txt = "data/db/drjohnson/images/points3D.txt"
    # image_path = "data/db/drjohnson/images/input"
    # dataset = Dataset_Colmap(image_txt,camera_txt,image_path)
    dataset = Dataset('data/monkey')

    observers = []

    for i in range (0, 360):
        radius = 5.0
        observer = Camera(dataset.img_shape[1:])
        observer = Camera((720,1280))
        x = radius * np.cos(np.deg2rad(i))
        z = radius * np.sin(np.deg2rad(i))
        observer.setup_cam(60, up=[0.0, 1.0, 0.0], pos=[x, 0.0, z], focus=[0.0, 0.0, 0.0])
        observers.append(observer)
     


    #observer2 = Camera((1080,1920))
    #observer2.setup_cam(60, up=[0.0, 1.0, 0.0], pos=[0.0, 0.0, -5.0], focus=[0.0, 0.0, 0.0])


    bbox  = BoundingBox(lo=np.array([-2.0, -2.0, -2.0]), hi=np.array([2.0, 2.0, 2.0]))
    #scene = Scene(bbox,init_method="from-dataset",points_txt=point_txt)
    scene = Scene(bbox,init_method="random")


    rasterizer = GausRast()

    # Optimizer & loss setup
    scene.init_optimizer(hparams['lrs'])
    scene.optimizer.zero_grad(set_to_none=True)

    # Learning rate is specified for each param separately
    loss_fn = nn.L1Loss()
    dssim_scale = hparams['dssim_scale']


    ########## TEST
    from PIL import Image
    '''
    from torchvision.transforms import v2

    
    img_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(1080, 1920)),
        v2.ToDtype(torch.float32, scale=True),
    ])
    '''

    img_file1 = "data/yosemite1.jpg"

    if not torch.cuda.is_available():
        raise Exception("CUDA is required for fused-ssim")
    
    device = torch.device("cuda")
    print(f"Running model on device: {device}")

    img1 = Image.open(img_file1).convert("RGB")
    newsize = (1920, 1080)
    img1 = img1.resize(newsize)

    #New v2 img1 loading
    #img1 = img_transform(img1)

    #Old version img1 loading
    img1 = np.array(img1, dtype='float32')

    #img1 = np.permute_dims(img1, (2,0,1))
    img1 = np.transpose(img1, (2,0,1))
    img1 = torch.tensor(img1, device=device, dtype=torch.float32)
    img1 = img1 / 255.0
    #############


    # Train
    num_epochs = hparams["num_epochs"]
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch}")

        for i in range(dataset.num_images):
            camera     = dataset.cameras[i]
            target_img = dataset.images[i]

            # Rasterize the scene given a camera
            img_out, viewspace_points, visible_filter, radii = rasterizer.forward(scene, camera)

            # rgb = img_out.cpu().detach()
            # rgb = rgb.permute((1,2,0))
            # plt.imshow(rgb)
            # plt.show()
            # rgb = target_img.cpu().detach()
            # rgb = rgb.permute((1,2,0))
            # plt.imshow(rgb)
            # plt.show()

            # Compute loss (rendering loss + dssim)
            ssim_loss  = 1.0 - fused_ssim(img_out.unsqueeze(0), target_img.unsqueeze(0))

            l1_loss = loss_fn(img_out, target_img)

            # Improved regularization loss
            #reg_loss = scene.regularization_loss() * hparams['regularization_weight']
            total_loss = (1.0 - dssim_scale) * l1_loss + dssim_scale * ssim_loss # + reg_loss
            # Backward pass
            total_loss.backward()

            with torch.no_grad():
                # Viz
                #if (i == 0) and ((epoch + 1) % 5 == 0 or epoch == 0):
                cam_id = ((epoch * dataset.num_images) + i) % 360
                rgb, _, _, _= rasterizer.forward(scene, observers[cam_id])

                rgb = rgb.cpu().detach()
                rgb = rgb.permute((1,2,0))
                plt.imshow(rgb)
                #plt.show()
                plt.savefig(f"outputs/monkey_test/{(epoch * dataset.num_images) + i}")

                #Refinement Iteration
                #scene.max_radii = torch.max(scene.max_radii, radii)
                scene.add_densification_data(viewspace_points, visible_filter)
                if epoch < hparams["densify_until_epoch"]:
                    if ((epoch + 1) % hparams["densification_interval"] == 0) and i == dataset.num_images - 1:
                        scene.prune_and_densify()
                    
                if ((epoch + 1) % hparams["opacity_reset_interval"] == 0) and i == dataset.num_images - 1:
                    scene.reset_opacities()

                scene.optimizer.step()
                scene.optimizer.zero_grad(set_to_none=True)

            # Logging
            if (i==0) and ((epoch + 1) % 1 == 0 or epoch == 0):
                print(f"Epoch {epoch + 1}/{num_epochs}, L1 Loss: {l1_loss.item():.4f}, SSIM Loss: {ssim_loss.item():.4f} Total Loss: {total_loss.item():.4f}, Num Gaussians: {scene.optimizer.param_groups[0]['params'][0].shape}")

            # Optional: Save intermediate rendered images for debugging
            # if (epoch + 1) % 20 == 0:
            #     plt.imshow(img_out), f"output/render_epoch_{epoch + 1}.png")     
        

if __name__ == "__main__":
    g_splat()
