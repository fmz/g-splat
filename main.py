from utils.dataloader import Dataset
from utils.dataloader_glomap import Dataset_Colmap
from utils.dataloader_dummy import DatasetDummy
from model.scene import Scene, BoundingBox
from model.camera import Camera
from model.rasterizer_wrapper import GausRast
from fused_ssim import fused_ssim

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import argparse
import os

# Maybe makes pytorch more memory friendly?
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
    'densification_interval': 2,
    'opacity_reset_interval': 30,
    'densify_until_epoch':70,
    'dssim_scale':0.2
}

def g_splat(args):
    global hparams

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    # Right now we have a very crude bounding box to work with.
    # This needs to be improved!
    if args.mode == 'blender':
        bbox  = BoundingBox(lo=np.array([-2.0, -2.0, -2.0]), hi=np.array([2.0, 2.0, 2.0]))
        dataset = Dataset(args.data) # e.g. data/cube or data/monkey
        scene = Scene(bbox,init_method="random")
    elif args.mode == 'colmap':
        bbox  = BoundingBox(lo=np.array([-10.0, -10.0, -10.0]), hi=np.array([10.0, 10.0, 10.0]))
        image_txt = os.path.join(args.data, "images/images.txt")
        camera_txt = os.path.join(args.data, "images/cameras.txt")
        point_txt = os.path.join(args.data, "images/points3D.txt")
        image_path = os.path.join(args.data, "images/input")

        dataset = Dataset_Colmap(image_txt,camera_txt,image_path) # e.g. data/db/drjohnson
        scene = Scene(bbox,init_method="from-dataset",points_txt=point_txt)
    elif args.mode == 'singleimg':
        bbox  = BoundingBox(lo=np.array([-5.0, -10.0, -10.0]), hi=np.array([5.0, 5.0, 5.0]))

        dataset = DatasetDummy(args.data) # e.g. data/yosemite1.jpg
        scene = Scene(bbox,init_method="random")

        hparams['num_epochs'] *= 10
    else:
        raise RuntimeError(f"Unknown mode {args.mode}")

    # Didn't hook up the visualizer yet, so all you get is a couple random cameras (:
    observer = Camera((720,1280))
    observer.setup_cam(60, up=[0.0, 1.0, 0.0], pos=[0.0, 0.0, 5.0], focus=[0.0, 0.0, 0.0])

    observer2 = Camera((720,1280))
    observer2.setup_cam(60, up=[0.0, 1.0, 0.0], pos=[0.0, 0.0, -5.0], focus=[0.0, 0.0, 0.0])

    rasterizer = GausRast()

    # Optimizer & loss setup
    scene.init_optimizer(hparams['lrs'])
    scene.optimizer.zero_grad(set_to_none=True)

    # Learning rate is specified for each param separately
    loss_fn = nn.L1Loss()
    dssim_scale = hparams['dssim_scale']

    # Train
    num_epochs = hparams["num_epochs"]
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch}")

        for i in range(dataset.num_images):
            camera     = dataset.cameras[i]
            target_img = dataset.images[i]

            # Rasterize the scene given a camera
            img_out, viewspace_points, visible_filter, radii = rasterizer.forward(scene, camera)

            # Compute loss (rendering loss + dssim)
            ssim_loss  = 1.0 - fused_ssim(img_out.unsqueeze(0), target_img.unsqueeze(0))

            l1_loss = loss_fn(img_out, target_img)

            # Improved regularization loss
            total_loss = (1.0 - dssim_scale) * l1_loss + dssim_scale * ssim_loss # + reg_loss
            # Backward pass
            total_loss.backward()

            with torch.no_grad():
                # Viz
                if (i == 0) and ((epoch + 1) % args.viz_interval == 0 or epoch == 0):
                    rgb, _, _, _= rasterizer.forward(scene, observer)

                    rgb = rgb.cpu().detach()
                    rgb = rgb.permute((1,2,0))
                    plt.imshow(rgb)
                    plt.show()
                    
                    rgb, _, _, _= rasterizer.forward(scene, observer2)

                    rgb = rgb.cpu().detach()
                    rgb = rgb.permute((1,2,0))
                    plt.imshow(rgb)
                    plt.show()
                    # plt.imsave(f'outputs/drjohnson2/{epoch}_100.png', np.clip(rgb.numpy(), 0, 1))


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

            torch.cuda.empty_cache()

            # Optional: Save intermediate rendered images for debugging
            # if (epoch + 1) % 20 == 0:
            #     plt.imshow(img_out), f"output/render_epoch_{epoch + 1}.png")

    with torch.no_grad():
        # Viz-pos
        for i, camera in enumerate(dataset.cameras):
            rgb, _, _, _= rasterizer.forward(scene, camera)

            rgb = rgb.cpu().detach()
            rgb = rgb.permute((1,2,0))
            plt.imsave(f'outputs/drjohnson2/final_{i}.png', np.clip(rgb.numpy(), 0, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='G-Splat: Gaussian Splatting, but worse')
    parser.add_argument('--mode', choices=['singleimg', 'blender', 'colmap'], default='colmap')
    parser.add_argument('--data', default='/data/db/drjohnson/')
    parser.add_argument('--viz_interval', type=int, default=int(5))  # in epochs

    args = parser.parse_args()

    g_splat(args)
