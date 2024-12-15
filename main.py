from utils.dataloader import Dataset
from model.scene import Scene, BoundingBox
from model.camera import Camera
from model.rasterizer_wrapper import GausRast
from fused_ssim import fused_ssim

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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
    'densification_interval':5,
    'densify_until_iteration':20,
    'dssim_scale':0.2
}

def g_splat():
    global hparams

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    data = Dataset("data/cube")

    #observer = Camera(data.img_shape[1:])
    observer = Camera([1080,1920])
    observer.setup_cam(60, up=[0.0, 1.0, 0.0], pos=[0.0, 0.0, -10.0], focus=[0.0, 0.0, 0.0])

    bbox  = BoundingBox(lo=np.array([-5.0, -5.0, -5.0]), hi=np.array([5.0, 5.0, 5.0]))
    scene = Scene(bbox)

    rasterizer = GausRast()

    # Optimizer & loss setup
    scene.init_optimizer(hparams['lrs'])
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
        scene.optimizer.zero_grad(set_to_none=True)

        for i in range(data.num_images):
            camera     = data.cameras[i]
            target_img = data.images[i]

            # Rasterize the scene given a camera
            img_out, viewspace_points, visible_filter = rasterizer.forward(scene, observer)

            if (i == 0) and ((epoch + 1) % 25 == 0 or epoch == 0):
                rgb, _, _ = rasterizer.forward(scene, observer)

                rgb = rgb.cpu().detach()
                rgb = rgb.permute((1,2,0))
                plt.imshow(rgb)
                plt.show()

            # Compute loss (rendering loss + dssim)
            ssim_loss  = 1.0 - fused_ssim(img_out.unsqueeze(0), img1.unsqueeze(0))

            l1_loss = loss_fn(img_out, img1)

            # Improved regularization loss
            #reg_loss = scene.regularization_loss() * hparams['regularization_weight']
            total_loss = (1.0 - dssim_scale) * l1_loss + dssim_scale * ssim_loss # + reg_loss

            # Backward pass
            total_loss.backward()
            scene.optimizer.step()

            #Refinement Iteration
            #scene.add_densification_data(viewspace_points, visible_filter)
            #if epoch < hparams["densify_until_iteration"]:
            #    if (epoch + 1) % hparams["densification_interval"] == 0:
            #        scene.prune_and_densify()


            # Logging
            if (i==0) and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"Epoch {epoch + 1}/{num_epochs}, L1 Loss: {l1_loss.item():.4f}, SSIM Loss: {ssim_loss.item():.4f} Total Loss: {total_loss.item():.4f}")

            # Optional: Save intermediate rendered images for debugging
            # if (epoch + 1) % 20 == 0:
            #     plt.imshow(img_out), f"output/render_epoch_{epoch + 1}.png")     
        

if __name__ == "__main__":
    g_splat()
