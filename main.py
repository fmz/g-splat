from utils.dataloader import Dataset
from model.scene import Scene, BoundingBox
from model.camera import Camera
from model.rasterizer_wrapper import GausRast

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

hparams = {
    'learning_rate': 0.01,
    'num_epochs': 10,
    'regularization_weight': 0.1,
    'densification_interval':100,
    'densify_until_iteration':5
}

def g_splat():
    global hparams

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    data = Dataset("data/cube")

    camera = Camera(data.img_shape)
    camera.setup_cam(60, up=[0.0, 1.0, 0.0], pos=[0.0, 0.0, -10.0], focus=[0.0, 0.0, 0.0])

    bbox  = BoundingBox(lo=np.array([-15.0, -15.0, -15.0]), hi=np.array([15.0, 15.0, 15.0]))
    scene = Scene(bbox)

    rasterizer = GausRast()

    params = scene.get_optimizable_params()
    optimizer = optim.Adam(params, hparams["learning_rate"])
    loss_fn = nn.MSELoss()

    num_epochs = hparams["num_epochs"]
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        for i in range(data.num_images):
            camera     = data.cameras[i]
            target_img = data.images[i]


            # Rasterize the scene given a camera
            img_out, viewspace_points, visible_filter = rasterizer.forward(scene, camera)
            
            # Compute loss (rendering loss + regularization)
            render_loss = loss_fn(img_out, target_img)
            
            # Improved regularization loss
            #reg_loss = scene.regularization_loss() * hparams['regularization_weight']
            total_loss = render_loss # + reg_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            #Refinement Iteration
            if epoch < hparams["densify_until_iteration"]:
                if (epoch + 1) % hparams["densification_interval"] == 0:
                    scene.prune_and_densify(viewspace_points, visible_filter)


            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Render Loss: {render_loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

            # Optional: Save intermediate rendered images for debugging
            # if (epoch + 1) % 20 == 0:
            #     plt.imshow(img_out), f"output/render_epoch_{epoch + 1}.png")     
        

if __name__ == "__main__":
    g_splat()
