from utils.dataloader import Dataset
from model.scene import Scene, BoundingBox
from model.camera import Camera
from model.rasterizer_wrapper import GausRast
from fused_ssim import fused_ssim
from utils import network_gui
import traceback
import socket

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
    'densification_interval':1000,
    'densify_until_iteration':5,
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
    params = scene.get_optimizable_params(hparams['lrs'])
    # Learning rate is specified for each param separately
    optimizer = optim.Adam(params, 0.0)
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

    #img1 = img_transform(img1)

    #Old version img1 loading
    img1 = np.array(img1, dtype='float32')
    #img1 = img1[:,:,:3]
    #img1 /= 255.0
    # Convert to CHW
    #img1 = np.permute_dims(img1, (2,0,1))
    img1 = np.transpose(img1, (2,0,1))
    img1 = torch.tensor(img1, device=device, dtype=torch.float32)
    img1 = img1 / 255.0
    #############

    source_path = '/home/arinidhant/workspace/cv/tandt/truck'

    # Train
    num_epochs = hparams["num_epochs"]
    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)

        for i in range(data.num_images):
            camera     = data.cameras[i]
            target_img = data.images[i]

            # Rasterize the scene given a camera
            img_out, viewspace_points, visible_filter = rasterizer.forward(scene, observer)

            # if (i == 0) and ((epoch + 1) % 25 == 0 or epoch == 0):
                # rgb, _, _ = rasterizer.forward(scene, observer)
                # rgb = rgb.cpu().detach()
                # rgb = rgb.permute((1,2,0))

            # net_image_clamped = rgb.clamp(0, 1) 
            # net_image_scaled = (net_image_clamped * 255).byte()

            # # if net_image_scaled.shape[0] == 3:  # Check if channels are first (C, H, W)
            # #     net_image_scaled = net_image_scaled.permute(1, 2, 0)  # Rearrange to (H, W, C)

            # # Convert to numpy array for compatibility
            # raw_image = net_image_scaled.contiguous().cpu().numpy()  # Convert to numpy

            # # Handle only RGB channels if alpha exists
            # raw_image_rgb = raw_image[..., :3]  # Discard alpha if present

            # if network_gui.conn == None:
            #     network_gui.try_connect()
            #     print(network_gui.conn)
            # rgb = img_out.cpu().detach()  # Move tensor to CPU and detach it from computation graph
            # rgb = torch.clamp(rgb, min=0, max=1)  # Clamp values to [0, 1]
            # rgb = (rgb * 255).byte()  # Scale to [0, 255] and convert to byte

            # # If the tensor is in (C, H, W) format, permute to (H, W, C)
            # if rgb.shape[0] == 3:  # Assuming 3 channels means (C, H, W)
            #     rgb = rgb.permute(1, 2, 0)

            # # Convert to contiguous Numpy array
            # raw_image_rgb = rgb.contiguous().cpu().numpy()

            # # Convert to memoryview
            # net_image_bytes = memoryview(raw_image_rgb)
            # source_path = '/home/arinidhant/workspace/cv/tandt/truck'
            # print('Success connection')
            # network_gui.send(net_image_bytes, source_path)




            # rgb = rgb.cpu().detach()
            # rgb = rgb.permute((1,2,0))
            # #     plt.imshow(rgb)
            # #     plt.show()
            # # net_image = rasterizer.forward(scene, custom_cam)
            # # raw_image = (net_image.clamp(0, 1) * 255).byte().contiguous().cpu().numpy()
            # # Scale to [0, 255]
            # raw_image_rgb = rgb[..., :3]  # Ensure only R, G, B channels are included if img has an alpha channel
            # net_image_bytes = memoryview((torch.clamp(rgb, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                
            
            if network_gui.conn == None:
                network_gui.try_connect()
                print(network_gui.conn)
            
            while network_gui.conn != None:
                try:
                    print("We are starting the while loop here)")

                    print("We are receiving here)")
                    
                    net_image_bytes = None
                    
                    custom_cam, do_training, p2, p3, keep_alive, scaling_modifer = network_gui.receive()
                    print("Connection status" + str(network_gui.conn))
                    print("We have received camera and info receiving here)")

                    if custom_cam != None:
                        print("We are sending data)))))")
                        rgb, _, _ = rasterizer.forward(scene, observer)
                        # raw_image = (net_image.clamp(0, 1) * 255).byte().contiguous().cpu().numpy()
                        print("We are in here))))")

                        rgb = torch.clamp(rgb, min=0, max=1)  # Clamp values to [0, 1]
                        rgb = rgb.byte().permute(1,2,0).contiguous()
                        # rgb = rgb.cpu().detach()  # Move tensor to CPU and detach it from computation graph
                    
                        # rgb = (rgb * 255).byte()  # Scale to [0, 255] and convert to byte

                        # # If the tensor is in (C, H, W) format, permute to (H, W, C)
                        # if rgb.shape[0] == 3:  # Assuming 3 channels means (C, H, W)
                        #     rgb = rgb.permute(1, 2, 0)

                        # # Convert to contiguous Numpy array
                        raw_image_rgb = rgb.cpu().numpy()

                        # Convert to memoryview
                        net_image_bytes = memoryview(raw_image_rgb)


                        # rgb = rgb.cpu().detach()
                        # rgb = rgb.permute((1,2,0))
                        # raw_image_rgb = rgb[..., :3]  # Ensure only R, G, B channels are included if img has an alpha channel

                        # net_image_bytes = memoryview(raw_image_rgb)
                        
                        print('Success connection')
                        network_gui.send(net_image_bytes, source_path)
                    
                    else:
                        network_gui.send(None, source_path)

                    
                    # source_path = '/home/arinidhant/workspace/cv/tandt/truck'
                    # network_gui.send(net_image_bytes, source_path)
                    if do_training:
                        print("We are exiting the while loop)")

                        break


                except socket.timeout:
                    print("Socket timed out. No data received")
                    break

                except Exception as e:
                    print("handling exception")
                    print(e)
                    traceback.print_exc() 
                    network_gui.conn = None

            
            print("We exited while loop and training now)")
            

            # Compute loss (rendering loss + dssim)
            ssim_loss  = 1.0 - fused_ssim(img_out.unsqueeze(0), img1.unsqueeze(0))

            l1_loss = loss_fn(img_out, img1)

            # Improved regularization loss
            #reg_loss = scene.regularization_loss() * hparams['regularization_weight']
            total_loss = (1.0 - dssim_scale) * l1_loss + dssim_scale * ssim_loss # + reg_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            #Refinement Iteration
            # if epoch < hparams["densify_until_iteration"]:
            #     if (epoch + 1) % hparams["densification_interval"] == 0:
            #         scene.prune_and_densify(viewspace_points, visible_filter)


            # Logging
            if (i==0) and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"Epoch {epoch + 1}/{num_epochs}, L1 Loss: {l1_loss.item():.4f}, SSIM Loss: {ssim_loss.item():.4f} Total Loss: {total_loss.item():.4f}")

            # Optional: Save intermediate rendered images for debugging
            # if (epoch + 1) % 20 == 0:
            #     plt.imshow(img_out), f"output/render_epoch_{epoch + 1}.png")     
        

if __name__ == "__main__":
    network_gui.init("127.0.0.1", 6009)
    g_splat()
