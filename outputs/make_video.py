import cv2
import numpy as np
import os



images = []
image_dirs = [img for img in os.listdir("test1")]
image_dirs.sort(key = int)
print(image_dirs)

for dir in image_dirs:
    images.append(cv2.imread(f"test1/{dir}"))
#images.sort()  # Ensure images are in order
"""
if not images:
    print("No images found in the folder.")

frame = images[0]
height, width, _ = frame.shape
size = (height,width)
 
 
out = cv2.VideoWriter('test1.avi',-1, 5, size)
 
for i in range(len(images)):
    out.write(images[i])
out.release()
"""