import cv2
import numpy as np
import os



images = []
image_dirs = [img for img in os.listdir("test1")]
image_dirs.sort(key = lambda a: int(a[0:-4]))
print(image_dirs)

for dir in image_dirs:
    images.append(cv2.imread(f"test1/{dir}"))
#images.sort()  # Ensure images are in order

print(images[0].shape)
size = images[0].shape
 
 
out = cv2.VideoWriter('test1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1, size)
 
for i in range(len(images)):
    out.write(images[i])
out.release()
