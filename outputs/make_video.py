import cv2
import numpy as np
import os



images = []
image_dirs = [img for img in os.listdir("monkey_test")]
image_dirs.sort(key = lambda a: int(a[0:-4]))
print(image_dirs)

for dir in image_dirs:
    images.append(cv2.imread(f"monkey_test/{dir}"))
#images.sort()  # Ensure images are in order

height, width, layers = images[0].shape
size = width, height

 
 
video = cv2.VideoWriter("monkey_test.avi", cv2.VideoWriter_fourcc(*'DIVX'), 40, (width, height))

 
for i in range(len(images)):
    video.write(images[i])
video.release()
cv2.destroyAllWindows()