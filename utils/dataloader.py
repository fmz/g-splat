from PIL import Image
import numpy as np

import os
import glob

class Dataset():
    # The dataset format is detailed in https://vision.middlebury.edu/mview/data/
    # The most important part is:

    # name_par.txt: camera parameters. There is one line for each image.
    # The format for each line is:
    # "imgname.png k11 k12 k13 k21 k22 k23 k31 k32 k33 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3".
    # The projection matrix for that image is K*[R t]. The image origin is top-left, with x increasing horizontally, y vertically.

    def __init__(self, data_dir):
        if os.path.isdir(data_dir):
            print(f'Loading data from {data_dir}...')
            self.data_dir = data_dir
        else:
            raise ValueError(f'The given directiory {data_dir} does not exist!')
        
        par_file = glob.glob(self.data_dir+"/*_par.txt")
        if len(par_file) != 1:
            raise ValueError(f'The given directiory {data_dir} does not have a par file!')

        self.par_file = par_file[0]

        # Read the images
        with open(self.par_file, encoding='utf8') as file:
            first_line = True
            i = 0
            for line in file:
                if (first_line):
                    n_img = int(line)
                    self.cam_K   = np.zeros((n_img, 3, 3))
                    self.cam_R   = np.zeros((n_img, 3, 3))
                    self.cam_t   = np.zeros((n_img, 3, 1))
                    self.cam_mat = np.zeros((n_img, 3, 4))
                    self.cam_pos = np.zeros((n_img, 3))

                    self.images = []
                    first_line = False
                    continue
                
                args = line.split()
                img_filename = args[0]

                self.images.append(Image.open(os.path.join(self.data_dir, img_filename)))
                self.images[-1] = np.array(self.images[-1], dtype=np.float32)
                self.images[-1] /= 255.

                if i == 1:
                    self.img_shape = self.images[-1].shape

                self.cam_K[i]   = np.array(args[1:10]).reshape(3,3)
                self.cam_R[i]   = np.array(args[10:19]).reshape(3,3)
                self.cam_t[i]   = np.array(args[19:]).reshape(3,1)
                self.cam_mat[i] = self.cam_K[i] @ (np.concatenate((self.cam_R[i], self.cam_t[i]), axis=1))

                R_inv = np.linalg.inv(self.cam_R[i])
                t_inv = -self.cam_t[i]
                self.cam_pos[i] = np.squeeze(R_inv @ t_inv)

                i += 1
        
        print("Loading complete")

