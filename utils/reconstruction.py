import pycolmap
import numpy as np
from argparse import ArgumentParser

def get_colmap_camera_info(file_path):
    camera_info = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        parts = line.split()
        if len(parts) >= 4:
            camera_id = parts[0]
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            camera_info.append({
                'camera_id': camera_id,
                'model': model,
                'width': width,
                'height': height,
                'params': params
            })s

    return camera_info


def get_colmap_images_info(file_path):
   
    quaternions = []
    translation vector = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue

        parts = line.split()
        if len(parts) == 8 and parts[7].lower().endswith(('.jpg', '.png', '.jpeg')):
            image_id, qw, qx, qy, qz, tx, ty, tz, image_name = (
                parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]),
                float(parts[5]), float(parts[6]), float(parts[7]), parts[8]
            )

            quaternions.append({
                'image_name': image_name,
                'quaternion': (qw, qx, qy, qz),
                'translation_vector': (tx, ty, tz)
            })

    return quaternions



def main():
    get_colmap_camera_info(args.camera_path)
    get_colmap_images_info(args.image_path)

if __name__=="__main__":
    parser = ArgumentParser("Parser")
    parser.add_argument("--image_path", "-i", required=True, type=str)
    parser.add_argument("--camera_path", "-c", required=True, type=str)
    args =  parser.parse_args()

