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
            })
    print(f"{camera_info=}")

    return camera_info


def get_colmap_images_info(file_path):
   
    quaternions = []
    translation_vector = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            elems = line.split()
            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            camera_id = int(elems[8])
            image_name = elems[9]
            elems = file.readline().split()
            xys = np.column_stack(
                [
                    tuple(map(float, elems[0::3])),
                    tuple(map(float, elems[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, elems[2::3])))
                    
            quaternions.append({
                'image_id': image_id,
                'quaternion': qvec,
                'translation_vector': tvec,
                'xys' : xys,
                'point3D_ids': point3D_ids
                })

    print(f"{quaternions=}")

    return quaternions




def main():
    print("here")
    get_colmap_camera_info(args.camera_path)
    get_colmap_images_info(args.image_path)

if __name__=="__main__":
    parser = ArgumentParser("Parser")
    parser.add_argument("--image_path", "-i", required=True, type=str)
    parser.add_argument("--camera_path", "-c", required=True, type=str)
    args =  parser.parse_args()
    main()
