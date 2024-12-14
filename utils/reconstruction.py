import pycolmap
import numpy as np
from argparse import ArgumentParser

###Models adopted from https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py#L113


def get_colmap_camera_info(file_path):
    camera_info = {}
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
            camera_info['camera_id'] =  camera_id
            camera_info['models'] = model
            camera_info['width'] =  width
            camera_info['height'] = height
            camera_info['params'] = params
    print(f"{camera_info=}")

    return camera_info


def get_colmap_images_info(file_path):
   
    images_info = []
   
    with open(file_path, 'r') as file:
         while True:
            line = file.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
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
                images_info.append({
                'image_id': image_id,
                'quaternion': qvec,
                'translation_vector': tvec,
                'xys' : xys,
                'point3D_ids': point3D_ids
                })

    #print(f"{quaternions=}")

    return images_info

def build_k_matrix(camera_info):
    params = camera_info["params"]
    focal = params[0]
    principle_x = params[1]
    principle_y = params[2]
    k = np.array([[focal,0,principle_x],
                        [0,focal,principle_y],
                        [0,0,1]])
    return k


def build_extrinsic_per_image(images_info):
    translation_vectors = {}
    rotation_matricies = {}
    extrinsic_matricies = {}
    for image_info in images_info:
        image_id = image_info['image_id']
        
        translation_vector = image_info['translation_vector']
        qvec = image_info['quaternion']
        rotation_matrix = np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ])
        extrinsic_matrix = np.hstack(rotation_matrix,translation_vector)

        translation_vectors[image_id] = translation_vector 
        rotation_matricies[image_id] = rotation_matrix
        extrinsic_matricies[image_id] = extrinsic_matrix
        print(f"{translation_vector=}")
        print(f"{rotation_matrix=}")
        print(f"{translation_vector=}")

    
    return extrinsic_matricies, rotation_matricies, translation_vectors
        

def main():
    print("Getting Camera Information")
    camera_info = get_colmap_camera_info(args.camera_path)
    print("Building intrinsic matrix")
    intrinsic = build_k_matrix(camera_info)
    print("Getting Image Information")
    images_info = get_colmap_images_info(args.image_path)
    print("Building Extrinsic Matricies")
    build_extrinsic_per_image(images_info)

if __name__=="__main__":
    parser = ArgumentParser("Parser")
    parser.add_argument("--image_path", "-i", required=True, type=str)
    parser.add_argument("--camera_path", "-c", required=True, type=str)
    args =  parser.parse_args()
    main()
