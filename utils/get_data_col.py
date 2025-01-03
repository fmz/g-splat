import numpy as np
from argparse import ArgumentParser
import json
from plyfile import PlyData

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

    return camera_info

###Models adopted from https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py#L113

def get_colmap_images_info(file_path):
   
    images_info = {}
   
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
                images_info[image_id] = ({
                'image_id': image_id,
                'image_name': image_name,
                'quaternion': qvec,
                'translation_vector': tvec,
                'xys' : xys,
                'point3D_ids': point3D_ids
                })


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


def build_extrinsic_per_image(images_info,camera_info):
    translation_vectors = []
    rotation_matricies = []
    extrinsic_matricies = []
    images = []
    camera_json_info = []
    expose_info = {}

    for image_info in images_info:
        image_info = images_info[image_info]
        image_id = image_info['image_id']
        
        translation_vector = image_info['translation_vector']
        translation_vector = np.array([[translation_vector[0]],
                                       [translation_vector[1]],
                                       [translation_vector[2]]])
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
        extrinsic_matrix = np.hstack((rotation_matrix,translation_vector))
        focal = camera_info["params"][0]
        camera_json_info.append({
            "id:" : image_id,
            "image_name": image_info['image_name'],
            "width": camera_info["width"],
            "height": camera_info["height"],
            "position": np.array(rotation_matrix.T @ translation_vector).tolist(),
            "rotation": rotation_matrix.tolist(),
            "fy": focal,
            "fx": focal
        })

        expose_info[image_info['image_name']] = [[1.0,0.0,0.0], [0.0, 1.0 , 0.0], [0.0, 0.0, 1.0]]
        translation_vectors.append(translation_vector)
        rotation_matricies.append(rotation_matrix)
        extrinsic_matricies.append(extrinsic_matrix)
        images.append(image_info['image_name'])


        # c_filename = "utils/outputs/camera.json"
        # # Open the file in write mode ('w')
        # with open(c_filename, 'w') as f:
        #     json.dump(camera_json_info, f) 
        # e_filename = "utils/outputs/exposure.json"

        # with open(e_filename, 'w') as f:
        #     json.dump(expose_info,f)

    
    return extrinsic_matricies, rotation_matricies, translation_vectors, images


###Models adopted from https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py#L113

def read_points3D_text(path):
    points3D_output = {}
    points3d_coord = []
    points3d_rgb = []

    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                points3d_coord.append(xyz)
                rgb = np.array(tuple(map(int, elems[4:7])))/255
                points3d_rgb.append(rgb)
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D_output[point3D_id] = {
                        'id' : point3D_id,
                        'xyz' :xyz,
                        'rgb' :rgb,
                        'error' :error,
                        'image_ids' :image_ids,
                        'point2D_idxs' : point2D_idxs,
                }
    num_points = len(points3d_coord)
    
    return points3D_output, points3d_coord, points3d_rgb

def read_dense_file(dense_path):
    
    #PLY Reader from 
    #https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/scene/dataset_readers.py#L128
    
    plydata = PlyData.read(dense_path)
    vertices = plydata['vertex']
    points_3d = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

    return points_3d, colors
        

def main():
    print("Getting Camera Information")
    camera_info = get_colmap_camera_info(args.camera_path)
    print("Building intrinsic matrix")
    intrinsic = build_k_matrix(camera_info)
    print("Getting Image Information")
    images_info = get_colmap_images_info(args.image_path)
    print("Building Extrinsic Matricies")
    matricies_nd_vectors = build_extrinsic_per_image(images_info,camera_info)
    print("Gathering Point Clouds")
    points3d = read_points3D_text(args.points_path)
    if args.dense_path:
        read_dense_file(args.dense_path)
    

if __name__=="__main__":
    parser = ArgumentParser("Parser")
    parser.add_argument("--image_path", "-i", required=True, type=str)
    parser.add_argument("--camera_path", "-c", required=True, type=str)
    parser.add_argument("--points_path", "-p", required=True, type=str)
    parser.add_argument("--dense_path", "-d", required=False, type=str)
    args =  parser.parse_args()
    main()
