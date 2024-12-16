from argparse import ArgumentParser
import logging
import os

from PIL import Image




colmap_command = "colmap"
glomap_command = "glomap"

# def resize():
#     print("Resizing")
#     images_path = args.source_path + "/input"
#     print(f"{images_path=}")
#     img_dir = os.listdir(images_path)
#     print(f"images = {len(img_dir)}")
#     i = 0
#     for img_name in img_dir: 
#         print(f"{img_name=}")
#         image = Image.open(images_path+"/"+img_name)
#         image = image.resize((1280,720))
#         image.save(images_path+"/"+img_name)
#         i +=1
#         print(image)
#     print(f"image count = {i}")
#     print("Done Resizing")
#     return

def extract_features(database_path):
    feat_extracton_cmd = colmap_command + " feature_extractor --database_path " + database_path +  " --image_path " + args.source_path + "/input --ImageReader.single_camera 1 \
        --ImageReader.camera_model SIMPLE_PINHOLE"  + " \
        --SiftExtraction.use_gpu 1" 
    print(feat_extracton_cmd)
    exit_code = os.system(feat_extracton_cmd)
    return
def match_features(database_path):
    feat_match_cmd  =  colmap_command + " sequential_matcher  --database_path " +database_path + " --SiftMatching.use_gpu 1"
    exit_code = os.system(feat_match_cmd)
    return
def map_features(database_path):
    feat_map_cmd = glomap_command + " mapper --database_path " +database_path +" --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/sparse --TrackEstablishment.max_num_tracks 5000"
    exit_code = os.system(feat_map_cmd)
    return

def undistort():
    undist_cmd = colmap_command + " image_undistorter --image_path " + args.source_path + "/input \
    --input_path " +args.source_path + "/sparse/0"+ " --output_path " + args.source_path + "/dense/ --output_type COLMAP"
    exit_code = os.system(undist_cmd)
    return

def dense_stero():
    dense_cmd = colmap_command +" patch_match_stereo \
    --workspace_path "+ args.source_path +"/dense/ \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency 1"
    exit_code = os.system(dense_cmd)
    return 

def dense_fusion():
    fusion_cmd = colmap_command +" stereo_fusion \
    --workspace_path " + args.source_path +"/dense/ --workspace_format COLMAP\
    --input_type geometric --output_path "+ args.source_path+"/dense/fused.ply"
    exit_code = os.system(fusion_cmd)
    return
def possion():
    possion_command = colmap_command + " poisson_mesher \
    --input_path " + args.source_path+"/dense/fused.ply \
    --output_path " +args.source_path+"/dense/meshed-poisson.ply"
    exit_code = os.system(possion_command)
    return
def meshed():
    meshed_cmd = colmap_command + " delaunay_mesher \
    --input_path " +args.source_path + "/dense --input_type dense --output_path " +args.source_path+"/dense/meshed-delaunay.ply"
    exit_code = os.system(meshed_cmd)

def main():
    #print("Resize")
    # resize()
    parent_dir = os.path.abspath(os.path.join(args.source_path, os.pardir))
    distorted_folder = os.path.join(parent_dir, 'distorted/distorted-classroom')
    database_path = os.path.join(distorted_folder, 'database.db')

    print('Extracting')
    extract_features(database_path)
    print('Matching')
    match_features(database_path)
    print("Mapping")
    map_features(database_path)
    print("Distorting")
    undistort()
    print("Dense Stereo")
    dense_stero()
    print("Dense Fusion")
    #dense_fusion()
    #print("Converting file types")
    #binary_to_text()
    print("Possion")
    possion()
    print("Mesh")
    meshed()


if __name__=="__main__":
    glomap_parser = ArgumentParser("Glomap Parser")
    glomap_parser.add_argument("--source_path", "-s", required=True, type=str)
    args =  glomap_parser.parse_args()
    main()


