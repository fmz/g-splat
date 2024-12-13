from argparse import ArgumentParser
import logging
import os






colmap_command = "colmap"
glomap_command = "glomap"

def extract_features(database_path):
    feat_extracton_cmd = colmap_command + " feature_extractor --database_path " + database_path +  " --image_path " + args.source_path + "/input --ImageReader.single_camera 1 \
        --ImageReader.camera_model SIMPLE_PINHOLE"  + " \
        --SiftExtraction.use_gpu 0" 
    print(feat_extracton_cmd)
    exit_code = os.system(feat_extracton_cmd)
    return
def match_features(database_path):
    feat_match_cmd  =  colmap_command + " sequential_matcher  --database_path " +database_path + " --SiftMatching.use_gpu 0"
    exit_code = os.system(feat_match_cmd)
    return
def map_features(database_path):
    feat_map_cmd = glomap_command + " mapper --database_path " +database_path +" --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse --TrackEstablishment.max_num_tracks 5000"
    exit_code = os.system(feat_map_cmd)
    return

def undistort(sparse_zero_folder):
    undist_cmd = colmap_command + " image_undistorter --image_path " + args.source_path + "/input \
    --input_path " +sparse_zero_folder+ " --output_path " + args.source_path + " --output_type COLMAP"
    exit_code = os.system(undist_cmd)
    return

def binary_to_text(output_folder_bin,output_folder_txt):
    b_to_t_cmd = colmap_command + "model_converter --input_path " +output_folder_bin + " -- output_path "+ output_folder_txt + " --output_type TXT "
    exit_code = os.system(b_to_t_cmd)
    return



def main():
    parent_dir = os.path.abspath(os.path.join(args.source_path, os.pardir))
    distorted_folder = os.path.join(parent_dir, 'distorted')
    database_path = os.path.join(distorted_folder, 'database.db')
    sparse_folder = os.path.join(parent_dir, 'sparse')
    sparse_zero_folder = os.path.join(sparse_folder, '0')
    sparse_one_folder = os.path.join(sparse_folder, '1')
      
    os.makedirs(distorted_folder, exist_ok=True)
    os.makedirs(sparse_folder, exist_ok=True)
    os.makedirs(sparse_zero_folder, exist_ok=True)
    os.makedirs(sparse_one_folder,exist_ok=True)



    print('Extracting')
    extract_features(database_path)
    print('Matching')
    match_features(database_path)
    print("Mapping")
    map_features(database_path)
    print("Distorting")
    undistort(sparse_zero_folder)
    print("Converting file types")
    binary_to_text(sparse_zero_folder,sparse_one_folder)


if __name__=="__main__":
    glomap_parser = ArgumentParser("Glomap Parser")
    glomap_parser.add_argument("--source_path", "-s", required=True, type=str)
    args =  glomap_parser.parse_args()
    main()


