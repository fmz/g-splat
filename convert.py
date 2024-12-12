from argparse import ArgumentParser
import logging
import os




glomap_parser = ArgumentParser("Glomap Parser")
glomap_parser.add_argument("--source_path", "-s", required=True, type=str)
glomap_parser.add_argument("--database_path", "-d", required=True, type=str)

colmap_command = "colmap"
glomap_command = "glomap"

def extract_features():
    feat_extracton_cmd = colmap_command + " feature_extractor --database_path " + args.database_path + "--image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model SIMPLE_PINHOLE"  + " \
        --SiftExtraction.use_gpu 1" 
    exit_code = os.system(feat_extracton_cmd)
    return


def match_features():
    feat_match_cmd  =  colmap_command + " sequential_matcher  --database_path" + args.database_path
    exit_code = os.system(feat_extracton_cmd)
    return
def map_features():
    feat_map_cmd = glomap_command + " mapper --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001"


def main():
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)
    extract_features()
    match_features()
    map_features()


if __name__=="__main__":
    main()