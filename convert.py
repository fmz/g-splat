from argparse import ArgumentParser
import logging
import shuthil 


'Todo : add arguemnts such as file path 
'



glomap_parser = ArgumentParser("Glomap Parser");

def extract_features():
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)

def map_features():
    return

def resize():
    return




def main():
    #Extract features, 
    #Match images 


if __name__=="__main__":
    main()