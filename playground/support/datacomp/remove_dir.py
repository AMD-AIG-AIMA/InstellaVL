import argparse
from PIL import Image
import os
import glob
import json
import numpy as np
from tqdm import tqdm
import shutil

def process_image_data(root_path, image_path, start_value=0, end_value=500):
    def list_directories(path):
        # List all directories in the specified path
        directories = [entry.name for entry in os.scandir(path) if entry.is_dir()]
        return directories

    def get_json_files(path):
        # Find all .json files
        json_files = glob.glob(os.path.join(path, '*.json'))
        return json_files

    # Prepare directory list with zero-padded numbers
    dirs = ['%05d' % i for i in range(start_value, end_value)]

    new_data = []

    for folder in tqdm(dirs):
        image_folder = os.path.join(root_path, image_path, folder)
        if os.path.isdir(image_folder):
            print(image_folder)
            shutil.rmtree(image_folder)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process image data from a specified range of directories.')
    
    # Add arguments with default values
    parser.add_argument('--root_path', 
                        default='playground/data/', 
                        help='Root path to the image directory')
    
    parser.add_argument('--image_path', 
                        default='Recap-DataComp-1B_webdataset', 
                        help='Relative path to the image directory')
    
    parser.add_argument('--start_value', 
                        type=int, 
                        default=0, 
                        help='Starting directory number (default: 0)')
    
    parser.add_argument('--end_value', 
                        type=int, 
                        default=500, 
                        help='Ending directory number (default: 500)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the processing function with parsed arguments
    process_image_data(
        root_path=args.root_path, 
        image_path=args.image_path, 
        start_value=args.start_value, 
        end_value=args.end_value
    )

if __name__ == "__main__":
    main()