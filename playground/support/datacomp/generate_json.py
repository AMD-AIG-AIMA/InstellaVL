import argparse
from PIL import Image
import os
import glob
import json
import numpy as np
from tqdm import tqdm

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

    prompts = [
        "Describe the image concisely.",
        "Provide a brief description of the given image.",
        "Offer a succinct explanation of the picture presented.",
        "Summarize the visual content of the image.",
        "Give a short and clear explanation of the subsequent image.",
        "Share a concise interpretation of the image provided.",
        "Present a compact description of the photo's key features.",
        "Relay a brief, clear account of the picture shown.",
        "Render a clear and concise summary of the photo.",
        "Write a terse but informative summary of the picture.",
        "Create a compact narrative representing the image presented."
    ]

    new_data = []

    for folder in tqdm(dirs):
        image_folder = os.path.join(root_path, image_path, folder)
        json_files = get_json_files(image_folder)
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            caption = data['caption']
            item = {}
            item['id'] = data['key']
            item['image'] = os.path.join(image_path, folder, item['id'] + '.jpg')
            
            if not os.path.exists(os.path.join(root_path, item['image'])):
                print(os.path.join(root_path, item['image']))
                continue
            
            conv = []
            instruction = prompts[np.random.randint(len(prompts))]
            conv.append({'from': 'human', 'value': '<image>\n' + instruction})
            conv.append({'from': 'gpt', 'value': caption})
            item['conversations'] = conv
            new_data.append(item)

    # Create packed_jsons directory if it doesn't exist
    packed_jsons_path = os.path.join(root_path, image_path, 'packed_jsons')
    os.makedirs(packed_jsons_path, exist_ok=True)

    # Generate output JSON filename
    json_file = f'packed_jsons/{image_path}_{start_value:05d}_{end_value:05d}.json'
    dst_json_file = os.path.join(root_path, image_path, json_file)

    print(f"Total items processed: {len(new_data)}")
    
    # Write processed data to JSON file
    with open(dst_json_file, 'w+') as f:
        json.dump(new_data, f, indent=4)
    
    return new_data

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