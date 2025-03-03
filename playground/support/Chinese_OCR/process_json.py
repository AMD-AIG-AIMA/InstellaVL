import os
from datasets import load_dataset
from tqdm import tqdm
import json
import numpy as np
import glob
#data['answer']['markData']['annotations'][0]['transferText']

prompts = [
        "Could you format the image details as a JSON structure?",
        "Can you provide a JSON representation of what's in this image?",
        "Transform this image's contents into a JSON data format.",
        "Generate a JSON breakdown of the elements in this image.",
        "Convert what you see in this image into a JSON object.",
        "Present the image analysis as structured JSON data.",
        "Create a JSON file describing all components of this image.",
        "Output a JSON-formatted description of the image contents.",
        "Give me a structured JSON summary of what this image shows.",
        "Parse this image's contents into a JSON data structure."
]


def get_iamge_files(path):
    # Find all .jpg and .JPG files
    json_files = glob.glob(os.path.join(path, '*.jpg'))
    return json_files

folders = ['screenshot', 'remake/book',  'remake/magazine',  'remake/newspaper',  'nature/ad',  'nature/banner', 'nature/bill',  'nature/billboard',
 'nature/board', 'nature/card', 'nature/map', 'nature/packaging', 'nature/sign', 'nature/ticket',  'nature/visiting_card' ]


converted_data = []

root_path = "playground/data/"
image_folder = "Chinese-OCR/data"
os.makedirs(os.path.join(root_path, image_folder), exist_ok=True)

for folder in folders:
    print(folder)
    images = get_iamge_files(os.path.join(root_path, image_folder, folder))
    for image in tqdm(images):
        json_data = {}
        json_data['id'] = image
        json_data['image'] = image
        conv = []
        json_file = json.load(open(image.replace('.jpg', '.json')))
        text = []
        anns = json_file['answer']['markData']['annotations']
        for ann in anns:
            text.append(ann['transferText'])
        text = ','.join(text)

        conv.append({'from': 'human', 'value': "<image>\n Please read out the text in the image. Seperate different phrases with comma" })
        conv.append({'from': 'gpt', 'value': text})
        json_data['conversations'] = conv
        converted_data.append(json_data)




new_json_file = 'Chinese-OCR_process.json'
dst_json_file = os.path.join('playground/data/Chinese-OCR', new_json_file)
print(len(converted_data))
with open(dst_json_file, 'w+') as f:
    json.dump(converted_data, f, indent=4)