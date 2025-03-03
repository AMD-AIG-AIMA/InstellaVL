import csv
import os
import numpy as np
import json

prompts = ["Describe the image concisely.",
           "Provide a brief description of the given image.",
           "Offer a succinct explanation of the picture presented.",
           "Summarize the visual content of the image.",
            "Give a short and clear explanation of the subsequent image.",
            "Share a concise interpretation of the image provided.",
            "Present a compact description of the photo’s key features.",
            "Relay a brief, clear account of the picture shown.",
            "Render a clear and concise summary of the photo.",
            "Write a terse but informative summary of the picture.",
            "Create a compact narrative representing the image presented."]


dataset = 'statista'
split = 'train'
csv_file = f'playground/data/Chart-to-text/{dataset}_dataset/dataset/dataset_split/{split}_index_mapping.csv'
dataroot = 'playground/data/'
image_dataroot = {'multi_col': f'chart2text/{dataset}_dataset/multiColumn', 'two_col': f'chart2text/{dataset}_dataset/imgs'}
annotation_dataroot = {'multi_col': f'Chart-to-text/{dataset}_dataset/dataset/multiColumn/captions', 'two_col': f'Chart-to-text/{dataset}_dataset/dataset/captions'}
new_data = []
with open(csv_file) as file:
    csv_reader = csv.reader(file)
    for line in csv_reader:
        line = line[0].strip()
        if not line.startswith('multi') and not line.startswith('two'):
            continue
        id = line.split('.')[0]
        image_split, image_id = id.split('-')
        image_path = os.path.join(image_dataroot[image_split], image_id+'.png')
        if not os.path.exists(os.path.join(dataroot, image_path)):
            print(os.path.join(dataroot, image_path))
            
            
        annotation_path = os.path.join(annotation_dataroot[image_split], image_id+'.txt')
        with open(os.path.join(dataroot, annotation_path)) as ann:
            lines = ann.readlines()
        caption = lines[0]
        new_example = {'id': id, 'image': image_path}
        question = '<image>\n' + prompts[np.random.randint(len(prompts))]
        conversations = [{'from': 'human', 'value': question}, {'from': 'gpt', 'value':caption}]
        new_example['conversations'] = conversations
        new_data.append(new_example)

new_json_file = '%s_%s_labels_process.json' % (dataset, split)
dst_json_file = os.path.join('playground/data/chart2text', new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)
