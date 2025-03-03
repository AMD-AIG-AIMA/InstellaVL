import json
import os
from tqdm import tqdm

dataroot = 'playground/data/'
image_folder = 'UniChart/images'
file = open(os.path.join(dataroot, "UniChart", 'unichart.json'))
data = json.load(file)

new_data = []

for item in tqdm(data):
    image_name = item['imgname']
    image_path = os.path.join(dataroot, image_folder, image_name)
    if not os.path.exists(image_path):
        print(image_path)
        
        
    query = item['query']
    label = item['label']
    if '<chartqa>' in query:
        human_word = '<image>\n' + query.replace('<chartqa>', '').strip() 
    elif '<opencqa>' in query:
        human_word = '<image>\n' + query.replace('<opencqa>', '').strip()  
    elif '<summarize_chart>' in query:
        human_word = '<image>\n' + 'What insights can be drawn from this chart?/Explain the trends shown in this chart'
    elif '<extract_data_table>' in query:
        human_word = '<image>\n' + 'Please extract data from the table from left to right and up to down.\n '
    
    new_example = {'id': image_name, 'image': os.path.join(image_folder, image_name)}
    conversations = [{'from': 'human', 'value': human_word}, {'from': 'gpt', 'value': label}]
    new_example['conversations'] = conversations
    new_data.append(new_example)

new_json_file = 'unichart_labels_process.json'
dst_json_file = os.path.join('playground/data/UniChart', new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)

