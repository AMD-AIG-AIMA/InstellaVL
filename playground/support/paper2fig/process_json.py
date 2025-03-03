import json
import os
import numpy as np
from tqdm import tqdm
dataroot = 'playground/data/'
image_dataroot = 'Paper2Fig100k/figures'
json_file = 'paper2fig_train.json'

file = open(os.path.join(dataroot, 'Paper2Fig100k', json_file))

data = json.load(file)

prompts = ["Generate a descriptive caption summarizing what the figure depicts, emphasizing any key observations or outcomes.",
           "Describe the main content of the figure, focusing on essential data, observations, or relationships shown.",
           "Write a caption that concisely captures the data or findings presented in the figure.",
           "Write a brief but informative caption explaining what the figure shows."]

new_data = []
for item in tqdm(data):
    image_path = os.path.join(image_dataroot, item['figure_id'] + '.png')
    if not os.path.exists(os.path.join(dataroot, image_path)):
        print(os.path.join(dataroot, image_path))
        
        
        continue
    new_example = {'id': item['figure_id']}
    new_example['image'] = image_path
    question = '<image>\n' + prompts[np.random.randint(len(prompts))]
    answer = item['captions_norm'][0].replace('figure number-tk', '').strip()
    conversations = [{'from': 'human', 'value': question}, {'from': 'gpt', 'value':answer}]
    new_example['conversations'] = conversations
    new_data.append(new_example)

new_json_file = 'paper2fig_labels_process.json'
dst_json_file = os.path.join('playground/data/Paper2Fig100k', new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)

    
    