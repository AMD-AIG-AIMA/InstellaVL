from datasets import load_dataset
from io import BytesIO
from base64 import b64decode
from PIL import Image
import os
import json
from tqdm import tqdm

split = 'validation'
ds_name = "visual-dialog"  # change the dataset name here
dataset = load_dataset("MMInstruction/M3IT", ds_name)
train_set = dataset[split]
dataroot = 'playground/data'
image_folder = 'Visual_Dialog/images/%s' % split
if not os.path.exists(os.path.join(dataroot, image_folder)):
    os.makedirs(os.path.join(dataroot, image_folder))

new_data = []
for id, train_instance in enumerate(tqdm(train_set)):
    item = {}
    image_base64_str_list = train_instance["image_base64_str"]  # str (base64)
    image_0 = Image.open(BytesIO(b64decode(image_base64_str_list[0]))).convert('RGB')
    image_0.save(os.path.join(dataroot, image_folder, '%06d.jpg'%id))
    item['id'] = os.path.join(image_folder, '%06d.jpg'%id)
    item['image'] = item['id'] 
    qa = train_instance['inputs'] + f'Answer: {train_instance["outputs"]}'
    qa = qa.replace('Image Descriptions:\n', '')
    qa_pairs = qa.split('Question: ')
    instruction = qa_pairs[0]
    conv = []
    for q_id, qa_pair in enumerate(qa_pairs[1:]):
        question, answer = qa_pair.split('Answer: ')
        if q_id == 0:
            question = '<image>\n' +  instruction + question
        # 
        # 
        conv.append({'from': 'human', 'value': question.strip() + '?'})
        conv.append({'from': 'gpt', 'value': answer.strip() + '.'})
    item['conversations'] = conv
    item['image_size'] = image_0.size
    new_data.append(item)

new_json_file = 'visual_dialog_%s_process.json' % split
if not os.path.exists('playground/data/Visual_Dialog'):
    os.makedirs('playground/data/Visual_Dialog')
dst_json_file = os.path.join('playground/data/Visual_Dialog', new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)
