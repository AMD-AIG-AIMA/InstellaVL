from datasets import load_dataset
import json
import os

dataroot = 'playground/data/mmc_instruction/'
dataset = load_dataset("json", data_files=os.path.join(dataroot, "mmc_instruction_non-arxiv_text.jsonl"))

data = []

for example in dataset['train']:
    new_example = {}
    image_path = 'mmc_instruction_non-arxiv_images/%s' % example['image_id']
    if not os.path.exists(os.path.join(dataroot, image_path)):
        
        
    else:
        new_example['id'] = image_path
        new_example['image'] = image_path
        # s
        # 
        question = example['question']
        # question = question.replace('<image>\n', '\n')
        if '<image>' not in question:
            question = "<image>\n" + question
        conversations = [{'from': 'human', 'value': question}, {'from': 'gpt', 'value': example['answer']}]
        new_example['conversations'] = conversations
        data.append(new_example)

dst_json_file = os.path.join(dataroot, 'mmc_instruction_non-arxiv_text.json')
print(len(data))
with open(dst_json_file, 'w+') as f:
    json.dump(data, f, indent=4)

