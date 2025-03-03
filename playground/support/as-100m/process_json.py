import os
import json
from tqdm import tqdm
import glob
 
dataroot = 'playground/data'
image_folder = 'SA-1B/' 


json_file_list = glob.glob(dataroot + '/AS-100M/jsonl_format/*.jsonl')
for json_file in json_file_list:
    train_dataset = [json.loads(line.strip()) for line in open(json_file)]

    idx = 0
    item = {}
    new_data = []
    pbar = tqdm(total= len(train_dataset))
    flag = False
    while idx < len(train_dataset):
        while True:
            if idx >= len(train_dataset):
                flag = True
                break
                
            train_instance = train_dataset[idx]
            if not os.path.exists(os.path.join(dataroot, image_folder, train_instance['image'] )):
                print(os.path.join( image_folder, train_instance['image'] ))
                continue
            train_instance['image'] =  os.path.join( image_folder, train_instance['image'] )
            # 
            # 
            if train_instance['image'] != item.get('image', None) or len(item.get('conversations', [])) >= 10:
                new_data.append(item)
                item = {}
                break
            item['conversations'].append({'from': 'human', 'value':  train_instance['question']})
            item['conversations'].append({'from': 'gpt', 'value': train_instance['answer']})
            pbar.update(1)
            idx += 1
        if flag:
            break
        item['id'] = train_instance['image'] 
        item['image'] = train_instance['image'] 
        conv = []
        conv.append({'from': 'human', 'value': '<image>\n' +  train_instance['question']})
        conv.append({'from': 'gpt', 'value': train_instance['answer']})
        item['conversations'] = conv
        pbar.update(1)
        idx += 1


    pbar.close()

    new_json_file = 'AS-100M_%s_pretrain_process.json' % os.path.basename(json_file).replace('.jsonl', '')
    if not os.path.exists('playground/data/AS-100M/processed'):
        os.makedirs('playground/data/AS-100M/processed')
    dst_json_file = os.path.join('playground/data/AS-100M/processed', new_json_file)
    print(len(new_data))
    with open(dst_json_file, 'w+') as f:
        json.dump(new_data, f, indent=4)
