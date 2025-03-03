import json
from copy import deepcopy

src_json_file = 'playground/data/mathv360k/train_samples_all_tuning.json'

with open(src_json_file, 'r') as f:
    data = json.load(f)


dataset = 'PlotQA'

dst_json_file = f'playground/data/mathv360k/{dataset}.json'
count = 0
new_data = []
for item in data:
    if item['image'].startswith(dataset):
        new_data.append(item)
        count += 1
print(count)

with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)
