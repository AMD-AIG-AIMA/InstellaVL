import json
import os
import numpy as np

def polygon_center(points):
    points = np.array(points)
    return np.mean(points, axis=0)

# Parse and sort annotations based on y-coordinates and then x-coordinates
def sort_annotations_by_position(annotations):
    # Calculate center points for all annotations
    new_annotations = []
    for annotation in annotations:
        if annotation['transcription'] != "###":
            new_annotations.append(annotation)
    annotations = new_annotations
    
    
    for annotation in annotations:
        annotation['center'] = polygon_center(annotation['points'])

    # Sort by y (top-to-bottom), then by x (left-to-right)
    sorted_annotations = sorted(annotations, key=lambda ann: (ann['center'][1], ann['center'][0]))

    # Combine sorted transcriptions
    sentence = ','.join([ann['transcription'] for ann in sorted_annotations])
    return sentence

dataroot = 'playground/data/ArT'
split = 'train_task2'
image_folder = '%s_images' % split
annotaions = '%s_labels.json' % split

file = open(os.path.join(dataroot, annotaions))
data = json.load(file)

new_data = []
for key, value in data.items():
    new_example = {}
    image_id = '%s_%s' % (split, key)
    image_name = '%s/%s' % (image_folder, key)+'.jpg'
    if not os.path.exists(os.path.join(dataroot, image_name)):
        print(image_name)
        continue
    new_example['id'] = image_id
    new_example['image'] = "ArT/"+ image_name
    sentence = sort_annotations_by_position(value)

    conversations = [{'from': 'human', 'value': '<image>\nOCR this image section by section, from top to bottom, and left to right. Do not insert line breaks in the output text. Use a comma to split different parts of text'}, {'from': 'gpt', 'value':sentence}]
    new_example['conversations'] = conversations
    new_data.append(new_example)

new_json_file = '%s_labels_process.json' % split
dst_json_file = os.path.join(dataroot, new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)