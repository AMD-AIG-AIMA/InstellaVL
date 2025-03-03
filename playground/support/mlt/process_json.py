import json
import os
import numpy as np
import glob

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

dataroot = 'playground/data/mlt'
split = 'train'
image_folder = '%s' % split
annotaions = '%s_anns' % split

ann_folder = os.path.join(dataroot, annotaions)
files = glob.glob(ann_folder + '/*.txt')

new_data = []
# 
# 
for file in files:
    base_name = os.path.basename(file).split('.')[0][3:]
    new_example = {}
    image_id = '%s_%s' % (split, base_name)
    image_name_jpg = '%s/%s' % (image_folder, base_name)+'.jpg'
    image_name_png = '%s/%s' % (image_folder, base_name)+'.png'
    image_name_gif = '%s/%s' % (image_folder, base_name)+'.gif'
    if not os.path.exists(os.path.join(dataroot, image_name_jpg)) and not os.path.exists(os.path.join(dataroot, image_name_png)) and not os.path.exists(os.path.join(dataroot, image_name_gif)):
        print(base_name)
        
        
        continue
    if os.path.exists(os.path.join(dataroot, image_name_jpg)):
        image_name = image_name_jpg
    elif os.path.exists(os.path.join(dataroot, image_name_png)):
        image_name = image_name_png
    else:
        image_name = image_name_gif
    new_example['id'] = image_id
    new_example['image'] =  "mlt/" + image_name
    f_l = open(file, 'r')
    lines = f_l.readlines()
    value = []
    for line in lines:
        tokens = line.strip().split(',')
        points = []

        for i in range(4):
            points.append([int(tokens[2 * i]), int(tokens[2 * i + 1])])
        value.append({"transcription": tokens[-1], "points": points})    
    sentence = sort_annotations_by_position(value)

    conversations = [{'from': 'human', 'value': '<image>\nOCR this image section by section, from top to bottom, and left to right. Do not insert line breaks in the output text. Use a comma to split different parts of text'}, {'from': 'gpt', 'value':sentence}]
    new_example['conversations'] = conversations
    # 
    # 
    new_data.append(new_example)

new_json_file = '%s_labels_process.json' % split
dst_json_file = os.path.join(dataroot, new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)