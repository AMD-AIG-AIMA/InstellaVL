import json
import os
import numpy as np
import glob

def get_points_from_seg(segs):
    segs = segs[0]
    assert len(segs) % 2 == 0
    points = []
    for i in range(len(segs) // 2):
        points.append((segs[2*i], segs[2*i+1]))
    return points

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

split = 'validation'
json_file = 'playground/data/openvino/text_spotting_openimages_v5_%s.json' % split
dataroot = 'playground/data/'
with open(json_file, 'r') as f:
    data = json.load(f)

images = data['images']
num_ann = len(data['annotations'])

ind = 0
image_id = 0
current_anns = []
new_data = []
print(num_ann)
while ind < num_ann:
    if image_id == data['annotations'][ind]['image_id']:
        
        if data['annotations'][ind]['attributes']['legible']:
            ann = {'points': get_points_from_seg(data['annotations'][ind]['segmentation']), 
                   'transcription': data['annotations'][ind]['attributes']['transcription']}
            current_anns.append(ann)
        ind += 1
    else:
        image_info = images[image_id]
        file_name = 'openvino/' + image_info['file_name']
        image_size = (image_info['height'], image_info['width'])
        question = "<image>\n" + 'OCR this image section by section, from top to bottom, and left to right. Do not insert line breaks in the output text. Use a comma to split different parts of text'
        sentence = sort_annotations_by_position(current_anns)
        conversations = [{'from': 'human', 'value': question}, {'from': 'gpt', 'value': sentence}]
        new_example = {}
        new_example['id'] = file_name
        new_example['image'] = file_name
        if not os.path.exists(os.path.join(dataroot, file_name)):
            print(file_name)
            
            
        new_example['image_size'] = image_size
        new_example['conversations'] = conversations
        if sentence.strip():
            new_data.append(new_example)
        current_anns= []
        image_id = data['annotations'][ind]['image_id']
    
new_json_file = 'openvino/%s_labels_process.json' % split
dst_json_file = os.path.join(dataroot, new_json_file)
print(len(new_data))
with open(dst_json_file, 'w+') as f:
    json.dump(new_data, f, indent=4)


        
