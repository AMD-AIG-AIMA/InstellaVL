import os
from PIL import Image
import yaml
import json
import tqdm

image_file = '/home/ximensun/code/LLaVA-NeXT/scripts/train/LLaVA-MidStage_3.yaml'
image_dataroot = '/home/ximensun/code/LLaVA-NeXT/playground/data/LLaVA-Stage1.5'

num_piles = 3

def generate_pin_points(num_piles, base_size):
    pin_points = [(x * base_size, y * base_size) for x in range(1, num_piles+1) for y in range(1, num_piles+1) if x * y <=num_piles]
    return pin_points

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


# grid_pinpoints = generate_pin_points(num_piles, 336)
grid_pinpoints = [(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]


# out_boundary = 0
compute_dict  = {}
import matplotlib.pyplot as plt
base_pixel = 336
output_folder = 'image_size_plot'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if image_file.endswith('yaml'):
    with open(image_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    datasets = yaml_data.get("datasets")
    
    json_files = []
    for dataset in datasets:
        json_path = dataset.get("json_path")
        json_files.append(json_path)
else:
    json_files = [image_file]



# 
# 
heights = []
widths = []
piles = []
for json_path in json_files:
    count = 0
    with open(json_path, "r") as json_file:
        cur_data_dict = json.load(json_file)
        print(json_path)
        flag = False
        doc_heights = []
        doc_widths = []
        doc_piles = []
        for item in tqdm.tqdm(cur_data_dict):
            if not item.get('image', None):
                continue
            if item.get('image_size', None):
                image_size = item['image_size']
            else:
                flag = True
                image_path = os.path.join(image_dataroot, item['image'])
                try:
                    image = Image.open(image_path).convert("RGB")
                except:
                    continue
                image_size = image.size
            # 
            # 
            # 
            # 
            best_fit = select_best_resolution(image_size, grid_pinpoints)
            # if num < image_size[0] or best_fit[1] < image_size[1]:
            #     out_boundary += 1
            key = "%d_%d" % (best_fit[0], best_fit[1])
            compute_dict[key] = compute_dict.get(key, 0) + 1
            item['image_size'] = image_size
            if flag and (count + 1) % 100000 == 0:
                with open(json_path, "w") as file:
                    json.dump(cur_data_dict, file, indent=4)
                flag = False
            count += 1
            heights.append(min(image_size[0], 4000))
            widths.append(min(image_size[1], 4000))
            piles.append(min(image_size[0] * image_size[1] / (base_pixel ** 2), 40 ))
            doc_heights.append(min(image_size[0], 4000))
            doc_widths.append(min(image_size[1], 4000))
            doc_piles.append(min(image_size[0] * image_size[1] / (base_pixel ** 2), 40 ))
    with open(json_path, "w") as file:
        json.dump(cur_data_dict, file, indent=4)
    fig, axs = plt.subplots(3, 1, sharey=True, tight_layout=True)
    # 
    # 
    axs[0].hist(doc_heights, bins=100)
    axs[1].hist(doc_widths, bins=100)
    axs[2].hist(doc_piles, bins=int(max(piles))+1)

    # 
    # 
    split = os.path.basename(json_path)[:-5]
    plt.savefig(os.path.join(output_folder, f'{split}.png'), bbox_inches='tight')
    
fig, axs = plt.subplots(3, 1, sharey=True, tight_layout=True)
# 
# 
axs[0].hist(heights, bins=100)
axs[1].hist(widths, bins=100)
axs[2].hist(piles, bins=int(max(piles))+1)

# 
# 
split = 'overall_pretrain'
plt.savefig(os.path.join(output_folder, f'{split}.png'), bbox_inches='tight')




