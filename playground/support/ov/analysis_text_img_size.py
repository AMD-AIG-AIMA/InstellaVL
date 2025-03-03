from datasets import load_dataset
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

dataset_name = "echo840/OCRBench"
subset = None
split = "test"
dataset = load_dataset(dataset_name, subset, split=split)

heights = []
widths = []
base_size = 336
piles = []
for item in tqdm(dataset):
    if isinstance(item['image'], list):
        image_size = item['image'][0].size
    else:
        image_size = item['image'].size
    heights.append(min(image_size[0], 4000))
    widths.append(min(image_size[1], 4000))
    piles.append(min(heights[-1] * widths[-1] / (base_size ** 2), 40))

if subset:
    output_folder = "image_size_plot/%s_%s" % (dataset_name.replace('/', '--'), subset)
else:
    output_folder = "image_size_plot/%s" % (dataset_name.replace('/', '--'))
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

fig, axs = plt.subplots(3, 1, sharey=True, tight_layout=True)
# 
# 
axs[0].hist(heights, bins=100)
axs[1].hist(widths, bins=100)
axs[2].hist(piles, bins=int(max(piles))+1)

# 
# 
plt.savefig(os.path.join(output_folder, f'{split}.png'), bbox_inches='tight')