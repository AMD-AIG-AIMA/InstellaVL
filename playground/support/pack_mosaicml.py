import os
import json
import argparse
import numpy as np

from streaming import MDSWriter
from multiprocessing import Pool
from typing import Iterator, Tuple
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--image_folder", type=str, help=("image_folder"), required=True)
    parser.add_argument("--json_file", type=str, nargs='+', required=True)
    parser.add_argument("--root_path", type=str, default='playground/data')
    parser.add_argument("--folder_name", type=str, required=True)
    parser.add_argument('--processes', default=100, type=int)
    args = parser.parse_args()

    return args

args = parse_args()

# This could be a list of URLs needs to download
dataset = []
for file in args.json_file:
    with open(os.path.join(args.root_path, file), 'r') as f:
        tmp_dataset = json.load(f)
        dataset += tmp_dataset
columns = {'image': 'jpeg:95', 'text': 'json'}

def get_data_2(item: dict)->dict:
    r"""
    Processes an item to generate a sample containing an image and text.
    ---
    Args:
        - item (dict): A dictionary containing the data. It should have a key 'image' 
                     which holds the image filename.
    Returns:
        - dict: A dictionary with keys 'image' and 'text'. The 'image' key holds the 
              processed image (PIL Image object) and the 'text' key holds the original item.
              If an error occurs during processing, returns None.
    """
    try:
        if item.get('image', None):
            image_path = os.path.join(args.root_path, args.image_folder, item['image'])
            img = Image.open(image_path).convert('RGB')
            sample = {"image": img, "text": item}
        else:
            sample = {"image": Image.fromarray(np.random.randint(0, 256, (1, 1, 3), np.uint8)), "text": item}
    except:
        return None

    return sample

def init_worker():
    # Get the pid for the current worker process
    pid = os.getpid()
    print(f'\nInitialize Worker PID: {pid}', flush=True, end='')

def each_task(out_root: str, groups: int) -> Iterator[Tuple[str, int, int]]:
    """Get the sub-directory path and the sample range for each sub-directory.

    Args:
        out_root (str): base output mds directory
        groups (int): Number of sub-directories to create

    Yields:
        Iterator[Tuple[str, int, int]]: Each argument tuple
    """
    num_items = len(dataset)
    num_items_per_group = num_items // groups + 1
    for data_group in range(groups):
        sub_out_root = os.path.join(out_root, str(data_group))
        start_sample_idx = num_items_per_group * data_group
        end_sample_idx = min(num_items_per_group * (1+ data_group) - 1, num_items-1)
        yield sub_out_root, start_sample_idx, end_sample_idx

def convert_to_mds(args: Iterator[Tuple[str, int, int]]) -> None:
    """Convert raw dataset into MDS format

    Args:
        args (Iterator[Tuple[str, int, int]]): All arguments, packed into a tuple because
            process pools only pass one argument.

    Yields:
        Dict: A sample
    """
    sub_out_root, start_sample_idx, end_sample_idx = args

    def get_data(start: int, end: int):
        for i in range(start, end + 1):
            yield  get_data_2(dataset[i])


    columns = {'image': 'jpeg:95', 'text': 'json'}

    with MDSWriter(out=sub_out_root,
                   columns=columns, size_limit='64mb') as out:
        for sample in get_data(start_sample_idx, end_sample_idx):
            if sample is None:
                continue
            try:
                out.write(sample)
            except:
                continue

import shutil
shutil.rmtree(os.path.join(args.root_path, args.folder_name), ignore_errors=True)

arg_tuples = each_task(os.path.join(args.root_path, args.folder_name), groups=args.processes)

# Process group of data in parallel into directories of shards.
with Pool(initializer=init_worker, processes=args.processes) as pool:
    for count in pool.imap(convert_to_mds, arg_tuples):
        pass
print('Finished')

from streaming.base.util import merge_index
merge_index(os.path.join(args.root_path, args.folder_name), keep_local=True)
