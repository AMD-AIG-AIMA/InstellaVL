# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.

r"""This module provides various utility functions for processing images, including resizing, cropping, padding, 
and extracting patches. It also includes functions for processing images with different resolutions and 
tokenizing image prompts."""

import re
import ast
import math
import torch
import base64

from PIL import Image
from io import BytesIO
from typing import List, Tuple, Union, Any
from instellavl.constants import IMAGE_TOKEN_INDEX
from transformers import StoppingCriteria, PreTrainedTokenizer

def resize_and_center_crop(image: Image.Image, shortest_edge_length: int) -> Image.Image:
    r"""
    Resize the given image such that its shortest edge matches the specified length,
    and then center crop it to a square of the same size.
    
    Args:
        - image (`Image.Image`): The input image to be resized and cropped.
        - shortest_edge_length (`int`): The length of the shortest edge after resizing.
    
    Returns:
        `Image.Image`: The resized and center-cropped image.
    """
    
    # Calculate new dimensions and resize
    aspect_ratio = float(image.width) / float(image.height)
    if (aspect_ratio > 1):
        new_width = int(shortest_edge_length * aspect_ratio)
        new_height = shortest_edge_length
    else:
        new_width = shortest_edge_length
        new_height = int(shortest_edge_length / aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Calculate the position and perform the center crop
    left = (new_width - shortest_edge_length) / 2
    top = (new_height - shortest_edge_length) / 2
    right = (new_width + shortest_edge_length) / 2
    bottom = (new_height + shortest_edge_length) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))

    return cropped_image


def auto_pad_images(image: Image.Image, grid_params: list) -> Image.Image:
    r"""
    Automatically pads an input image to match the closest aspect ratio from a list of grid parameters.
    
    Args:
        - image (`Image.Image`): The input image to be padded. Must be a Pillow Image object.
        - grid_params (`list`): A list of integers representing the grid parameters to determine the target aspect ratio.
    
    Returns:
        `Image.Image`: The padded image with the closest aspect ratio from the grid parameters.
    
    Raises:
        `AssertionError`: If the input is not a Pillow Image object or if the grid parameters list is empty.
    """

    assert isinstance(image, Image.Image), "Input should be a Pillow Image"
    assert len(grid_params) > 0, "Grid parameters should not be empty"

    # Step 1: Calculate and find the closest aspect ratio
    input_width, input_height = image.size
    input_aspect_ratio = input_width / input_height
    candidate_resolutions = [(w / h, w, h) for w in grid_params for h in grid_params]
    closest_aspect_ratio = min(candidate_resolutions, key=lambda x: abs(input_aspect_ratio - x[0]))

    candidate_resolutions = [(x[1], x[2]) for x in candidate_resolutions if abs(x[0] - closest_aspect_ratio[0]) < 1e-3]

    target_resolution = min(candidate_resolutions, key=lambda res: abs(max(input_width, input_height) / max(res) - 1))

    resize_width, resize_height = target_resolution
    if input_width > input_height:
        resize_height = int(resize_width / input_aspect_ratio)
    else:
        resize_width = int(resize_height * input_aspect_ratio)
    resized_image = image.resize((resize_width, resize_height), Image.ANTIALIAS)

    # Step 5: Pad the resized image if necessary to match the target resolution
    pad_width = target_resolution[0] - resize_width
    pad_height = target_resolution[1] - resize_height
    padded_image = Image.new("RGB", target_resolution, color=(0, 0, 0))
    padded_image.paste(resized_image, (pad_width // 2, pad_height // 2))

    return padded_image


def extract_patches(image: Image.Image, patch_size: int, overlap_ratio: float) -> List[Image.Image]:
    r"""
    Extracts patches from a given image with specified patch size and overlap ratio.
    
    Args:
        - image (`Image.Image`): The input image from which patches are to be extracted. Must be a Pillow Image.
        - patch_size (`int`): The size of each patch (both width and height). Must be greater than 0.
        - overlap_ratio (`float`): The ratio of overlap between adjacent patches. Must be between 0 and 1 (exclusive).
    
    Returns:
        `List[Image.Image]`: A list of extracted patches as Pillow Images.
    
    Raises:
        `AssertionError`: If the input image is not a Pillow Image.
        `AssertionError`: If the patch size is not greater than 0.
        `AssertionError`: If the overlap ratio is not between 0 and 1.
    """

    assert isinstance(image, Image.Image), "Input should be a Pillow Image"
    assert patch_size > 0, "Patch size should be greater than 0"
    assert 0 <= overlap_ratio < 1, "Overlap ratio should be between 0 and 1"

    W, H = image.size
    patches = []

    stride = int(patch_size * (1 - overlap_ratio))

    num_patches_y = (H - patch_size) // stride + 1
    num_patches_x = (W - patch_size) // stride + 1

    y_start = (H - (num_patches_y - 1) * stride - patch_size) // 2
    x_start = (W - (num_patches_x - 1) * stride - patch_size) // 2

    for y in range(y_start, y_start + num_patches_y * stride, stride):
        for x in range(x_start, x_start + num_patches_x * stride, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)

    return patches


def process_highres_image_crop_split(image: Image.Image, data_args, processor=None) -> torch.Tensor:
    """
    Process a high-resolution image by cropping and splitting it into patches.

    Args:
        - image (`PIL.Image.Image`): The input image to be processed.
        - data_args: The data arguments containing crop and split resolutions.
        - processor: The image processor object. If None, it will be taken from data_args.

    Returns:
        `torch.Tensor`: A tensor containing the processed image patches.
    """
    crop_resolution = data_args.image_crop_resolution
    split_resolution = data_args.image_split_resolution
    if processor is None:
        processor = data_args.image_processor
    image_crop = resize_and_center_crop(image, crop_resolution)
    image_patches = extract_patches(image_crop, patch_size=split_resolution, overlap_ratio=0)
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def process_highres_image(image: Image.Image, processor, grid_pinpoints: str) -> torch.Tensor:
    r"""
    Processes a high-resolution image by resizing, padding, and extracting patches.
    
    Args:
        - image (`Image.Image`): The input image to be processed.
        - processor: An object that contains image processing parameters and methods.
        - grid_pinpoints (`str`): A comma-separated string of grid sizes to consider for resizing.
    
    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """

    grid_params = [int(x) for x in grid_pinpoints.split(",")]
    width_height = max(image.size)
    fit_grid_params = [x for x in grid_params if x >= width_height]
    if len(fit_grid_params) == 0:
        select_size = max(grid_params)
    else:
        select_size = min(fit_grid_params)
    # FIXME: always select the 448
    select_size = max(grid_params)
    image_padded = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))

    # FIXME: this seems to be a bug that it always resizes instead of padding
    image_original_resize = image.resize((processor.size["shortest_edge"], processor.size["shortest_edge"]))
    image_padded = image_padded.resize((select_size, select_size))
    image_patches = extract_patches(image_padded, patch_size=processor.size["shortest_edge"], overlap_ratio=0)
    image_patches = [image_original_resize] + image_patches
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def select_best_resolution(original_size: tuple, possible_resolutions: List[Tuple[int, int]]) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        - original_size (`tuple`): The original size of the image in the format (width, height).
        - possible_resolutions (`List[Tuple[int, int]]`): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        `tuple`: The best fit resolution in the format (width, height).
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


def resize_and_pad_image(image: Image.Image, target_resolution: tuple) -> Image.Image:
    r"""
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        - image (`Image.Image`): The input image.
        - target_resolution (`tuple`): The target resolution (width, height) of the image.

    Returns:
        `Image.Image`: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image: Image.Image, patch_size: int) -> list:
    """
    Divides an image into patches of a specified size.

    Args:
        - image (`Image.Image`): The input image.
        - patch_size (`int`): The size of each patch.

    Returns:
        `list`: A list of Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size: Tuple[int, int], grid_pinpoints: Union[str, list], patch_size: int) -> Tuple[int, int]:
    r"""
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        - image_size (`tuple`): The size of the input image in the format (width, height).
        - grid_pinpoints (`str` or `list`): A string representation of a list of possible resolutions.
        - patch_size (`int`): The size of each image patch.

    Returns:
        `tuple`: The shape of the image patch grid in the format (width, height).
    """
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image: Image.Image, processor: Any, grid_pinpoints: Union[str, List[Tuple[int, int]]]) -> torch.Tensor:
    r"""
    Process an image with variable resolutions.

    Args:
        - image (`Image.Image`): The input image to be processed.
        - processor: The image processor object.
        - grid_pinpoints (`str`): A string representation of a list of possible resolutions.

    Returns:
        `torch.Tensor`: A tensor containing the processed image patches.
    """
    # Convert grid_pinpoints from string to list
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        try:
            patch_size = processor.size[0]
        except Exception as e:
            patch_size = processor.size["shortest_edge"]
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size["height"])

    # FIXME: this seems to be a bug that it resizes instead of pad. # FIXME
    # but to keep it consistent with previous, i will keep it as it is
    # TODO: uncomment below to ablate with the padding
    if isinstance(processor.size, dict):
        shortest_edge = processor.size["shortest_edge"]
    else:
        shortest_edge = min(processor.size)
    image_original_resize = image.resize((shortest_edge, shortest_edge))
    # image_padded_square = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
    # image_original_resize = image_padded_square.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    image_patches = torch.stack(image_patches, dim=0)
    return image_patches


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img: Image.Image, background_color: tuple) -> Image.Image:
    r"""
    Expands a given PIL image to a square by adding a background color.

    Args:
        - pil_img (`Image.Image`): The input PIL image to be expanded.
        - background_color (`tuple`): The background color to use for expansion, specified as an RGB tuple.

    Returns:
        `Image.Image`: The expanded square PIL image.
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images: List[Image.Image], image_processor: Any, model_cfg: Any) -> Union[torch.Tensor, List[torch.Tensor]]:
    r"""
    Processes a list of images based on the specified model configuration.

    Args:
        - images (`list`): A list of images to be processed.
        - image_processor (`ImageProcessor`): An instance of the image processor to be used.
        - model_cfg (`ModelConfig`): Configuration object containing model settings.

    Returns:
        `torch.Tensor` or list: Processed images as a tensor if all images have the same shape, 
                              otherwise a list of processed images.
    """
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "highres":
        for image in images:
            image = process_highres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    elif image_aspect_ratio == "crop_split":
        for image in images:
            image = process_highres_image_crop_split(image, model_cfg, image_processor)
            new_images.append(image)
    elif image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            new_images.append(image)
    else:
        return image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt: str, tokenizer: PreTrainedTokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None)->Union[torch.Tensor, List[torch.Tensor]]:
    r"""
    Tokenizes a prompt containing image tokens and inserts the specified image token index at the appropriate positions.

    Args:
        - prompt (str): The input prompt string containing text and "<image>" placeholders.
        - tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the text chunks.
        - image_token_index (int): The token index to use for the image placeholders. Default is IMAGE_TOKEN_INDEX.
        - return_tensors (str, optional): The type of tensor to return. If "pt", returns a PyTorch tensor. Default is None.

    Returns:
        list or torch.Tensor: The tokenized input IDs as a list or a PyTorch tensor if return_tensors is specified.
    """
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]
    # FIXME: prompt_chunks = [tokenizer(chunk, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def get_model_name_from_path(model_path: str)->str:
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0] :] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
