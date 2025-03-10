# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.

import re
import base64
import dataclasses

from PIL import Image
from io import BytesIO
from enum import auto, Enum
from typing import List, Any, Dict, Union, Tuple

from transformers import AutoTokenizer


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    MPT = auto()
    INSTELLA = auto()


@dataclasses.dataclass
class Conversation:
    r"""A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    tokenizer_id: str = ""
    tokenizer: Any = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    skip_next: bool = False

    def get_prompt(self):
        """
        Generates a formatted prompt string based on the messages and separator style.
        The function processes the messages stored in the instance, applies specific formatting rules 
        based on the separator style, and returns the resulting prompt string.
        
        Returns:
            `str`: The formatted prompt string.
        
        Raises:
            `ValueError`: If an invalid separator style is specified.
        """

        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0]
            if "mmtag" in self.version:
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            elif not init_msg.startswith("<image>"):
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, "<image>\n" + init_msg)
            else:
                messages[0] = (init_role, init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role

        elif self.sep_style == SeparatorStyle.INSTELLA:
            seps = [self.sep, self.sep2]
            ret = "|||IP_ADDRESS|||" 
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i % 2 == 1:
                        message = message.strip()
                    ret += role + message + seps[i % 2]
                else:
                    ret += role 
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(self, image: Union[str, Image.Image], image_process_mode: str, return_pil: bool=False, image_format: str="PNG")->Union[str, Image.Image]:
        r"""
        Processes an image according to the specified mode and returns either a PIL image or a base64 encoded string.
        
        Args:
            - image (Union[str, Image.Image]): The image to be processed. Can be a file path or a PIL Image object.
            - image_process_mode (str): The mode of image processing. Options are "Pad", "Default", "Crop", or "Resize".
            - return_pil (bool, optional): If True, returns a PIL Image object. If False, returns a base64 encoded string. Defaults to False.
            - image_format (str, optional): The format to save the image in if returning a base64 encoded string. Defaults to "PNG".
        
        Returns:
            Union[str, Image.Image]: The processed image, either as a PIL Image object or a base64 encoded string.
        
        Raises:
            ValueError: If an invalid image_process_mode is provided.
        """
        
        if image_process_mode == "Pad":

            def expand2square(pil_img, background_color=(122, 116, 104)):
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

            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")

        if type(image) is not Image.Image:
            image = Image.open(image).convert("RGB")

        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        max_len, min_len = 672, 448
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str

    def get_images(self, return_pil: bool=False, return_path: bool=False) -> List[Union[str, Image.Image]]:
        """
        Retrieve images from the conversation messages.

        Args:
            return_pil (bool): If True, return images as PIL objects. Defaults to False.
            return_path (bool): If True, return the image file paths instead of processing them. Defaults to False.

        Returns:
            list: A list of images or image paths depending on the arguments.
        """
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    if type(image) != list:
                        image = [image]
                    for img in image:
                        if not return_path and self.is_image_file(img):
                            img = self.process_image(img, image_process_mode, return_pil=return_pil)
                        else:
                            images.append(img)
        return images

    def is_image_file(self, filename: str)->bool:
        image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]
        return any(filename.lower().endswith(ext) for ext in image_extensions)

    def is_video_file(self, filename: str)->bool:
        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"]
        return any(filename.lower().endswith(ext) for ext in video_extensions)

    def to_gradio_chatbot(self)->list:
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    if type(image) != list:
                        image = [image]
                    if len(image) == 1:
                        msg = "<image>\n" + msg.replace("<image>", "").strip()
                    else:
                        msg = re.sub(r"(<image>)\n(?=<image>)", r"\1 ", msg)

                    img_str_list = []                         
                    for img in image:
                        if self.is_image_file(img):
                            img_b64_str = self.process_image(img, "Default", return_pil=False, image_format="JPEG")
                            img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" style="max-width: 256px; max-height: 256px; width: auto; height: auto; object-fit: contain;"/>'
                            img_str_list.append(img_str)
                        elif self.is_video_file(img):
                            ret.append(((img,), None))

                    msg = msg.strip()
                    img_place_holder = ""
                    for img_str in img_str_list:
                        img_place_holder += f"{img_str}\n\n"

                    if len(img_str_list) > 0:
                        msg = f"{img_place_holder}\n\n{msg}"

                    if len(msg) > 0:
                        ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self)->"Conversation":
        return Conversation(system=self.system, roles=self.roles, messages=[[x, y] for x, y in self.messages], offset=self.offset, sep_style=self.sep_style, sep=self.sep, sep2=self.sep2, version=self.version)

    def dict(self)->Dict[str, Any]:
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. " "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=[
        ["Human", "What are the key differences between renewable and non-renewable energy sources?"],
        [
            "Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n",
        ],
    ],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_instella = Conversation(
    system="",
    roles=("<|user|>\n", "<|assistant|>\n"),
    version="instella",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.INSTELLA,
    sep="\n",
    sep2='|||IP_ADDRESS|||\n'
)


default_conversation = conv_vicuna_v0
conv_templates = {
    "default": conv_vicuna_v0,
    "mpt": conv_mpt,
    "instella": conv_instella,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
