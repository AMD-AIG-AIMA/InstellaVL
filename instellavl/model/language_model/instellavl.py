#    Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
import torch.nn as nn

from typing import List, Optional, Tuple, Union

from transformers import (AutoConfig, AutoModelForCausalLM,
                          OlmoConfig, OlmoModel, OlmoForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from instellavl.model.instellavl_arch import InstellaVLMetaModel, InstellaVLMetaForCausalLM


class InstellaVLConfig(OlmoConfig):
    """
    Configuration class for the InstellaVL model.
    Attributes:
        model_type (str): The type of the model, set to "instellavl".
    """

    model_type = "instellavl"


class InstellaVLModel(InstellaVLMetaModel, OlmoModel):
    config_class = InstellaVLConfig

    def __init__(self, config: OlmoConfig):
        super(InstellaVLModel, self).__init__(config)


class InstellaVLForCausalLM(OlmoForCausalLM, InstellaVLMetaForCausalLM):
    r"""
    InstellaVLForCausalLM is a class that extends OlmoForCausalLM and InstellaVLMetaForCausalLM to provide
    a language model with multimodal capabilities, specifically for handling images along with text.
    
    1. Attributes:
        - config_class (type): The configuration class to use for this model.
        - model (InstellaVLModel): The underlying model.
        - lm_head (nn.Linear): The linear layer for language modeling head.
    
    2. Methods:
        
        1. `__init__(config: InstellaVLConfig)`:
            Initializes the InstellaVLForCausalLM model with the given configuration.

        2. `get_model() -> InstellaVLModel`:
            Returns the underlying model.

        3. `forward() -> Union[Tuple, CausalLMOutputWithPast]`:
            Performs a forward pass through the model.
            
        4. `generate() -> Union[GenerateOutput, torch.LongTensor]`:
            Generates text based on the input.
            
        5. `prepare_inputs_for_generation(input_ids: torch.LongTensor,) -> dict`:
            Prepares inputs for text generation.
            
    """

    config_class = InstellaVLConfig

    def __init__(self, config: OlmoConfig):
        r"""
        Initializes the InstellaVLForCausalLM model.

        Args:
            - config (OlmoConfig): Configuration object for the model.

        Attributes:
            - model (InstellaVLModel): The main model instance.
            - lm_head (torch.nn.Linear): Linear layer that maps hidden states to vocabulary size.
        """
        super(OlmoForCausalLM, self).__init__(config)
        config.model_type = "instellavl"
        self.model = InstellaVLModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            - input_ids (torch.LongTensor, optional): Input token IDs.
            - attention_mask (torch.Tensor, optional): Attention mask.
            - position_ids (torch.LongTensor, optional): Position IDs.
            - past_key_values (List[torch.FloatTensor], optional): Past key values for caching.
            - inputs_embeds (torch.FloatTensor, optional): Input embeddings.
            - labels (torch.LongTensor, optional): Labels for language modeling.
            - use_cache (bool, optional): Whether to use cache.
            - output_attentions (bool, optional): Whether to output attentions.
            - output_hidden_states (bool, optional): Whether to output hidden states.
            - images (torch.FloatTensor, optional): Input images.
            - image_sizes (List[List[int]], optional): Sizes of input images.
            - return_dict (bool, optional): Whether to return a dictionary.
            - modalities (List[str], optional): List of modalities.
            - cache_position (optional): Cache position.
        
        Returns:
            Union[Tuple, CausalLMOutputWithPast]: The output of the forward pass.
        """
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                modalities,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""
        Args:
            - inputs (torch.Tensor, optional): Input tensor.
            - images (torch.Tensor, optional): Input images.
            - image_sizes (torch.Tensor, optional): Sizes of input images.
            - modalities (List[str], optional): List of modalities.
            - **kwargs: Additional arguments.
        
        Returns:
            Union[GenerateOutput, torch.LongTensor]: The generated text.
        """
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        r"""
        Args:
            - input_ids (torch.LongTensor): Input token IDs.
            - past_key_values (List[torch.FloatTensor], optional): Past key values for caching.
            - inputs_embeds (torch.FloatTensor, optional): Input embeddings.
            - **kwargs: Additional arguments.
        
        Returns:
            dict: Prepared inputs for generation.
        """
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("instellavl", InstellaVLConfig)
AutoModelForCausalLM.register(InstellaVLConfig, InstellaVLForCausalLM)
