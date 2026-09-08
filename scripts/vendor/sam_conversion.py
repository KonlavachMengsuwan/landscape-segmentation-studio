# Modified for Landscape Segmentation Studio: retain only pure local conversion
# functions and required imports; omit upstream download, publish and CLI flows.
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Only pure local key/config conversion functions retained; see THIRD_PARTY.md."""
import re
import torch
from transformers import SamConfig, SamVisionConfig


def get_config(model_name):
    if "slimsam-50" in model_name:
        vision_config = SamVisionConfig(
            hidden_size=384,
            mlp_dim=1536,
            num_hidden_layers=12,
            num_attention_heads=12,
            global_attn_indexes=[2, 5, 8, 11],
        )
    elif "slimsam-77" in model_name:
        vision_config = SamVisionConfig(
            hidden_size=168,
            mlp_dim=696,
            num_hidden_layers=12,
            num_attention_heads=12,
            global_attn_indexes=[2, 5, 8, 11],
        )
    elif "sam_vit_b" in model_name:
        vision_config = SamVisionConfig()
    elif "sam_vit_l" in model_name:
        vision_config = SamVisionConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            global_attn_indexes=[5, 11, 17, 23],
        )
    elif "sam_vit_h" in model_name:
        vision_config = SamVisionConfig(
            hidden_size=1280,
            num_hidden_layers=32,
            num_attention_heads=16,
            global_attn_indexes=[7, 15, 23, 31],
        )

    config = SamConfig(
        vision_config=vision_config,
    )

    return config

KEYS_TO_MODIFY_MAPPING = {
    "iou_prediction_head.layers.0": "iou_prediction_head.proj_in",
    "iou_prediction_head.layers.1": "iou_prediction_head.layers.0",
    "iou_prediction_head.layers.2": "iou_prediction_head.proj_out",
    "mask_decoder.output_upscaling.0": "mask_decoder.upscale_conv1",
    "mask_decoder.output_upscaling.1": "mask_decoder.upscale_layer_norm",
    "mask_decoder.output_upscaling.3": "mask_decoder.upscale_conv2",
    "mask_downscaling.0": "mask_embed.conv1",
    "mask_downscaling.1": "mask_embed.layer_norm1",
    "mask_downscaling.3": "mask_embed.conv2",
    "mask_downscaling.4": "mask_embed.layer_norm2",
    "mask_downscaling.6": "mask_embed.conv3",
    "point_embeddings": "point_embed",
    "pe_layer.positional_encoding_gaussian_matrix": "shared_embedding.positional_embedding",
    "image_encoder": "vision_encoder",
    "neck.0": "neck.conv1",
    "neck.1": "neck.layer_norm1",
    "neck.2": "neck.conv2",
    "neck.3": "neck.layer_norm2",
    "patch_embed.proj": "patch_embed.projection",
    ".norm": ".layer_norm",
    "blocks": "layers",
}

def replace_keys(state_dict):
    model_state_dict = {}
    state_dict.pop("pixel_mean", None)
    state_dict.pop("pixel_std", None)

    output_hypernetworks_mlps_pattern = r".*.output_hypernetworks_mlps.(\d+).layers.(\d+).*"

    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        if re.match(output_hypernetworks_mlps_pattern, key):
            layer_nb = int(re.match(output_hypernetworks_mlps_pattern, key).group(2))
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_nb == 2:
                key = key.replace("layers.2", "proj_out")

        model_state_dict[key] = value

    model_state_dict["shared_image_embedding.positional_embedding"] = model_state_dict[
        "prompt_encoder.shared_embedding.positional_embedding"
    ]

    return model_state_dict
