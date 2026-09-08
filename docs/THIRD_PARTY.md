# Third-party conversion code

`scripts/vendor/sam_conversion.py` contains only the configuration and key-renaming functions needed from Hugging Face's `convert_sam_to_hf.py`, revision `db1e6f1e7fda38af4c4d20192bc35ac647a7a21f`. Copyright 2023 The HuggingFace Inc. team; Apache License 2.0.

`scripts/vendor/sam3_conversion.py` contains only the key mapping and QKV splitting functions from `convert_sam3_to_hf.py`, revision `22278df3198c4219f033f2a4b0931da3e5c21af4`. Copyright 2025 The Meta AI Authors and The HuggingFace Inc. team; Apache License 2.0. Network downloads, hub publishing, CLI code, sample fetching, and tokenizer downloads from the upstream scripts are excluded. Local conversion uses `torch.load(weights_only=True)` and strict model-state validation.

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt) applies to those retained functions. Copyright/license headers are preserved. Modifications: retain only pure local conversion functions and necessary imports; the SAM3.1 detector extraction and layout bridge are separate application code documented in SAM3_ACCESS.md. These files do not include model weights. Model licenses remain separate; Meta SAM3.1 gated weights are not redistributed.

Source links:
- [SAM key converter](https://github.com/huggingface/transformers/blob/db1e6f1e7fda38af4c4d20192bc35ac647a7a21f/src/transformers/models/sam/convert_sam_to_hf.py)
- [SAM3 key converter](https://github.com/huggingface/transformers/blob/22278df3198c4219f033f2a4b0931da3e5c21af4/src/transformers/models/sam3/convert_sam3_to_hf.py)
- [SAM 1 checkpoint architectures](https://github.com/facebookresearch/segment-anything)
- [SAM 2 checkpoints](https://github.com/facebookresearch/sam2)

Other dependencies retain their licenses in installed package metadata. The public sample's separate attribution is in `samples/ATTRIBUTION.md`.
