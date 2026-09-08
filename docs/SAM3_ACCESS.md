# SAM 3.1 access and local image inference

SAM3.1 is an optional advanced model. Each new recipient must review Meta's [SAM License](LICENSING.md) and obtain access through the [official SAM3.1 checkpoint page](https://huggingface.co/facebook/sam3.1). Sign in and download files through the publisher's own interface; do not paste access tokens into Studio, chat, or GitHub. No token or model weights are included in the application package.

For the original development installation, the owner obtained approval and supplied the checkpoint. No further approval is needed for that already authorized local file. The earlier HTTP 401 probe is historical, not a statement that recipients inherit the owner's access.

**SAM 3.1 text inference now works locally on this M4 through an experimental image-detector adapter.** It is not an official Transformers SAM 3.1 integration. The publisher README explicitly says that SAM 3.1 has no Transformers integration. The application does not load tracking state, video predictors, video decoders, or training components.

The supplied checkpoint SHA256 is `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`. Conversion follows the official Meta image loader's `detector.` extraction and upstream Hugging Face key mapping. The multiplex detector has three FPN scales `[4, 2, 1]`. All **1,464 image-detector tensors** load with `strict=True`; no missing parameter is randomly initialized. Unused tracking/interactive branches and precomputed rotary buffers are explicitly excluded and recorded in the local manifest.

Transformers 5.16.1 normally computes four pyramid levels and discards the last. `backend/sam31.py` appends an alias of the final genuine feature to the three-level output, which that existing slice immediately discards. This preserves all three detector features without inventing weights or changing their values. This narrow bridge is specific to the pinned version and is why the selector says **experimental**. Numerical parity against the upstream CUDA implementation has not been established; successful execution is not an accuracy claim.

Real offline MPS float32 / SDPA test: public sample 768×512, text `mountain`, one nonempty mask, model load 1.61 seconds, forward 7.29 seconds, total inference/postprocessing 7.32 seconds. The application also processed the canonical 3072×2048 image: one 443,050-pixel mask, 2.97-second load and 5.35-second measured inference including explicit CPU postprocessing. These are individual observations, not speed promises. No remote inference or token lookup was used.

Text controls: detection confidence and mask probability thresholds, each 0–1; optional overlap filtering, 0–100%. Text runs retain the phrase and thresholds, and previous phrases appear in prompt history. Point/box prompting is not exposed for this experimental detector. Empty text results are valid; large proposal sets are bounded by the shared result pixel limit.

## Install from your authorized files

Complete baseline Studio setup first. In the dedicated workspace, create `SAM3.1/` **beside `app/`**. Download the following files from the official model repository into it:

```text
LandscapeSegmentationStudio/
  app/
  SAM3.1/
    sam3.1_multiplex.pt
    config.json
    processor_config.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    vocab.json
    merges.txt
    LICENSE
```

These are the files supplied for the tested conversion. The checkpoint alone is insufficient: the processor needs its local tokenizer/configuration. Keep all files from the same publisher release. This adapter expects the checkpoint checksum documented above; if it differs, stop and inspect the upstream change instead of renaming another model or disabling the check. Keep the downloaded originals, including their license. Allow space for a separate converted copy and temporary memory during conversion.

From `app/`, using the project Python (`../runtime/venv/bin/python` on Mac, `..\runtime\venv\Scripts\python.exe` on Windows):

Windows Command Prompt:

```bat
..\runtime\venv\Scripts\python.exe -B scripts\convert_local_checkpoints.py --model sam31
..\runtime\venv\Scripts\python.exe -B scripts\verify_models.py --model sam31
```

Mac Terminal:

```sh
../runtime/venv/bin/python -B scripts/convert_local_checkpoints.py --model sam31
../runtime/venv/bin/python -B scripts/verify_models.py --model sam31
```

Conversion is offline, preserves the original, and refuses to overwrite an existing converted directory. New conversions include a copy of the SAM License. A fresh device must pass its own local test before becoming selectable; restart Studio afterward. NVIDIA/Windows execution remains unverified here. All original and converted files stay local and are excluded from GitHub and release packages. Read the [license review](LICENSING.md) for the experimental conversion's legal interpretation limits.

Sources: [official SAM 3.1 release](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/RELEASE_SAM3p1.md), [Meta builder](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model_builder.py), [upstream HF key converter](https://github.com/huggingface/transformers/blob/22278df3198c4219f033f2a4b0931da3e5c21af4/src/transformers/models/sam3/convert_sam3_to_hf.py).
