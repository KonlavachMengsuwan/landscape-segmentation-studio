# Transformers SAM 2.1 image adapter

Version 0.2 extends this same image adapter to all eight SAM 2/SAM 2.1 Tiny, Small, Base-Plus and Large checkpoints. Each passed real MPS prompt and automatic tests; the complete manifest and measurements are in [MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md). The original Tiny/Small source notes below explain the common adapter. Device selection now supports MPS, CUDA and CPU; native Windows/Intel/NVIDIA execution remains unverified. Optional overlap suppression hides whole proposals reversibly after generation; it does not flatten or alter their pixel membership.

Source inspection: 7 September 2026, Transformers **5.16.1**. The active adapter lives in `backend/models.py` and passed real MPS image inference. Runtime results belong in `MODEL_COMPATIBILITY.md` and the measured compatibility report.

The adapter uses the official `facebook/sam2.1-hiera-tiny` and `facebook/sam2.1-hiera-small` image weights through `Sam2Model` and `Sam2Processor`. It requires the Transformers, PyTorch, torchvision, Pillow, and NumPy wheel environment. It does not need Meta's Python source package, Hydra/OmegaConf, a CUDA extension, or any video input package.

| Model | Revision | Safetensors bytes | SHA256 |
| --- | --- | ---: | --- |
| Tiny | `de431c4043854a71d8101e17995dfe596bf101a5` | 155,908,064 | `48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7` |
| Small | `ee5bba1d82bb8749febdf90f45e84b687142ba03` | 184,305,280 | `0a4067b11ce1e23d5229203f11c718a823060d15a4b23fa2372a7d4b77cbbc60` |

Public metadata: [Tiny](https://huggingface.co/api/models/facebook/sam2.1-hiera-tiny?blobs=true), [Small](https://huggingface.co/api/models/facebook/sam2.1-hiera-small?blobs=true). The local snapshot also contains `config.json`, `preprocessor_config.json`, and `processor_config.json`. A video preprocessor configuration is not needed for the selected image classes.

## Exact image and embedding API

The [documented image API](https://huggingface.co/docs/transformers/en/model_doc/sam2) accepts points grouped as `[image][object][point][xy]`, point labels as `[image][object][point]`, and boxes as `[image][object][xyxy]`. Canonical source coordinates go into the processor. Its coordinate transform scales x and y independently into the 1024-square model grid. Postprocessing resizes logits back to the canonical height and width.

The [5.16.1 model implementation](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/sam2/modeling_sam2.py) supports SDPA. `get_image_embeddings(pixel_values)` returns a **list** of feature tensors, including high-resolution features; retain the whole list. `model(image_embeddings=embeddings, input_points=..., input_labels=..., input_boxes=...)` reuses them without another image-encoder run. Supply either embeddings or pixel values, never both. The adapter retains one image's list and clears it when unloading or changing the source/model. No low-resolution-mask refinement is exposed before a separate test.

## Automatic generation and exact control mapping

The official [`mask-generation` pipeline](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/pipelines/mask_generation.py) uses these spellings:

| App concept | Official pipeline argument |
| --- | --- |
| Points per side | `points_per_crop` |
| Points per batch | `points_per_batch` |
| Crop layers | `crops_n_layers` |
| Quality threshold | `pred_iou_thresh` |
| Stability threshold | `stability_score_thresh` |
| Stability offset | `stability_score_offset` |
| Binary logit threshold | `mask_threshold` |
| Global NMS cutoff | `crops_nms_thresh` |

The pipeline returns `masks` and `scores`; it optionally returns `bounding_boxes` and `rle_mask`. It does not return per-mask stability scores. Its base forward stage moves output tensors to CPU before final NMS. This explains CPU NMS without assuming an unsupported Metal NMS kernel works.

The adapter implements the one-image, zero-crop-layer case directly using the public processor/model calls, so it can retain each mask's actual stability score. A uniform grid uses `(index + 0.5) / points_per_side` in each dimension. It processes a bounded number of point objects at a time, reusing the image embedding. CPU postprocessing interpolates logits onto the canonical grid, filters strict quality and stability cutoffs, discards empty masks, and runs torchvision NMS on mask bounding boxes. This is a documented application algorithm, not a claim that the Transformers pipeline itself returns stability values.

Stability is the number of pixels above logit `+1` divided by the number above `-1`, at a binary threshold of `0`. Zero-denominator masks receive stability zero and are discarded as empty. Separate overlapping instances remain separate unless their bounding boxes trigger the selected NMS threshold. Predicted quality and stability are model diagnostics, not accuracy. The formula and strict filtering follow the [5.16.1 image processor](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/sam2/image_processing_sam2.py).

The adapter keeps `crop_n_layers=0`. It uses the application's `box_nms_thresh` as its one active NMS threshold; `crop_nms_thresh` is rejected and is absent from presets. Disable both crop-layer and crop-NMS controls in this adapter. Do not claim crop support from a source signature alone: the current pipeline's `_forward` filters only the first image/crop, so complete multi-crop behavior needs its own validation.

Automatic generation accepts up to 8 megapixels. Batch size decreases as canonical pixel count grows to keep full-resolution logits bounded; each run records both the requested and effective batch size. This preserves the original grid and only changes how work is grouped. Candidates use lossless bit packing before NMS, capped at 128 MB of packed membership. Selected outputs are capped at 128 million Boolean pixels after NMS. Crossing the limit produces an actionable error instead of retaining an unbounded mask list. Cancellation is checked after embedding, between automatic batches, and before returning a result.

## Device and omitted processing

The model is explicitly placed on MPS with float32 and SDPA. Canonical mask interpolation, stability filtering, and NMS are explicitly on CPU. End-to-end timing includes those transfers and CPU work and synchronizes MPS. This is planned CPU postprocessing, not implicit `PYTORCH_ENABLE_MPS_FALLBACK`. Real performance and output still require measured smoke tests.

The Transformers processor's hole/sprinkle arguments currently have no implementation, despite appearing in its signature. Both remain zero, and the controls must stay disabled. Overlap-flattening postprocessing also remains off to preserve instance membership. These omissions are separate from Meta's documented CUDA-extension omission.
