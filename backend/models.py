"""Image-only Transformers SAM 2.1 adapter, verified with real local MPS inference.

Source API checked against Transformers v5.16.1. Model forward runs on MPS;
mask interpolation, stability filtering and torchvision NMS run explicitly on CPU.
No CUDA extension, implicit MPS fallback, crop pyramid, or hole cleanup is used.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

IMPLEMENTATION = "Hugging Face Transformers"
MODEL_ENTRIES = json.loads((Path(__file__).parents[1] / "config/model-manifest.json").read_text(encoding="utf-8"))
LOCAL_MANIFEST = Path(__file__).parents[2] / "setup-notes/local-model-manifest.json"
if LOCAL_MANIFEST.is_file():
    MODEL_ENTRIES.extend(json.loads(LOCAL_MANIFEST.read_text(encoding="utf-8")))
MODEL_CATALOG = {entry["id"]: entry for entry in MODEL_ENTRIES}
REVISIONS = {key: entry["revision"] for key, entry in MODEL_CATALOG.items()}
from .overlap import DEFAULTS as OVERLAP_DEFAULTS, validate_overlap
PRESETS = {
    "Fast preview": dict(points_per_side=8, points_per_batch=4, pred_iou_thresh=.8, stability_score_thresh=.90, crop_n_layers=0, box_nms_thresh=.7),
    "Balanced": dict(points_per_side=16, points_per_batch=4, pred_iou_thresh=.88, stability_score_thresh=.95, crop_n_layers=0, box_nms_thresh=.7),
    "Detailed": dict(points_per_side=24, points_per_batch=4, pred_iou_thresh=.88, stability_score_thresh=.95, crop_n_layers=0, box_nms_thresh=.7),
}
for preset in PRESETS.values():
    preset.update(OVERLAP_DEFAULTS, detection_threshold=.3, mask_threshold=.5)

MAX_AUTOMATIC_PIXELS = 8_000_000
MAX_CANDIDATE_PIXELS = 128_000_000
MAX_BATCH_MASK_PIXELS = 32_000_000


def validate_automatic_settings(settings: dict, image_pixels: int) -> dict:
    """Validate public controls and bound full-resolution logits before model work."""
    if not isinstance(settings, dict):
        raise ValueError("Inference controls must be an object.")
    unknown = set(settings) - set(PRESETS["Fast preview"])
    if unknown:
        raise ValueError("Unsupported inference controls: " + ", ".join(sorted(unknown)) + ". Crop NMS is unavailable.")
    if image_pixels > MAX_AUTOMATIC_PIXELS:
        raise ValueError("Automatic generation supports up to 8 megapixels in this release. Use point/box segmentation or import a smaller display copy.")
    if image_pixels <= 0:
        raise ValueError("Image dimensions must be positive.")
    values = {**PRESETS["Fast preview"], **settings}
    for name, low, high in (("points_per_side", 2, 32), ("points_per_batch", 1, 16), ("crop_n_layers", 0, 0)):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} to {high}. Crop pyramids are unavailable.")
    for name in ("pred_iou_thresh", "stability_score_thresh", "box_nms_thresh", "detection_threshold", "mask_threshold"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite number from 0 to 1.")
        values[name] = float(value)
    values.update(validate_overlap(values))
    values["requested_points_per_batch"] = values["points_per_batch"]
    # Three logits per sampled object. Lowering batch size changes work grouping,
    # not source resolution; requested and effective values are recorded per run.
    values["points_per_batch"] = min(values["points_per_batch"], max(1, MAX_BATCH_MASK_PIXELS // (3 * image_pixels)))
    return values


class ModelManager:
    def __init__(self, root: Path, guard):
        self.root, self.guard = root, guard
        self.model = self.predictor = self.processor = self.embeddings = None
        self.active = self.image_key = self.loading = None
        self.load_seconds = None
        self.cancel_check = None
        self.source_image = None
        import torch
        requested = os.environ.get("LSS_DEVICE", "auto")
        self.device = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
        if self.device not in ("mps", "cuda", "cpu"):
            raise ValueError("LSS_DEVICE must be auto, mps, cuda, or cpu.")
        if self.device == "cpu": torch.set_num_threads(min(8, os.cpu_count() or 1))

    def _check_cancel(self):
        if self.cancel_check is not None and self.cancel_check():
            raise InterruptedError("Inference cancelled. Its unfinished result was discarded.")

    def catalog(self):
        evidence = self.root / "setup-notes" / "compatibility.json"
        verified = json.loads(evidence.read_text(encoding="utf-8")) if evidence.is_file() else []
        text_evidence = self.root / "setup-notes/sam31-text-compatibility.json"
        if text_evidence.is_file(): verified.append(json.loads(text_evidence.read_text(encoding="utf-8")))
        result = []
        for key, entry in MODEL_CATALOG.items():
            revision = entry["revision"]
            modes = [row.get("mode") for row in verified if row.get("model") == key
                     and row.get("device") == self.device and row.get("outcome") == "passed" and "Transformers" in row.get("implementation", "")]
            installed = all((self.root / "model-cache" / entry["directory"] / item["name"]).is_file() for item in entry["files"])
            result.append(dict(
                id=key, name=entry["name"], checkpoint=entry["checkpoint"],
                revision=revision, device=self.device, implementation=IMPLEMENTATION, installed=installed,
                verified="point_box" in modes or "text" in modes,
                capabilities=dict(point="point_box" in modes, box="point_box" in modes,
                                  automatic="automatic" in modes, text="text" in modes, refinement=False),
                state="loading" if self.loading == key else "loaded" if self.active == key else "installed" if installed else "missing",
                reason=None if installed else "Use model setup to install official checkpoint files.",
                limitations="Image-only. CPU mask postprocessing and NMS. Crop layers and hole/sprinkle cleanup are unavailable.",
            ))
        return result

    def sync(self):
        import torch
        if self.device == "mps": torch.mps.synchronize()
        elif self.device == "cuda": torch.cuda.synchronize()

    def unload(self):
        self.model = self.predictor = self.processor = self.embeddings = None
        self.active = self.image_key = self.loading = None
        self.source_image = None
        gc.collect()
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def load(self, key: str):
        if key not in REVISIONS:
            raise ValueError("This adapter is unavailable for local inference.")
        self.guard.check()
        import torch
        if (self.device == "mps" and not torch.backends.mps.is_available()) or (self.device == "cuda" and not torch.cuda.is_available()):
            raise RuntimeError(f"Requested {self.device.upper()} device is unavailable. Select a supported local device and run compatibility checks.")
        if self.active == key:
            return 0.0
        self.unload()
        self.loading = key
        try:
            manifest = json.loads((self.root / "setup-notes" / "hf-model-manifest.json").read_text(encoding="utf-8"))
            entries = manifest.get("models", []) if isinstance(manifest, dict) else manifest
            entries = entries + (json.loads(LOCAL_MANIFEST.read_text(encoding="utf-8")) if LOCAL_MANIFEST.is_file() else [])
            entry = next(item for item in entries if item["id"] == key)
            if entry["revision"] != REVISIONS[key]:
                raise ValueError("Checkpoint revision differs from the adapter's pinned official revision.")
            snapshot = self.root / "model-cache" / MODEL_CATALOG[key]["directory"]
            cache = (self.root / "model-cache").resolve()
            if not snapshot.resolve().is_relative_to(cache):
                raise ValueError("Snapshot path escaped the model cache.")
            for item in entry["files"]:
                path = snapshot / item["name"]
                if not path.is_file() or not path.resolve().is_relative_to(cache):
                    raise ValueError("Checkpoint file is missing or escaped the model cache.")
                if path.stat().st_size != item["bytes"]:
                    raise ValueError(f"Checkpoint size verification failed for {item['name']}.")
                with path.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                if digest != item["sha256"]:
                    raise ValueError(f"Checkpoint SHA256 verification failed for {item['name']}.")
            from transformers import Sam2Model, Sam2Processor, SamModel, SamProcessor, Sam3Model, Sam3Processor
            model_class, processor_class = (SamModel, SamProcessor) if entry["family"] == "sam1" else (Sam2Model, Sam2Processor)
            if entry["family"] == "sam31":
                model_class, processor_class = Sam3Model, Sam3Processor
            self.sync()
            started = time.perf_counter()
            self.model = model_class.from_pretrained(str(snapshot), local_files_only=True, token=False,
                                                   dtype=torch.float32, attn_implementation="sdpa").to(self.device).eval()
            if entry["family"] == "sam31":
                from .sam31 import detector_neck_output
                self.model.vision_encoder.neck.register_forward_hook(detector_neck_output)
            self.processor = processor_class.from_pretrained(str(snapshot), local_files_only=True, token=False)
            self.predictor = self.processor  # compatibility with the manager's existing public attributes
            self.sync()
            self.load_seconds = time.perf_counter() - started
            self.active = key
            return self.load_seconds
        except Exception:
            self.unload()
            raise
        finally:
            self.loading = None

    def _embed(self, image, image_key):
        import torch
        identity = (image_key, tuple(image.shape))
        if self.embeddings is None or image_key is None or self.image_key != identity:
            processed = self.processor(images=image, return_tensors="pt")
            pixels = processed["pixel_values"].to(device=self.device, dtype=torch.float32)
            self.embeddings = self.model.get_image_embeddings(pixels)
            self.source_image = image
            self.image_key = identity
        return [[int(image.shape[0]), int(image.shape[1])]]

    def _predict(self, original_sizes, points=None, labels=None, boxes=None):
        import torch
        sam1 = MODEL_CATALOG[self.active]["family"] == "sam1"
        arguments = dict(images=self.source_image, return_tensors="pt") if sam1 else dict(original_sizes=original_sizes, return_tensors="pt")
        if points is not None:
            arguments.update(input_points=points, input_labels=labels)
        if boxes is not None:
            arguments["input_boxes"] = boxes
        processed = self.processor(**arguments)
        inputs = {key: value.to(device=self.device, dtype=torch.float32 if value.is_floating_point() else value.dtype) for key, value in processed.items() if key not in ("original_sizes", "reshaped_input_sizes", "pixel_values")}
        outputs = self.model(image_embeddings=self.embeddings, multimask_output=True, **inputs)
        # CPU placement is explicit and measured as part of the end-to-end run.
        if sam1:
            masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), original_sizes, processed["reshaped_input_sizes"], binarize=False)[0]
        else:
            masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), original_sizes,
                                                       binarize=False, max_hole_area=0, max_sprinkle_area=0,
                                                       apply_non_overlapping_constraints=False)[0]
        return masks, outputs.iou_scores[0].cpu()

    def infer(self, key, image, mode, points, box, settings, image_key=None, text=None):
        import torch
        self.guard.check()
        self._check_cancel()
        if MODEL_CATALOG[key]["family"] == "sam31":
            if mode != "text": raise ValueError("SAM3.1 supports text prompts in this image-only adapter.")
            return self._infer_text(key, image, text, settings, image_key)
        effective = validate_automatic_settings(settings, int(image.shape[0]) * int(image.shape[1])) if mode == "automatic" else None
        loaded = self.load(key)
        self.sync()
        started = time.perf_counter()
        with torch.inference_mode():
            original_sizes = self._embed(image, image_key)
            self._check_cancel()
            if mode == "interactive":
                if not points and box is None:
                    raise ValueError("Add a point or a box before running segmentation.")
                point_input = [[[[p["x"], p["y"]] for p in points]]] if points else None
                label_input = [[[p["label"] for p in points]]] if points else None
                logits, scores = self._predict(original_sizes, point_input, label_input, [[list(box)]] if box is not None else None)
                best = int(torch.argmax(scores[0]))
                results = [dict(mask=(logits[0, best] > 0).numpy(), score=float(scores[0, best]), stability=None)]
            elif mode == "automatic":
                n = effective["points_per_side"]
                batch = effective["points_per_batch"]
                height, width = image.shape[:2]
                grid = [((x + .5) * width / n, (y + .5) * height / n) for y in range(n) for x in range(n)]
                candidates, all_scores, all_stabilities, all_boxes = [], [], [], []
                candidate_bytes = 0
                from torchvision.ops import masks_to_boxes, nms
                for offset in range(0, len(grid), batch):
                    self.guard.check()
                    self._check_cancel()
                    chunk = grid[offset:offset + batch]
                    logits, scores = self._predict(original_sizes, [[[list(point)] for point in chunk]], [[[1] for _ in chunk]])
                    self._check_cancel()
                    logits, scores = logits.flatten(0, 1), scores.flatten()
                    intersections = (logits > 1.0).sum(dim=(-2, -1))
                    unions = (logits > -1.0).sum(dim=(-2, -1))
                    stability = torch.where(unions > 0, intersections.float() / unions.clamp(min=1), 0.0)
                    binary = logits > 0
                    keep = ((scores > effective["pred_iou_thresh"])
                            & (stability > effective["stability_score_thresh"])
                            & binary.flatten(1).any(1))
                    binary, scores, stability = binary[keep], scores[keep], stability[keep]
                    if len(binary):
                        all_boxes.append(masks_to_boxes(binary))
                        all_scores.append(scores)
                        all_stabilities.append(stability)
                        # Pack candidates losslessly before duplicate suppression.
                        # Keeping full-resolution boolean views also retained each
                        # entire batch, exhausting memory on ordinary 6 MP photos.
                        packed = [np.packbits(mask.numpy(), axis=None) for mask in binary]
                        candidate_bytes += sum(mask.nbytes for mask in packed)
                        if candidate_bytes > MAX_CANDIDATE_PIXELS:
                            raise MemoryError("Compressed automatic candidates exceed the 128 MB memory limit. Use fewer grid points or raise quality/stability thresholds.")
                        candidates.extend(packed)
                if candidates:
                    scores, stability, boxes = torch.cat(all_scores), torch.cat(all_stabilities), torch.cat(all_boxes)
                    indices = nms(boxes.float(), scores.float(), effective["box_nms_thresh"])
                    if len(indices) * height * width > MAX_CANDIDATE_PIXELS:
                        raise MemoryError("Selected masks exceed the 128-million-pixel result limit. Raise quality/stability thresholds or use a smaller display image.")
                    results = [dict(mask=np.unpackbits(candidates[int(index)], count=height*width).reshape(height,width).astype(bool), score=float(scores[index]),
                                    stability=float(stability[index])) for index in indices]
                else:
                    results = []
            else:
                raise ValueError("Unsupported mode. SAM 3 text inference is unavailable.")
        self.sync()
        elapsed = time.perf_counter() - started
        self.guard.check()
        self._check_cancel()
        return results, dict(load_seconds=loaded, inference_seconds=elapsed, device=self.device, dtype="float32",
                             implementation=IMPLEMENTATION, adapter="hf_" + MODEL_CATALOG[key]["family"], attention="sdpa",
                             model_input=[1024, 1024], canonical_size=[int(image.shape[1]), int(image.shape[0])],
                             cpu_fallback=False, cpu_postprocessing=True, postprocess_device="cpu",
                             postprocessing="Canonical interpolation, quality/stability filtering, and torchvision NMS on CPU",
                             hole_sprinkle_postprocessing=False, crop_layers=0,
                             effective_automatic_settings=effective)

    def _infer_text(self, key, image, text, settings, image_key):
        import torch
        from types import SimpleNamespace
        if not isinstance(text, str) or not text.strip() or len(text) > 160:
            raise ValueError("Enter a text prompt from 1 to 160 characters.")
        options = validate_automatic_settings(settings, int(image.shape[0]) * int(image.shape[1]))
        loaded = self.load(key)
        self.sync(); started = time.perf_counter()
        with torch.inference_mode():
            inputs = self.processor(images=image, text=text.strip(), return_tensors="pt").to(self.device)
            identity = (image_key, tuple(image.shape))
            if self.embeddings is None or image_key is None or self.image_key != identity:
                self.embeddings = self.model.get_vision_features(inputs.pop("pixel_values"), return_dict=True)
                self.image_key = identity
            else:
                inputs.pop("pixel_values")
            self._check_cancel()
            outputs = self.model(vision_embeds=self.embeddings, **inputs)
            self.sync(); self._check_cancel()
            cpu = SimpleNamespace(**{name: getattr(outputs, name).cpu() for name in ("pred_masks", "pred_logits", "pred_boxes", "presence_logits")})
            count = int(((cpu.pred_logits.sigmoid() * cpu.presence_logits.sigmoid().unsqueeze(1)).flatten() > options["detection_threshold"]).sum())
            if count * image.shape[0] * image.shape[1] > MAX_CANDIDATE_PIXELS:
                raise MemoryError("Text proposals exceed the result pixel budget. Raise detection confidence or use a smaller display image.")
            processed = self.processor.post_process_instance_segmentation(cpu, threshold=options["detection_threshold"], mask_threshold=options["mask_threshold"], target_sizes=[tuple(image.shape[:2])])[0]
            results = [dict(mask=mask.numpy().astype(bool), score=float(score), stability=None) for mask, score in zip(processed["masks"], processed["scores"])]
        self.guard.check(); self._check_cancel()
        return results, dict(load_seconds=loaded, inference_seconds=time.perf_counter()-started, device=self.device, dtype="float32", implementation=IMPLEMENTATION,
                             adapter="hf_sam31_detector_experimental", attention="sdpa", model_input=[1008,1008], canonical_size=[int(image.shape[1]),int(image.shape[0])],
                             cpu_postprocessing=True, cpu_fallback=False, postprocess_device="cpu", text=text.strip(),
                             effective_text_settings={name:options[name] for name in ("detection_threshold","mask_threshold")}, tracking=False)
