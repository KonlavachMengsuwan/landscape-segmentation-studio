#!/usr/bin/env python3
"""Bounded SAM 3 image smoke test. No weight downloads or credential access.

Use --check-access for an explicitly unauthenticated public metadata/HEAD check.
Use --snapshot and --image for a local-only run after personally obtaining access.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

MODEL = "facebook/sam3"
REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
WEIGHT_BYTES = 3439938512
WEIGHT_SHA256 = "6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a"
ROOT = Path(__file__).resolve().parents[2]


def check_volume() -> None:
    """Check the mounted identity, never create a replacement mount directory."""
    note = ROOT / "setup-notes" / "volume.json"
    if not note.is_file():
        raise RuntimeError("SSD identity record is missing. Run the documented environment setup.")
    expected = json.loads(note.read_text(encoding="utf-8"))
    mount = Path(expected["mount"])
    if not mount.is_mount() or not ROOT.is_relative_to(mount.resolve()):
        raise RuntimeError("Expected SSD is unavailable. Reconnect it before continuing.")
    raw = subprocess.check_output(["diskutil", "info", "-plist", str(mount)], timeout=10)
    actual = plistlib.loads(raw)
    if actual.get("VolumeUUID") != expected["uuid"] or actual.get("MountPoint") != str(mount):
        raise RuntimeError("Mounted volume identity differs from the project SSD.")


def local_path(value: str, *, directory: bool = False) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_relative_to(ROOT):
        raise ValueError("Compatibility inputs must be inside the dedicated project workspace.")
    if directory and not path.is_dir():
        raise ValueError("Snapshot must be a directory.")
    if not directory and not path.is_file():
        raise ValueError("Image must be a file.")
    return path


def baseline() -> dict:
    return {
        "model": MODEL, "revision": REVISION, "implementation": "transformers.Sam3Model",
        "requested_dtype": "float32", "requested_attention": "sdpa",
        "dtype": None, "attention": None, "load_seconds": None,
        "inference_seconds": None, "postprocess_seconds": None,
        "actual_device": None, "input_size": None, "model_input_size": None,
        "mask_count": None, "outcome": "unavailable", "offline_loading": True,
        "mps_cpu_fallback": False,
    }


def public_access_check() -> dict:
    """urllib has no HF token lookup or account/session integration."""
    result = baseline()
    result.update(check="unauthenticated_metadata_and_head", offline_loading=None)
    headers = {"User-Agent": "LandscapeSegmentationStudio-compatibility-check/1"}
    try:
        request = urllib.request.Request(f"https://huggingface.co/api/models/{MODEL}?blobs=true", headers=headers)
        with urllib.request.urlopen(request, timeout=15) as response:
            metadata = json.load(response)
        result["public_repo_revision"] = metadata.get("sha")
        result["gate"] = metadata.get("gated")
        result["checkpoint"] = next(
            (entry for entry in metadata.get("siblings", []) if entry["rfilename"] == "model.safetensors"), None
        )
        request = urllib.request.Request(
            f"https://huggingface.co/{MODEL}/resolve/{REVISION}/model.safetensors", method="HEAD", headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                result["http_status"] = response.status
                result["outcome"] = "public_access_available_inference_not_tested"
        except urllib.error.HTTPError as error:
            result["http_status"] = error.code
            result["error_code"] = error.headers.get("X-Error-Code")
            result["reason"] = error.headers.get("X-Error-Message", "Checkpoint request failed.")
            result["outcome"] = "access_blocked" if result["error_code"] == "GatedRepo" else "access_check_failed"
        result["account_access_tested"] = False
    except Exception as error:
        result.update(outcome="access_check_failed", reason=f"{type(error).__name__}: {error}")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_output(path: Path, data: bytes) -> None:
    check_volume()
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError("Output escaped the project workspace.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        check_volume()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_worker(connection, options: dict) -> None:
    report = baseline()
    report["requested_device"] = options["device"]
    report["prompt"] = options["prompt"]
    try:
        check_volume()
        # Explicit project-local caches. Never change HOME or print environment/token values.
        for key, value in {
            "HF_HOME": str(ROOT / "model-cache" / "huggingface"),
            "HF_HUB_CACHE": str(ROOT / "model-cache" / "huggingface" / "hub"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "TMPDIR": str(ROOT / "tmp"), "XDG_CACHE_HOME": str(ROOT / "runtime" / "cache"),
        }.items():
            os.environ[key] = value
        import numpy as np
        import torch
        import transformers
        from PIL import Image, ImageOps
        from transformers import Sam3Model, Sam3Processor

        report["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__}
        report["mps_built"] = torch.backends.mps.is_built()
        report["mps_available"] = torch.backends.mps.is_available()
        if options["device"] == "mps" and not report["mps_available"]:
            raise RuntimeError("MPS is unavailable. CPU is a separate explicit compatibility test.")
        torch.set_num_threads(4)
        snapshot = local_path(options["snapshot"], directory=True)
        for entry in snapshot.iterdir():
            if entry.is_symlink() and not entry.resolve(strict=True).is_relative_to(ROOT):
                raise ValueError("Snapshot file symlink escapes the project workspace.")
        weights = snapshot / "model.safetensors"
        if not weights.is_file() or not weights.resolve().is_relative_to(ROOT):
            raise RuntimeError("Pinned model.safetensors is missing from the selected local snapshot.")
        verification_started = time.perf_counter()
        if weights.stat().st_size != WEIGHT_BYTES or sha256_file(weights) != WEIGHT_SHA256:
            raise RuntimeError("Weight size or SHA256 differs from the pinned official checkpoint.")
        report["weight_verification_seconds"] = time.perf_counter() - verification_started
        report["weight_sha256"] = WEIGHT_SHA256
        image_path = local_path(options["image"])
        source = image_path.read_bytes()
        report["source_sha256"] = hashlib.sha256(source).hexdigest()
        with Image.open(io.BytesIO(source)) as original:
            if original.format not in {"JPEG", "PNG"} or getattr(original, "n_frames", 1) != 1:
                raise ValueError("This compatibility script accepts one still JPEG or PNG.")
            if original.width * original.height > 20_000_000:
                raise ValueError("Use a sample at or below 20 megapixels for the compatibility test.")
            canonical = ImageOps.exif_transpose(original)
            if "A" in canonical.getbands() or "transparency" in canonical.info:
                rgba = canonical.convert("RGBA")
                image = Image.new("RGBA", rgba.size, "white")
                image.alpha_composite(rgba)
                image = image.convert("RGB")
            else:
                image = canonical.convert("RGB")
        report["input_size"] = [image.width, image.height]
        report["canonicalization"] = "EXIF transpose; RGB; transparency composited on white"

        def synchronize():
            if options["device"] == "mps":
                torch.mps.synchronize()

        synchronize()
        started = time.perf_counter()
        model = Sam3Model.from_pretrained(
            str(snapshot), local_files_only=True, token=False,
            dtype=torch.float32, attn_implementation="sdpa",
        ).to(options["device"]).eval()
        processor = Sam3Processor.from_pretrained(str(snapshot), local_files_only=True, token=False)
        synchronize()
        report["load_seconds"] = time.perf_counter() - started
        report["actual_device"] = str(next(model.parameters()).device)
        report["dtype"] = str(next(model.parameters()).dtype).removeprefix("torch.")
        report["attention"] = model.config._attn_implementation
        inputs = processor(images=image, text=options["prompt"], return_tensors="pt").to(options["device"])
        report["model_input_size"] = list(inputs["pixel_values"].shape[-2:][::-1])
        synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs)
        synchronize()
        report["inference_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        result = processor.post_process_instance_segmentation(
            outputs, threshold=0.5, mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]
        synchronize()
        report["postprocess_seconds"] = time.perf_counter() - started
        masks = result["masks"].detach().to("cpu").numpy().astype(bool)
        scores = result["scores"].detach().to("cpu").tolist()
        report["mask_count"] = len(masks)
        if len(masks) and tuple(masks.shape[-2:]) != (image.height, image.width):
            raise RuntimeError("Returned masks do not match the canonical image grid.")
        report["mask_pixel_counts"] = [int(mask.sum()) for mask in masks]
        report["mask_scores"] = scores
        report["mask_sha256"] = [hashlib.sha256(mask.tobytes()).hexdigest() for mask in masks]
        report["thresholds"] = {"detection": 0.5, "mask": 0.5}
        report["outcome"] = "passed_empty_result" if not len(masks) else "passed_real_inference"
        if options.get("output_dir"):
            destination = Path(options["output_dir"]).resolve()
            if not destination.is_relative_to(ROOT):
                raise ValueError("Output directory must be within the project workspace.")
            for index, mask in enumerate(masks):
                buffer = io.BytesIO()
                Image.fromarray(mask.astype(np.uint8) * 255).save(buffer, format="PNG")
                atomic_output(destination / f"mask-{index + 1:03d}.png", buffer.getvalue())
            atomic_output(destination / "compatibility.json", json.dumps(report, indent=2).encode())
        connection.send(report)
    except Exception as error:
        report.update(outcome="failed", reason=f"{type(error).__name__}: {error}")
        # A traceback containing source paths is useful locally but no environment is dumped.
        report["traceback"] = traceback.format_exc(limit=8)
        connection.send(report)
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-access", action="store_true", help="Public metadata and unauthenticated HEAD only; never downloads weights")
    parser.add_argument("--snapshot", help="Pinned Hugging Face snapshot folder within the SSD project")
    parser.add_argument("--image", help="Still JPEG/PNG sample within the SSD project")
    parser.add_argument("--prompt", default="boat")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--timeout-seconds", type=int, default=180, help="Entire subprocess limit, including model load, 30 to 600 seconds")
    parser.add_argument("--output-dir", help="Optional project-local folder for real binary mask PNGs and a report")
    args = parser.parse_args()
    if args.check_access:
        result = public_access_check()
        print(json.dumps(result, indent=2))
        return 2 if result["outcome"] != "public_access_available_inference_not_tested" else 0
    if not args.snapshot or not args.image:
        parser.error("Local inference requires --snapshot and --image. No download is automatic.")
    if not 30 <= args.timeout_seconds <= 600:
        parser.error("--timeout-seconds must be between 30 and 600.")
    if not args.prompt.strip() or len(args.prompt) > 200:
        parser.error("Provide a nonempty text prompt of at most 200 characters.")
    try:
        check_volume()
        local_path(args.snapshot, directory=True)
        local_path(args.image)
    except Exception as error:
        print(json.dumps({**baseline(), "reason": str(error)}, indent=2))
        return 2
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=run_worker, args=(child, vars(args)))
    process.start()
    child.close()
    started = time.monotonic()
    result = None
    while time.monotonic() - started < args.timeout_seconds:
        if parent.poll(0.25):
            try:
                result = parent.recv()
            except EOFError:
                pass
            break
        if not process.is_alive():
            break
    if result is None:
        running = process.is_alive()
        if running:
            process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        result = {**baseline(), "outcome": "timeout" if running else "worker_failed",
                  "reason": f"No completed inference report within {args.timeout_seconds}s." if running else f"Worker exited with code {process.exitcode}.",
                  "requested_device": args.device, "timeout_seconds": args.timeout_seconds}
    else:
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
    parent.close()
    print(json.dumps(result, indent=2))
    return 0 if result["outcome"].startswith("passed_") else 2


if __name__ == "__main__":
    sys.exit(main())
