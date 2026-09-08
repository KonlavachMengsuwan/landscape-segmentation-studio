"""Real local inference evidence. Never substitutes fixtures for model predictions."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
sys.path.insert(0, str(APP))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import numpy as np
from PIL import Image
import torch
from backend.models import ModelManager, PRESETS, REVISIONS
from backend.storage import Storage
from scripts.workspace import atomic_json

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=[*REVISIONS, "all"], default="all")
parser.add_argument("--mode", choices=["point_box", "point_only", "box_only", "positive_negative", "automatic", "independent", "all"], default="all")
args = parser.parse_args()
guard = Storage(ROOT).guard
guard.check()
out = ROOT / "setup-notes"
versions = {p: importlib.metadata.version(p) for p in ["torch", "torchvision", "numpy", "pillow", "transformers"]}
print(json.dumps(dict(versions=versions, mps_built=torch.backends.mps.is_built(), mps_available=torch.backends.mps.is_available())), flush=True)
manager = ModelManager(ROOT, guard)
x = torch.arange(16, device=manager.device, dtype=torch.float32).reshape(4, 4)
y = x @ x.T
manager.sync()
assert float(y[0, 0].cpu()) == 14.0
source = Image.open(APP / "samples" / "yosemite.jpg").convert("RGB")
source.thumbnail((768, 768))
source.save(out / "sample-preview.png")
array = np.array(source)
w, h = source.size
manager = ModelManager(ROOT, guard)
report_path = out / "compatibility.json"
report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else []
for key in ([args.model] if args.model != "all" else [key for key in REVISIONS if key != "sam31"]):
    for mode in (["point_only", "box_only", "positive_negative"] if args.mode=="independent" else [args.mode] if args.mode != "all" else ["point_box", "point_only", "box_only", "positive_negative", "automatic"]):
        row = dict(model=key, mode=mode, versions=versions, device=manager.device, dtype="float32", sample="National Park Service YOSE3441, resized test derivative", width=w, height=h, load_seconds=None, inference_seconds=None, outcome="failed", mask_count=None)
        try:
            # Position reviewed against the public sample before accepting evidence.
            points = [dict(x=w*.48,y=h*.32,label=1)] if mode in ("point_box","point_only","positive_negative") else []
            if mode=="positive_negative": points.append(dict(x=w*.55,y=h*.78,label=0))
            box = [w*.20,h*.20,w*.80,h*.60] if mode in ("point_box","box_only") else None
            masks, timing = manager.infer(key,array,"automatic" if mode=="automatic" else "interactive",points,box,PRESETS["Fast preview"],image_key="smoke-yosemite")
            assert masks, "No masks returned for this smoke-test input"
            assert all(m["mask"].shape == (h, w) for m in masks)
            assert any(0 < m["mask"].sum() < w*h for m in masks)
            row.update(timing, outcome="passed", mask_count=len(masks), pixels=[int(m["mask"].sum()) for m in masks])
            overlay = source.convert("RGBA")
            palette = [(36, 180, 139), (246, 169, 64), (99, 152, 239), (224, 102, 110)]
            for i, m in enumerate(masks):
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                rgba[m["mask"]] = (*palette[i % len(palette)], 100)
                overlay = Image.alpha_composite(overlay, Image.fromarray(rgba))
            overlay.save(out / f"{key}-{mode}-overlay.png")
            Image.fromarray(masks[0]["mask"].astype(np.uint8)*255).save(out / f"{key}-{mode}-mask.png")
        except Exception as exc:
            row["error"] = str(exc)
            traceback.print_exc()
        report = [r for r in report if (r.get("model"),r.get("mode"),r.get("device")) != (key,mode,manager.device)] + [row]
        guard.check()
        atomic_json(report_path, report)
        print(json.dumps(row), flush=True)
    manager.unload()
sys.exit(0 if all(r["outcome"] == "passed" for r in report if r.get("device")==manager.device and (args.model=="all" or r.get("model")==args.model) and (args.mode=="all" or args.mode=="independent" and r.get("mode") in ("point_only","box_only","positive_negative") or r.get("mode")==args.mode)) else 1)
