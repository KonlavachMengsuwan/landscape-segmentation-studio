"""Single local inference worker with bounded queue and cancellable result delivery."""
from __future__ import annotations
import copy
import importlib.metadata
import queue
import time
import threading
import uuid
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from .overlap import suppress_overlaps
from .models import ModelManager, REVISIONS, MODEL_CATALOG
from .storage import StorageError, encode_mask


def now():
    return datetime.now(timezone.utc).isoformat()


class Jobs:
    def __init__(self, storage, manager=None):
        self.storage = storage
        self.manager = manager or ModelManager(storage.root, storage.guard)
        self.queue = queue.Queue(maxsize=8)
        self.records = {}
        self.latest = {}
        self.lock = threading.RLock()
        self.unload_pending = False
        self.last_error = None
        self.worker = threading.Thread(target=self._work, name="landscape-inference", daemon=True)
        self.worker.start()

    def submit(self, request):
        self.storage.check()
        with self.lock:
            self._discard_cancelled_queued()
            if self.queue.full():
                raise StorageError("The image queue is full. Wait for a job or cancel queued work.")
            key = (request["project_id"], request["image_id"])
            # A newer request supersedes every earlier result for this image.
            for record in self.records.values():
                if (record["project_id"], record["image_id"]) == key and record["state"] in ("queued", "running"):
                    record["cancelled"] = True
            self._discard_cancelled_queued()
            job_id = uuid.uuid4().hex
            record = dict(id=job_id, **copy.deepcopy(request), state="queued", message="Waiting for the local inference worker", result=None, cancelled=False, created_at=now())
            self.latest[key] = job_id
            self.records[job_id] = record
            while len(self.records) > 24:
                old = next((k for k,v in self.records.items() if v["state"] in ("completed", "failed", "cancelled")), None)
                if old is None:
                    break
                del self.records[old]
            self.queue.put_nowait(job_id)
            return self.get(job_id)

    def get(self, job_id):
        with self.lock:
            if job_id not in self.records:
                raise StorageError("Job is no longer in memory. Saved runs remain in your project.")
            return copy.deepcopy(self.records[job_id])

    def cancel(self, job_id):
        with self.lock:
            row = self.records.get(job_id)
            if row is None:
                raise StorageError("Unknown job.")
            row["cancelled"] = True
            row["result"] = None
            if row["state"] in ("queued", "completed"):
                row["state"] = "cancelled"
                row["message"] = "Result delivery cancelled. Previously saved runs remain unchanged."
            elif row["state"] == "running":
                row["message"] = "Stopping after the current model operation. Its result will be discarded."
            self._discard_cancelled_queued()
            return self.get(job_id)

    def _discard_cancelled_queued(self):
        """Release cancelled queue slots even while a model call is running.

        The caller holds self.lock. An item already taken by the worker is
        checked separately before inference starts.
        """
        pending = []
        while True:
            try:
                job_id = self.queue.get_nowait()
            except queue.Empty:
                break
            record = self.records.get(job_id) if job_id is not None else None
            if job_id is None or (record is not None and not self._cancelled(record)):
                pending.append(job_id)
            elif record is not None:
                record.update(state="cancelled", result=None, finished_at=now(), message="Cancelled or superseded before inference.")
            self.queue.task_done()
        for job_id in pending:
            self.queue.put_nowait(job_id)

    def unload(self):
        with self.lock:
            if self.unload_pending:
                return
            if any(r["state"] in ("queued", "running") for r in self.records.values()):
                raise StorageError("Cancel or finish queued work before unloading the model.")
            try:
                self.queue.put_nowait(None)
                self.unload_pending = True
            except queue.Full as exc:
                raise StorageError("The worker queue is full. Retry unloading after pending work finishes.") from exc

    def _cancelled(self, record):
        key = (record["project_id"], record["image_id"])
        return record["cancelled"] or self.latest.get(key) != record["id"]

    def _work(self):
        while True:
            job_id = self.queue.get()
            if job_id is None:
                try:
                    self.manager.unload()
                    self.last_error = None
                except Exception as exc:
                    self.last_error = f"Model unload failed: {str(exc)[:500]}. The worker remains available to retry."
                finally:
                    with self.lock:
                        self.unload_pending = False
                    self.queue.task_done()
                continue
            with self.lock:
                record = self.records.get(job_id)
            if record is None:
                self.queue.task_done()
                continue
            try:
                with self.lock:
                    if self._cancelled(record):
                        raise InterruptedError("Superseded or cancelled before inference")
                    record.update(state="running", message="Loading model and computing masks locally", started_at=now())
                self.storage.check()
                path = self.storage.image_path(record["project_id"], record["image_id"])
                with Image.open(path) as source:
                    pixels = np.array(source.convert("RGB"))
                self.manager.cancel_check = lambda: self._cancelled(record)
                masks, timing = self.manager.infer(record["model"], pixels, record["mode"], record["points"], record["box"], record["settings"], image_key=record["image_id"], **({"text":record.get("text")} if record["mode"]=="text" else {}))
                if self._cancelled(record):
                    raise InterruptedError("Cancelled result discarded")
                self.storage.check()
                run_id = uuid.uuid4().hex
                instances = []
                filter_started = time.perf_counter()
                suppression = suppress_overlaps(masks, record["settings"], lambda: self.manager._check_cancel() if hasattr(self.manager, "_check_cancel") else None)
                timing["overlap_filter_seconds"] = time.perf_counter() - filter_started
                timing["overlap_suppressed_count"] = sum(item is not None for item in suppression)
                for i, result in enumerate(masks):
                    if self._cancelled(record):
                        raise InterruptedError("Cancelled during result encoding")
                    binary = np.asarray(result["mask"])
                    if binary.ndim != 2 or binary.shape != pixels.shape[:2]:
                        raise StorageError("The model returned a mask outside the canonical source grid. Its result was discarded.")
                    binary = binary.astype(bool)
                    yy, xx = np.nonzero(binary)
                    instances.append(dict(id=uuid.uuid4().hex, run_id=run_id, label=f"{(record.get('text') or 'Region')[:150]} {i+1:02d}", status="proposed", visible=suppression[i] is None, overlap_suppression=suppression[i], color=None, original_rle=encode_mask(binary), edited_rle=None, score=result.get("score"), stability=result.get("stability"), area=int(binary.sum()), bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else [0,0,0,0]))
                run = dict(id=run_id, image_id=record["image_id"], model=record["model"], checkpoint=MODEL_CATALOG[record["model"]]["checkpoint"], revision=REVISIONS[record["model"]], implementation="Hugging Face Transformers", versions={p:importlib.metadata.version(p) for p in ("torch","torchvision","transformers","numpy","pillow")}, device=timing.get("device", "mps"), dtype=timing.get("dtype", "float32"), settings=copy.deepcopy(record["settings"]), prompts=dict(points=copy.deepcopy(record["points"]),box=copy.deepcopy(record["box"]), **({"text":record.get("text")} if record["mode"]=="text" else {})), mode=record["mode"], timing=timing, created_at=now(), masks=instances, request_token=record["request_token"], note="Reviewable instance proposals. Original proposal pixels are retained. Optional overlap suppression hides proposals and can be reversed with Show all. Scores are not scientific accuracy.")
                with self.lock:
                    if self._cancelled(record):
                        raise InterruptedError("Stale result discarded")
                    self.last_error = None
                    record.update(state="completed", message=f"Completed: {len(instances)} proposals, {sum(m['visible'] for m in instances)} visible. Save the project to persist this run.", result=run, finished_at=now())
            except InterruptedError:
                with self.lock:
                    record.update(state="cancelled", result=None, message="Cancelled or superseded. No result was applied or saved.", finished_at=now())
            except Exception as exc:
                with self.lock:
                    message = str(exc)
                    if "out of memory" in message.lower():
                        message = "Unified memory was exhausted. Unload the model and retry with a smaller image or fewer points. No result was saved."
                    record.update(state="failed", result=None, message=message[:1000], finished_at=now())
                try:
                    self.manager.unload()
                except Exception:
                    pass
            finally:
                self.manager.cancel_check = None
                self.queue.task_done()
