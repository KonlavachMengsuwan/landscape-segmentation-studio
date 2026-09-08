"""Loopback-only API for the local, image-only research workspace."""
from __future__ import annotations

import copy
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from .jobs import Jobs
from .models import PRESETS, validate_automatic_settings
from .storage import Storage, StorageError, VolumeUnavailable, RevisionConflict, decode_mask, MAX_UPLOAD_BYTES

ROOT = Path(__file__).resolve().parents[2]
PORT = int(os.environ.get("STUDIO_PORT", "8765"))
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
MAX_REQUEST_BYTES = 220 * 1024 * 1024


class RequestBodyLimitMiddleware:
    """Count streamed bytes too; Content-Length alone is not a body limit."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_REQUEST_BYTES:
                    raise HTTPException(status_code=413, detail="This request exceeds the 220 MiB limit.")
            return message

        await self.app(scope, limited_receive, send)


class ProjectName(BaseModel):
    name: str = Field(default="Untitled landscape", max_length=160)


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float
    label: Literal[0, 1]


class JobInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str
    image_id: str
    model: str = Field(min_length=1, max_length=40)
    mode: Literal["interactive", "automatic", "text"]
    text: str | None = Field(default=None, max_length=160)
    points: list[Point] = Field(default_factory=list, max_length=128)
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)
    settings: dict = Field(default_factory=lambda: copy.deepcopy(PRESETS["Fast preview"]))
    request_token: str = Field(min_length=1, max_length=120)


def validate_project(document, previous=None):
    if not isinstance(document, dict):
        raise StorageError("Project data must be an object.")
    image_records = document.get("images", [])
    if (not isinstance(image_records, list) or len(image_records) > 16
            or any(not isinstance(image, dict) or not isinstance(image.get("id"), str)
                   or type(image.get("width")) is not int or type(image.get("height")) is not int
                   for image in image_records)):
        raise StorageError("Project images must contain up to 16 valid image records.")
    images = {image["id"]: image for image in image_records}
    if len(images) != len(image_records):
        raise StorageError("Image IDs must be unique within a project.")
    runs = document.get("runs", [])
    if not isinstance(runs, list) or len(runs) > 128:
        raise StorageError("A project can contain up to 128 runs. Export a bundle before starting another project.")
    old_runs = {r["id"]:r for r in (previous or {}).get("runs", [])}
    run_ids = set()
    for run in runs:
        if not isinstance(run, dict) or not SAFE_ID.fullmatch(str(run.get("id", ""))) or run["id"] in run_ids:
            raise StorageError("Run IDs must be valid and unique.")
        run_ids.add(run["id"])
        image = images.get(run.get("image_id"))
        if image is None:
            raise StorageError("A run must refer to an image in this project.")
        masks = run.get("masks", [])
        if not isinstance(masks, list) or len(masks) > 1024:
            raise StorageError("Too many mask instances in this run.")
        old = old_runs.get(run["id"])
        if old:
            for field in ("image_id","model","checkpoint","revision","implementation","versions","device","dtype","settings","prompts","timing","created_at","mode","request_token"):
                if field in old and run.get(field) != old[field]:
                    raise StorageError("Saved run provenance is immutable. Run the model again to create a new result.")
        old_masks = {m["id"]:m for m in old.get("masks", [])} if old else {}
        ids = set()
        for mask in run.get("masks", []):
            if not isinstance(mask, dict) or not SAFE_ID.fullmatch(str(mask.get("id", ""))) or mask["id"] in ids:
                raise StorageError("Mask IDs must be valid and unique within a run.")
            ids.add(mask["id"])
            if mask.get("run_id", run["id"]) != run["id"]:
                raise StorageError("Mask run identity does not match its parent run.")
            mask.setdefault("run_id", run["id"])
            if mask["id"] in old_masks and mask.get("original_rle") != old_masks[mask["id"]].get("original_rle"):
                raise StorageError("Original model masks are immutable. Store corrections in edited_rle.")
            if mask["id"] in old_masks and mask.get("overlap_suppression") != old_masks[mask["id"]].get("overlap_suppression"):
                raise StorageError("Overlap-filter provenance is immutable. Change visibility to recover a proposal.")
            if not isinstance(mask.get("original_rle"), dict):
                raise StorageError("Every mask needs lossless original membership.")
            expected = [image["height"],image["width"]]
            for field in ("original_rle", "edited_rle"):
                rle = mask.get(field)
                if rle is not None:
                    if not isinstance(rle, dict) or rle.get("size") != expected:
                        raise StorageError("A mask is not aligned to its source's canonical grid.")
                    decode_mask(rle)
            rle = mask.get("edited_rle") or mask.get("original_rle")
            if rle is None:
                raise StorageError("Every mask needs lossless original membership.")
            array = decode_mask(rle)
            yy,xx = np.nonzero(array)
            mask["area"] = int(array.sum())
            mask["bbox"] = [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else [0,0,0,0]
            if mask.get("status") not in ("proposed","accepted","rejected"):
                raise StorageError("Unknown mask review status.")
            if not isinstance(mask.get("label", ""), str) or len(mask.get("label", "")) > 160:
                raise StorageError("Keep labels within 160 characters.")
            for field in ("score","stability"):
                if mask.get(field) is not None and (type(mask[field]) not in (int,float) or not math.isfinite(mask[field])):
                    raise StorageError("Mask scores must be finite or unavailable.")
    return document


def create_app(storage=None, jobs=None):
    storage = storage or Storage(ROOT)
    # Pin Python's cached temporary directory. Merely setting TMPDIR allows
    # tempfile to select an internal fallback if the SSD disappears before
    # its first use. This process serves one dedicated workspace.
    tempfile.tempdir = str(storage._mkdir("tmp"))
    jobs = jobs or Jobs(storage)
    app = FastAPI(title="Landscape Segmentation Studio", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(RequestBodyLimitMiddleware)
    app.state.storage, app.state.jobs = storage, jobs
    allowed = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        host = request.headers.get("host", "")
        origin = request.headers.get("origin")
        if host not in allowed or (origin is not None and origin not in {"http://"+v for v in allowed}):
            return JSONResponse({"detail":"This application accepts same-origin loopback requests only."},status_code=403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail":"Cross-site requests are not allowed."},status_code=403)
        if request.method not in ("GET","HEAD") and request.headers.get("x-studio-request") != "1":
            return JSONResponse({"detail":"Missing local application request header."},status_code=403)
        try:
            length = int(request.headers.get("content-length", "0"))
            if length < 0:
                return JSONResponse({"detail":"Invalid request size."},status_code=400)
            if length > MAX_REQUEST_BYTES:
                return JSONResponse({"detail":"This request exceeds the 220 MiB limit."},status_code=413)
        except ValueError:
            return JSONResponse({"detail":"Invalid request size."},status_code=400)
        memory_only = (request.url.path == "/api/models/unload"
                       or re.fullmatch(r"/api/jobs/[A-Za-z0-9_-]{1,80}/cancel", request.url.path))
        if request.method not in ("GET", "HEAD") and not memory_only:
            try:
                # This runs before FastAPI's multipart parser can create a
                # spooled upload, not just before the endpoint's final write.
                await run_in_threadpool(storage.check)
            except (StorageError, OSError) as exc:
                return JSONResponse({"detail": str(exc)}, status_code=503)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; font-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(StorageError)
    async def storage_error(request, exc):
        status = 503 if isinstance(exc,VolumeUnavailable) else 409 if isinstance(exc,RevisionConflict) else 400
        return JSONResponse({"detail":str(exc)},status_code=status)

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({"detail":str(exc)},status_code=400)

    @app.get("/api/status")
    def status():
        try:
            storage.check()
            return dict(ok=True,drive_available=True,error=None,models=jobs.manager.catalog(),presets=PRESETS,active_model=jobs.manager.active,worker_error=jobs.last_error,local_only=True,version="0.2.1",instance=os.environ.get("LSS_INSTANCE"))
        except (StorageError,OSError) as exc:
            return dict(ok=False,drive_available=False,error=str(exc),models=[],presets=PRESETS)

    @app.get("/api/projects")
    def projects():
        return storage.list_projects()

    @app.post("/api/projects")
    def new_project(body: ProjectName):
        return storage.create_project(body.name)

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str):
        return storage.load_project(project_id)

    @app.put("/api/projects/{project_id}")
    def save_project(project_id: str, document: dict):
        if document.get("id") != project_id or type(document.get("revision")) is not int or document["revision"] < 1:
            raise StorageError("Save the complete project with its matching ID and last loaded revision.")
        # Validation and commit share the writer lock, so provenance is checked
        # against the same version used by the optimistic revision check.
        with storage._lock:
            previous = storage.load_project(project_id)
            validate_project(document, previous)
            return storage.save_project(project_id, document)

    @app.post("/api/projects/{project_id}/images")
    async def upload_images(project_id: str, files: list[UploadFile] = File(...)):
        if not files or len(files) > 16:
            raise StorageError("Keep up to 16 still images in one project.")
        from .storage import decode_image
        uploads=[]
        for file in files:
            data=await file.read(MAX_UPLOAD_BYTES+1)
            await run_in_threadpool(decode_image, data)  # Validate selection before importing any file.
            uploads.append((file.filename or "image",data))

        def import_selection():
            with storage._lock:
                project = storage.load_project(project_id)
                if len(uploads) + len(project["images"]) > 16:
                    raise StorageError("Keep up to 16 still images in one project.")
                for name, data in uploads:
                    storage.import_image(project_id, name, data)
                return storage.load_project(project_id)

        return await run_in_threadpool(import_selection)

    @app.get("/api/projects/{project_id}/images/{image_id}")
    def get_image(project_id: str, image_id: str):
        return FileResponse(storage.image_path(project_id,image_id),media_type="image/png")

    @app.post("/api/projects/{project_id}/images/{image_id}/relink")
    async def relink_image(project_id: str, image_id: str, file: UploadFile=File(...)):
        data=await file.read(MAX_UPLOAD_BYTES+1)
        await run_in_threadpool(storage.relink_image,project_id,image_id,file.filename or "image",data)
        return await run_in_threadpool(storage.load_project,project_id)

    @app.post("/api/sample")
    def sample():
        project=storage.create_project("Yosemite · landscape study")
        storage.import_image(project["id"],"Yosemite · NPS public-domain sample.jpg",storage.safe_path("app/samples/yosemite.jpg").read_bytes())
        return storage.load_project(project["id"])

    @app.post("/api/jobs")
    def submit(body: JobInput):
        project=storage.load_project(body.project_id)
        image=next((im for im in project["images"] if im["id"]==body.image_id),None)
        if image is None:
            raise StorageError("Choose an image from this project.")
        model=next((m for m in jobs.manager.catalog() if m["id"]==body.model),None)
        capability="point" if body.mode=="interactive" else body.mode
        if model is None or not model["installed"] or not model["capabilities"].get(capability,False):
            raise StorageError("This model/mode has not passed local compatibility checks or its weights are missing.")
        if body.mode == "text" and not (body.text and body.text.strip()):
            raise StorageError("Enter a text prompt before running.")
        width,height=image["width"],image["height"]
        for point in body.points:
            if not math.isfinite(point.x) or not math.isfinite(point.y) or not (0 <= point.x < width and 0 <= point.y < height):
                raise StorageError("Prompt points must lie within the canonical image grid.")
        if body.box is not None:
            box=body.box
            if len(box)!=4 or not all(math.isfinite(x) for x in box) or not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):
                raise StorageError("Draw a nonempty box within the canonical image.")
        if body.mode=="interactive" and not body.points and body.box is None:
            raise StorageError("Add a point or draw a box first.")
        validate_automatic_settings(body.settings,width*height if body.mode in ("automatic", "text") else min(width*height,8_000_000))
        request=body.model_dump();request["settings"]={**PRESETS["Fast preview"],**body.settings}
        return jobs.submit(request)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return jobs.get(job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return jobs.cancel(job_id)

    @app.post("/api/models/unload")
    def unload():
        jobs.unload()
        return {"ok":True,"message":"Model unload queued"}

    @app.post("/api/projects/{project_id}/export")
    async def export(project_id: str, run_id: str=Form(...), styled: UploadFile | None=File(None), overlay: UploadFile | None=File(None)):
        from .exports import build_export
        project=await run_in_threadpool(storage.load_project, project_id)
        styled_bytes=await styled.read(MAX_UPLOAD_BYTES+1) if styled else None
        overlay_bytes=await overlay.read(MAX_UPLOAD_BYTES+1) if overlay else None
        if any(data is not None and len(data) > MAX_UPLOAD_BYTES for data in (styled_bytes,overlay_bytes)):
            raise StorageError("Each rendered PNG must be at most 100 MiB.")
        path=await run_in_threadpool(build_export,storage,project,run_id,styled_bytes,overlay_bytes)
        return dict(url=f"/api/exports/{project_id}/{path.name}",name=path.name)

    @app.get("/api/exports/{project_id}/{filename}")
    def download(project_id: str,filename: str):
        storage.check()
        path=storage.export_path(project_id,filename)
        if not path.is_file():
            raise StorageError("This export file is unavailable. Export the saved run again.")
        return FileResponse(path,filename=filename,media_type="application/zip")

    @app.post("/api/import-bundle")
    async def reopen_bundle(file: UploadFile=File(...)):
        from .exports import import_bundle
        data=await file.read(200*1024*1024+1)
        return await run_in_threadpool(import_bundle,storage,data)

    static=storage.safe_path("app/frontend/dist")
    if static.is_dir():
        app.mount("/",StaticFiles(directory=static,html=True),name="studio")
    return app


app = create_app()
