"""Focused API race, body-limit, worker recovery, and responsiveness tests."""
import copy
import io
import threading
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image

from backend import main
from backend.main import validate_project
from backend.storage import StorageError, encode_mask
from test_api_jobs import local, body, wait_job

HEADERS = {"X-Studio-Request": "1"}


def test_streamed_body_limit_does_not_trust_content_length(local, monkeypatch):
    monkeypatch.setattr(main, "MAX_REQUEST_BYTES", 100)
    response = local[0].post("/api/projects", content=iter([b'{"name":"', b"x" * 200, b'"}']),
                             headers={**HEADERS, "Content-Type": "application/json"})
    assert response.status_code == 413
    response = local[0].post("/api/projects", content=b"{}", headers={**HEADERS, "Content-Length": "-1"})
    assert response.status_code == 400


def test_save_requires_matching_id_and_loaded_revision(local):
    client, storage, _, _, project_id, _ = local
    previous = storage.load_project(project_id)
    for field, value in (("revision", None), ("revision", True), ("id", "different")):
        changed = copy.deepcopy(previous)
        changed[field] = value
        response = client.put(f"/api/projects/{project_id}", json=changed, headers=HEADERS)
        assert response.status_code == 400
    assert storage.load_project(project_id) == previous


def test_disconnected_multipart_is_rejected_before_spool_or_decode(local, monkeypatch):
    import starlette.formparsers
    import backend.storage
    client, storage, _, _, project_id, _ = local
    touched = []

    def forbidden(*args, **kwargs):
        touched.append(True)
        raise AssertionError("Disconnected requests must not parse or spool image uploads")

    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", forbidden)
    monkeypatch.setattr(backend.storage, "decode_image", forbidden)
    storage.guard.info_reader = lambda _: {"VolumeUUID": "disconnected", "MountPoint": str(storage.guard.mount)}
    response = client.post(f"/api/projects/{project_id}/images", files={"files": ("fixture.png", b"x" * 2_000_000, "image/png")}, headers=HEADERS)
    assert response.status_code == 503
    assert not touched


def test_pinned_temp_directory_does_not_fall_back_when_missing(local):
    _, storage, *_ = local
    expected = storage.safe_path("tmp")
    assert tempfile.tempdir == str(expected)
    displaced = storage.safe_path("tmp-controlled-missing")
    expected.rename(displaced)
    try:
        with pytest.raises(FileNotFoundError):
            with tempfile.TemporaryFile():
                pass
        assert not expected.exists()
    finally:
        displaced.rename(expected)


def test_cancellation_remains_available_with_disconnected_ssd(local):
    client, storage, jobs, manager, _, _ = local
    row = jobs.submit(body(local))
    assert manager.entered.wait(2)
    storage.guard.info_reader = lambda _: {"VolumeUUID": "disconnected", "MountPoint": str(storage.guard.mount)}
    response = client.post(f"/api/jobs/{row['id']}/cancel", headers=HEADERS)
    assert response.status_code == 200
    manager.release.set()
    assert wait_job(jobs, row["id"])["state"] == "cancelled"


@pytest.mark.parametrize("invalid", ["images-none", "image-none", "masks-none", "original-string", "edited-only", "wrong-run", "label-object"])
def test_malformed_project_masks_are_recoverable_errors(local, invalid):
    _, storage, _, _, project_id, image_id = local
    document = storage.load_project(project_id)
    mask = dict(id="mask1", run_id="run1", status="proposed", label="Fixture", original_rle=encode_mask(np.ones((6, 9), bool)), edited_rle=None)
    document["runs"] = [dict(id="run1", image_id=image_id, masks=[mask])]
    if invalid == "images-none": document["images"] = None
    elif invalid == "image-none": document["images"] = [None]
    elif invalid == "masks-none": document["runs"][0]["masks"] = None
    elif invalid == "original-string": mask["original_rle"] = "not-rle"
    elif invalid == "edited-only": mask["edited_rle"] = mask.pop("original_rle")
    elif invalid == "wrong-run": mask["run_id"] = "other"
    elif invalid == "label-object": mask["label"] = {"invalid": "React text"}
    with pytest.raises(StorageError):
        validate_project(document)


def test_unavailable_catalog_model_and_missing_export_are_recoverable(local):
    client, _, _, _, project_id, _ = local
    request = body(local)
    request["model"] = "small"  # Fixture catalog intentionally has only Tiny.
    assert client.post("/api/jobs", json=request, headers=HEADERS).status_code == 400
    assert client.get(f"/api/exports/{project_id}/missing.zip").status_code == 400


def test_concurrent_uploads_recheck_the_image_limit_under_writer_lock(local):
    client, storage, _, _, project_id, _ = local
    data = io.BytesIO()
    Image.new("RGB", (2, 3)).save(data, "PNG")

    def upload():
        files = [("files", (f"fixture-{index}.png", data.getvalue(), "image/png")) for index in range(8)]
        return client.post(f"/api/projects/{project_id}/images", files=files, headers=HEADERS)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(upload) for _ in range(2)]
        responses = [future.result(timeout=20) for future in futures]
    assert sorted(response.status_code for response in responses) == [200, 400]
    assert len(storage.load_project(project_id)["images"]) == 9


@pytest.mark.parametrize("operation", ["upload", "export", "import"])
def test_expensive_async_routes_leave_status_responsive(local, monkeypatch, operation):
    client, storage, _, _, project_id, _ = local
    entered, release = threading.Event(), threading.Event()

    def pause():
        entered.set()
        assert release.wait(4)

    if operation == "upload":
        from backend import storage as storage_module
        actual_decode = storage_module.decode_image

        def delayed_decode(data):
            pause()
            return actual_decode(data)

        monkeypatch.setattr(storage_module, "decode_image", delayed_decode)
        data = io.BytesIO()
        Image.new("RGB", (2, 3)).save(data, "PNG")
        call = lambda: client.post(f"/api/projects/{project_id}/images", files={"files": ("fixture.png", data.getvalue(), "image/png")}, headers=HEADERS)
    elif operation == "export":
        from backend import exports

        def delayed_export(*args):
            pause()
            return storage.export_path(project_id, "synthetic-fixture.zip")

        monkeypatch.setattr(exports, "build_export", delayed_export)
        call = lambda: client.post(f"/api/projects/{project_id}/export", data={"run_id": "fixture"}, headers=HEADERS)
    else:
        from backend import exports

        def delayed_import(*args):
            pause()
            return {"fixture": "no real import performed"}

        monkeypatch.setattr(exports, "import_bundle", delayed_import)
        call = lambda: client.post("/api/import-bundle", files={"file": ("synthetic.zip", b"fixture", "application/zip")}, headers=HEADERS)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(call)
        try:
            assert entered.wait(2)
            started = time.monotonic()
            response = client.get("/api/status")
            assert response.status_code == 200
            assert time.monotonic() - started < 1
        finally:
            release.set()
        assert future.result(timeout=10).status_code == 200


def test_cancelled_queue_slots_are_reusable_during_active_inference(local):
    _, _, jobs, manager, _, _ = local
    first = jobs.submit(body(local))
    assert manager.entered.wait(2)
    queued = []
    for index in range(8):
        request = body(local, f"queued-{index}")
        request["image_id"] = f"synthetic-queued-{index}"
        queued.append(jobs.submit(request))
    assert jobs.queue.full()
    for row in queued:
        jobs.cancel(row["id"])
    assert jobs.queue.qsize() == 0
    next_job = jobs.submit(body(local, "after-cancellation"))
    manager.release.set()
    assert wait_job(jobs, first["id"])["state"] == "cancelled"
    assert wait_job(jobs, next_job["id"])["state"] == "completed"


def test_unload_failure_does_not_kill_worker_or_hide_error(local, monkeypatch):
    client, _, jobs, manager, _, _ = local
    calls = 0

    def failed_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("Synthetic unload failure")

    monkeypatch.setattr(manager, "unload", failed_once)
    jobs.unload()
    deadline = time.monotonic() + 3
    while jobs.unload_pending and time.monotonic() < deadline:
        time.sleep(.01)
    assert jobs.worker.is_alive()
    assert "Synthetic unload failure" in jobs.last_error
    assert "Synthetic unload failure" in client.get("/api/status").json()["worker_error"]
    manager.release.set()
    row = jobs.submit(body(local))
    completed = wait_job(jobs, row["id"])
    assert completed["state"] == "completed"
    assert completed["result"]["masks"][0]["color"] is None


def test_repeated_unload_requests_are_deduplicated(local, monkeypatch):
    _, _, jobs, manager, _, _ = local
    entered, release = threading.Event(), threading.Event()

    def delayed_unload():
        entered.set()
        release.wait(3)

    monkeypatch.setattr(manager, "unload", delayed_unload)
    try:
        jobs.unload()
        assert entered.wait(2)
        for _ in range(20):
            jobs.unload()
        assert jobs.queue.qsize() == 0
    finally:
        release.set()
