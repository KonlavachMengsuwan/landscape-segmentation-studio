"""Synthetic fixtures test lifecycle/security only, never model compatibility."""
import copy
import io
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import create_app, validate_project
from backend.jobs import Jobs
from backend.storage import Storage, VolumeGuard, StorageError, encode_mask


class FixtureManager:
    active = None
    cancel_check = None
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
    def catalog(self):
        return [dict(id="tiny",installed=True,capabilities={"point":True,"automatic":True})]
    def unload(self):
        self.active = None
    def infer(self, key,image,mode,points,box,settings,image_key=None):
        self.entered.set()
        self.release.wait(3)
        if self.cancel_check():
            raise InterruptedError()
        mask=np.zeros(image.shape[:2],bool);mask[1:3,2:5]=True
        return [dict(mask=mask,score=.9,stability=None)],dict(inference_seconds=.1)


@pytest.fixture
def local(tmp_path):
    root=tmp_path/"project";root.mkdir()
    guard=VolumeGuard(tmp_path,"fixture",lambda _:dict(VolumeUUID="fixture",MountPoint=str(tmp_path)))
    storage=Storage(root,guard)
    document=storage.create_project("Test fixture")
    data=io.BytesIO();Image.new("RGB",(9,6),(30,80,110)).save(data,"PNG")
    image=storage.import_image(document["id"],"asymmetric-fixture.png",data.getvalue())
    manager=FixtureManager();jobs=Jobs(storage,manager)
    app=create_app(storage,jobs)
    with TestClient(app,base_url="http://127.0.0.1:8765") as client:
        yield client,storage,jobs,manager,document["id"],image["id"]
    manager.release.set()


def body(local,token="first"):
    return dict(project_id=local[4],image_id=local[5],model="tiny",mode="interactive",points=[dict(x=3,y=2,label=1)],box=None,settings={},request_token=token)


def wait_job(jobs,job_id):
    deadline=time.monotonic()+4
    while time.monotonic()<deadline:
        row=jobs.get(job_id)
        if row["state"] in ("completed","cancelled","failed"):
            return row
        time.sleep(.01)
    raise AssertionError("Fixture worker did not finish")


def test_loopback_host_origin_and_csrf_header(local):
    client=local[0]
    assert client.get('/api/status').status_code==200
    assert client.get('/api/status',headers={'Host':'evil.example'}).status_code==403
    assert client.get('/api/status',headers={'Origin':'https://evil.example'}).status_code==403
    assert client.post('/api/projects',json={'name':'no header'}).status_code==403
    assert client.post('/api/projects',json={'name':'bad origin'},headers={'X-Studio-Request':'1','Origin':'https://evil.example'}).status_code==403
    assert client.post('/api/projects',json={'name':'allowed'},headers={'X-Studio-Request':'1','Origin':'http://127.0.0.1:8765'}).status_code==200


def test_canonical_prompt_bounds_and_unsupported_settings(local):
    client=local[0];headers={'X-Studio-Request':'1'}
    request=body(local);request['points'][0]['x']=9
    assert client.post('/api/jobs',json=request,headers=headers).status_code==400
    request=body(local);request['box']=[8,2,2,4]
    assert client.post('/api/jobs',json=request,headers=headers).status_code==400
    request=body(local);request['settings']={'crop_nms_thresh':.7}
    assert client.post('/api/jobs',json=request,headers=headers).status_code==400


def test_cancelled_job_discards_result_without_saving(local):
    client,storage,jobs,manager,pid,_=local
    revision=storage.load_project(pid)['revision']
    row=client.post('/api/jobs',json=body(local),headers={'X-Studio-Request':'1'}).json()
    assert manager.entered.wait(2)
    jobs.cancel(row['id']);manager.release.set()
    result=wait_job(jobs,row['id'])
    assert result['state']=='cancelled' and result['result'] is None
    assert storage.load_project(pid)['revision']==revision
    assert storage.load_project(pid)['runs']==[]


def test_newer_job_supersedes_running_job_without_overwrite(local):
    _,storage,jobs,manager,pid,_=local
    first=jobs.submit(body(local));assert manager.entered.wait(2)
    second=jobs.submit(body(local,'newer'));manager.release.set()
    assert wait_job(jobs,first['id'])['state']=='cancelled'
    completed=wait_job(jobs,second['id'])
    assert completed['state']=='completed' and completed['result']['request_token']=='newer'
    assert storage.load_project(pid)['runs']==[]
    jobs.cancel(second['id'])
    assert jobs.get(second['id'])['result'] is None


def test_drive_loss_status_and_writes_fail_without_replacement(local):
    client,storage,*_=local
    storage.guard.info_reader=lambda _:dict(VolumeUUID='different',MountPoint=str(storage.guard.mount))
    assert client.get('/api/status').json()['drive_available'] is False
    assert client.post('/api/projects',json={'name':'blocked'},headers={'X-Studio-Request':'1'}).status_code==503


def test_original_masks_and_run_provenance_remain_immutable(local):
    _,storage,_,_,pid,image_id=local
    document=storage.load_project(pid)
    mask=np.zeros((6,9),bool);mask[1:4,3:7]=True
    document['runs']=[dict(id='run1',image_id=image_id,settings={'points_per_side':8},prompts={},masks=[dict(id='mask1',status='proposed',label='Tree',original_rle=encode_mask(mask),edited_rle=None,score=.9,stability=None)])]
    validate_project(document)
    previous=copy.deepcopy(document)
    changed=copy.deepcopy(document);changed['runs'][0]['settings']['points_per_side']=16
    with pytest.raises(StorageError,match='provenance'):validate_project(changed,previous)
    changed=copy.deepcopy(document);changed['runs'][0]['masks'][0]['original_rle']=encode_mask(~mask)
    with pytest.raises(StorageError,match='Original model masks'):validate_project(changed,previous)
    changed=copy.deepcopy(document);changed['runs'][0]['masks'][0]['edited_rle']=encode_mask(~mask)
    validate_project(changed,previous)
    assert changed['runs'][0]['masks'][0]['area']==42
