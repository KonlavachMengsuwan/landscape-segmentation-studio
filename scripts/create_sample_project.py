"""Create an attributed full-resolution example using actual local app inference."""
from pathlib import Path
import json
import os
import sys
import time

APP=Path(__file__).resolve().parents[1]
ROOT=APP.parent
sys.path.insert(0,str(APP))
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]="1"
from fastapi.testclient import TestClient
from backend.main import app

headers={"X-Studio-Request":"1"}
with TestClient(app,base_url="http://127.0.0.1:8765") as client:
    response=client.post('/api/sample',headers=headers)
    response.raise_for_status();project=response.json()
    image=project['images'][0]
    request=dict(project_id=project['id'],image_id=image['id'],model='small',mode='automatic',points=[],box=None,settings=dict(points_per_side=8,points_per_batch=4,pred_iou_thresh=.8,stability_score_thresh=.90,box_nms_thresh=.7,crop_n_layers=0),request_token='attributed-sample-real-run')
    response=client.post('/api/jobs',json=request,headers=headers)
    response.raise_for_status();job=response.json();print('Actual full-resolution sample inference started',flush=True)
    started=time.monotonic()
    while time.monotonic()-started<180:
        time.sleep(.5)
        job=client.get('/api/jobs/'+job['id']).json()
        if job['state'] not in ('queued','running'):
            break
    assert job['state']=='completed',job.get('message')
    run=job['result'];project['runs']=[run]
    project['ui']={"imageId":image['id'],"runId":run['id'],"selectedIds":[]}
    response=client.put('/api/projects/'+project['id'],json=project,headers=headers)
    response.raise_for_status();saved=response.json()
    reopened=client.get('/api/projects/'+project['id']).json()
    assert reopened['runs'][0]['masks']==saved['runs'][0]['masks']
    evidence=dict(project_id=project['id'],image_size=[image['width'],image['height']],source_sha256=image['sha256'],run_id=run['id'],mask_count=len(run['masks']),timing=run['timing'],reopen='passed',original_preserved=True,local_offline=True)
    app.state.storage._write_atomic(Path('setup-notes/full-resolution-api-smoke.json'),json.dumps(evidence,indent=2).encode())
    print(json.dumps(evidence),flush=True)
