"""Actual local SAM3.1 cached text/empty-result check with external sockets blocked."""
import os,sys,socket,json
from pathlib import Path
APP=Path(__file__).resolve().parents[1];ROOT=APP.parent;sys.path.insert(0,str(APP))
from scripts.workspace import local_environment,atomic_json
os.environ.update(local_environment(offline=True))
attempts=[]
def blocked(self,address):
    attempts.append(str(address))
    raise RuntimeError('Network socket attempted during offline inference')
socket.socket.connect=blocked
import numpy as np
from PIL import Image
from backend.storage import Storage
from backend.models import ModelManager,PRESETS
manager=ModelManager(ROOT,Storage(ROOT).guard)
image=Image.open(APP/'samples/yosemite.jpg').convert('RGB');image.thumbnail((768,768));pixels=np.array(image)
first,t1=manager.infer('sam31',pixels,'text',[],None,PRESETS['Fast preview'],image_key='offline-text-sample',text='mountain')
embedding=id(manager.embeddings)
second,t2=manager.infer('sam31',pixels,'text',[],None,{**PRESETS['Fast preview'],'detection_threshold':1.0},image_key='offline-text-sample',text='mountain')
assert first and not second and id(manager.embeddings)==embedding and t2['load_seconds']==0
manager.unload()
assert not attempts
atomic_json(ROOT/'setup-notes/offline-expanded-smoke.json',dict(outcome='passed',device=manager.device,first_mask_count=len(first),second_mask_count=len(second),embedding_reused=True,network_attempts=attempts,first_timing=t1,cached_timing=t2))
print('SAM3.1 local text, cached embedding, and valid empty result passed with network sockets blocked.')
