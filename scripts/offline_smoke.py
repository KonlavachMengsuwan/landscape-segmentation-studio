"""Real inference with process-local external-network connections denied."""
import os
from pathlib import Path
import socket
import sys
import json

APP=Path(__file__).resolve().parents[1]
ROOT=APP.parent
sys.path.insert(0,str(APP))
os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
attempts=[]
original_connect=socket.socket.connect
def local_connect(self,address):
    if self.family in (socket.AF_INET,socket.AF_INET6) and address[0] not in ("127.0.0.1","::1","localhost"):
        attempts.append(str(address[0]))
        raise RuntimeError("External networking is denied inside this smoke-test process")
    return original_connect(self,address)
socket.socket.connect=local_connect
import numpy as np
from PIL import Image
from backend.models import ModelManager,PRESETS
from backend.storage import Storage

store=Storage(ROOT)
image=Image.open(APP/'samples/yosemite.jpg').convert('RGB');image.thumbnail((768,768))
manager=ModelManager(ROOT,store.guard)
results,timing=manager.infer('tiny',np.array(image),'interactive',[dict(x=360,y=170,label=1)],None,PRESETS['Fast preview'],image_key='offline-smoke')
assert results and 0<results[0]['mask'].sum()<image.width*image.height
assert attempts==[]
report=dict(outcome='passed',external_connection_attempts=attempts,loading='local_files_only + HF_HUB_OFFLINE + TRANSFORMERS_OFFLINE',network_test='Per-process connect guard; Mac global networking unchanged',timing=timing,pixels=int(results[0]['mask'].sum()))
store._write_atomic(Path('setup-notes/offline-smoke.json'),json.dumps(report,indent=2).encode())
manager.unload()
print(json.dumps(report),flush=True)
