"""Bounded local SAM3.1 text compatibility child; run with a 180-second supervisor."""
import sys,os,time,json
from pathlib import Path
APP=Path(__file__).resolve().parents[1];ROOT=APP.parent;sys.path.insert(0,str(APP))
from scripts.workspace import local_environment,atomic_json
os.environ.update(local_environment(offline=True))
import torch,numpy as np
from PIL import Image
from backend.models import ModelManager
from backend.storage import Storage
from types import SimpleNamespace
manager=ModelManager(ROOT,Storage(ROOT).guard)
source=Image.open(APP/'samples/yosemite.jpg').convert('RGB');source.thumbnail((768,768))
report=dict(model='sam31',mode='text',implementation='Hugging Face Transformers · experimental SAM3.1 detector bridge',device=manager.device,dtype='float32',outcome='failed')
try:
 report['load_seconds']=manager.load('sam31');print('SAM3.1 loaded',flush=True)
 inputs=manager.processor(images=source,text='mountain',return_tensors='pt').to(manager.device)
 manager.sync();start=time.perf_counter()
 with torch.inference_mode():outputs=manager.model(**inputs)
 manager.sync();report['forward_seconds']=time.perf_counter()-start;print('forward',report['forward_seconds'],flush=True)
 # CPU interpolation avoids allocating full-resolution masks on the accelerator.
 cpu=SimpleNamespace(**{k:getattr(outputs,k).cpu() for k in ('pred_masks','pred_logits','pred_boxes','presence_logits')})
 result=manager.processor.post_process_instance_segmentation(cpu,threshold=.3,mask_threshold=.5,target_sizes=[(source.height,source.width)])[0]
 masks=result['masks'].numpy().astype(bool)
 assert len(masks)>0 and all(m.shape==(source.height,source.width) for m in masks)
 assert any(0<m.sum()<source.width*source.height for m in masks)
 report.update(outcome='passed',inference_seconds=time.perf_counter()-start,mask_count=len(masks),pixels=[int(m.sum()) for m in masks],prompt='mountain',canonical_size=list(source.size),model_input=list(inputs['pixel_values'].shape[-2:]),revision=__import__('backend.models',fromlist=['REVISIONS']).REVISIONS['sam31'])
 overlay=np.array(source).copy();colors=[(240,162,75),(56,185,162),(114,138,227)]
 for i,m in enumerate(masks):overlay[m]=(overlay[m]*.55+np.array(colors[i%3])*.45).astype(np.uint8)
 Image.fromarray(overlay).save(ROOT/'setup-notes/sam31-text-overlay.png')
except Exception as exc:
 import traceback;traceback.print_exc();report['error']=str(exc)
finally:
 manager.unload();atomic_json(ROOT/'setup-notes/sam31-text-compatibility.json',report)
 print(json.dumps(report),flush=True)

sys.exit(0 if report['outcome']=='passed' else 1)
