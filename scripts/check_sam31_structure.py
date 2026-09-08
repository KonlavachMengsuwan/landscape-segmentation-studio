"""Inspect supplied SAM 3.1 detector against HF SAM 3 without allocating weights."""
import importlib.util,json,sys,os
from pathlib import Path
APP=Path(__file__).resolve().parents[1];ROOT=APP.parent
sys.path.insert(0,str(APP))
from scripts.workspace import check_volume,atomic_json
check_volume()
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from transformers import Sam3Config,Sam3Model
spec=importlib.util.spec_from_file_location('official_conversion',APP/'scripts/vendor/sam3_conversion.py')
converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
raw=torch.load(ROOT/'SAM3.1/sam3.1_multiplex.pt',map_location='meta',weights_only=True)
# Same detector-prefix extraction as Meta's image-model builder.
state={k.removeprefix('detector.'):v for k,v in raw.items() if k.startswith('detector.')}
mapping=converter.convert_old_keys_to_new_keys(list(state))
converted={mapping[k]:v for k,v in state.items()}
p='vision_encoder.backbone.embeddings.position_embeddings'
converted[p]=converted[p][:,1:,:]
converted=converter.split_qkv(converted)
p='text_encoder.text_projection.weight';converted[p]=converted[p].T
with torch.device('meta'):
 model=Sam3Model(Sam3Config.from_pretrained(ROOT/'SAM3.1',local_files_only=True))
expected=model.state_dict()
report=dict(checkpoint='SAM3.1/sam3.1_multiplex.pt',route='Meta detector extraction + upstream HF SAM3 key conversion',model_tensor_count=len(expected),converted_tensor_count=len(converted),missing=sorted(set(expected)-set(converted)),unexpected=sorted(set(converted)-set(expected)),shape_mismatches={k:dict(expected=list(expected[k].shape),actual=list(converted[k].shape)) for k in expected.keys() & converted.keys() if expected[k].shape!=converted[k].shape})
atomic_json(ROOT/'setup-notes/sam31-structure.json',report)
print(json.dumps(report,indent=2))
