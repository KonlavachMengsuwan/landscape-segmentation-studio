"""Strict, local-only conversion of supplied SAM checkpoints; originals stay intact."""
import argparse,gc,json,os,shutil,sys
from pathlib import Path
from workspace import APP,ROOT,check_volume,atomic_json,digest,safe,local_environment

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',choices=['sam1-h','sam31'],required=True);args=parser.parse_args()
 os.environ.update(local_environment(offline=True))
 import torch
 from transformers import SamModel,SamProcessor,SamImageProcessor,Sam3Model,Sam3Config,Sam3Processor
 from transformers.utils import logging
 logging.set_verbosity_error()
 check_volume();key=args.model;directory='sam1-h' if key=='sam1-h' else 'sam3.1-detector'
 source=ROOT/('SAM1/sam_vit_h_4b8939.pth' if key=='sam1-h' else 'SAM3.1/sam3.1_multiplex.pt')
 expected='a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e' if key=='sam1-h' else '0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6'
 if digest(source)!=expected:raise RuntimeError('Supplied checkpoint differs from the inspected checksum. Inspect it before converting.')
 dest=safe(ROOT/'model-cache'/directory)
 if dest.exists():raise RuntimeError('Converted destination already exists and was not overwritten.')
 raw=torch.load(source,map_location='cpu',weights_only=True)
 if key=='sam1-h':
  from vendor.sam_conversion import replace_keys,get_config
  state=replace_keys(raw);config=get_config('sam_vit_h');cls=SamModel
  processor=SamProcessor(image_processor=SamImageProcessor())
  excluded=[]
 else:
  from vendor.sam3_conversion import convert_old_keys_to_new_keys,split_qkv
  logging.set_verbosity_error()
  detector={k.removeprefix('detector.'):v for k,v in raw.items() if k.startswith('detector.')}
  mapping=convert_old_keys_to_new_keys(list(detector));state={mapping[k]:v for k,v in detector.items()}
  p='vision_encoder.backbone.embeddings.position_embeddings';state[p]=state[p][:,1:,:]
  state=split_qkv(state);p='text_encoder.text_projection.weight';state[p]=state[p].T
  config=Sam3Config.from_pretrained(ROOT/'SAM3.1',local_files_only=True)
  # Multiplex detector has precisely three FPN scales (Meta model_builder.py).
  config.vision_config.scale_factors=[4.,2.,1.];cls=Sam3Model
  processor=Sam3Processor.from_pretrained(ROOT/'SAM3.1',local_files_only=True)
 with torch.device('meta'):model=cls(config)
 expected_keys=model.state_dict()
 excluded=sorted(set(state)-set(expected_keys))
 if key=='sam31':
  allowed=('backbone.vision_backbone.interactive_convs.','backbone.vision_backbone.propagation_convs.','geometry_encoder.points_')
  if any(not k.startswith(allowed) and not k.endswith('.rotary_emb.rope_embeddings') for k in excluded):raise RuntimeError('Unrecognized detector tensors: '+str(excluded))
 elif excluded:raise RuntimeError('Unexpected SAM1 tensor keys')
 state={k:v for k,v in state.items() if k in expected_keys}
 # No random initialization: every image-model tensor must be present, exact shape.
 model.load_state_dict(state,strict=True,assign=True)
 del raw,state;gc.collect()
 check_volume();dest.mkdir()
 model.save_pretrained(dest,max_shard_size='5GB');processor.save_pretrained(dest)
 check_volume();shutil.copyfile(APP/'licenses'/('SAM1-APACHE-2.0.txt' if key=='sam1-h' else 'SAM3-LICENSE.txt'),dest/'LICENSE')
 files=[dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in dest.iterdir() if p.is_file() and not p.name.startswith('._')]
 entry=dict(id=key,name='SAM 1 ViT-H (Huge)' if key=='sam1-h' else 'SAM 3.1 image detector · experimental',family='sam1' if key=='sam1-h' else 'sam31',checkpoint=str(source.relative_to(ROOT)),revision=expected,directory=directory,source_sha256=expected,files=files,excluded_detector_keys=excluded,conversion='Strict upstream HF key mapping; detector-only three FPN scales for SAM3.1')
 note=ROOT/'setup-notes/local-model-manifest.json';entries=json.loads(note.read_text(encoding="utf-8")) if note.exists() else []
 atomic_json(note,[e for e in entries if e['id']!=key]+[entry]);print(json.dumps(dict(model=key,tensors=len(expected_keys),excluded=len(excluded),outcome='strict_conversion_passed')))
if __name__=='__main__':main()
