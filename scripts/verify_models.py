"""Run genuine local compatibility checks one process at a time, with time limits."""
import argparse,subprocess,sys,json
from workspace import APP,ROOT,check_volume,local_environment,safe

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--model',default='baseline',help='baseline, all, or a catalog model ID')
 parser.add_argument('--device',choices=['auto','mps','cpu','cuda'],default='auto')
 args=parser.parse_args();environment=local_environment(offline=True);environment['LSS_DEVICE']=args.device
 entries=json.loads((APP/'config/model-manifest.json').read_text(encoding='utf-8'))
 local=ROOT/'setup-notes/local-model-manifest.json'
 if local.is_file():entries+=json.loads(local.read_text(encoding='utf-8'))
 selected=[e for e in entries if (args.model=='all' or args.model=='baseline' and e['id'] in ('tiny','small') or e['id']==args.model)]
 if not selected:raise SystemExit('Unknown model ID. See config/model-manifest.json.')
 failures=[]
 for entry in selected:
  check_volume();key=entry['id']
  if not (ROOT/'model-cache'/entry['directory']/'model.safetensors').is_file():
   failures.append(key);print(key+': install checkpoint first',flush=True);continue
  command=[sys.executable,'-B',str(APP/'scripts'/('smoke_sam31.py' if key=='sam31' else 'smoke_models.py'))]
  if key!='sam31':command+=['--model',key]
  print('Testing '+key+' locally (180-second bound)…',flush=True)
  try:
   with safe(ROOT/'setup-notes'/f'verify-{key}.log').open('w',encoding='utf-8') as log:
    result=subprocess.run(command,cwd=APP,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=180)
   if result.returncode:failures.append(key)
  except subprocess.TimeoutExpired:
   failures.append(key);print(key+': exceeded time limit; unfinished process ended',flush=True)
  print(key+(': FAILED (see setup-notes log)' if key in failures else ': passed'),flush=True)
 return 1 if failures else 0
if __name__=='__main__':raise SystemExit(main())
