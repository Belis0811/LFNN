#!/usr/bin/env python3
import argparse, hashlib, json, platform, shutil, subprocess, sys
from pathlib import Path
import torch, torchvision
from datetime import datetime, timezone

def sha(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()
def cmd(args,cwd=None):
    try: return subprocess.check_output(args,cwd=cwd,stderr=subprocess.STDOUT,text=True).strip()
    except Exception as e: return f'UNAVAILABLE: {e}'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source-dir',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--repo-dir'); a=ap.parse_args(); src=Path(a.source_dir).resolve(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); snap=out/'source_snapshot'; shutil.rmtree(snap,ignore_errors=True); snap.mkdir(); hashes={}
    for p in sorted(src.iterdir()):
        if p.is_file() and p.suffix in {'.py','.sh','.txt','.md','.json'}: shutil.copy2(p,snap/p.name); hashes[p.name]=sha(p)
    g=[]
    for i in range(torch.cuda.device_count()):
        pr=torch.cuda.get_device_properties(i); g.append({'index':i,'name':pr.name,'total_memory':pr.total_memory,'major':pr.major,'minor':pr.minor})
    rep={'captured_at_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda_runtime_reported_by_torch':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'cuda_device_count':torch.cuda.device_count(),'gpus':g,'source_file_sha256':hashes}
    if a.repo_dir:
        repo=Path(a.repo_dir).resolve(); rep.update({'git_repo':str(repo),'git_commit':cmd(['git','rev-parse','HEAD'],repo),'git_status_porcelain':cmd(['git','status','--porcelain'],repo)})
    (out/'provenance.json').write_text(json.dumps(rep,indent=2)); print(json.dumps(rep,indent=2))
if __name__=='__main__': main()
