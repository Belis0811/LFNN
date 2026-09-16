#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from common import seed_everything, sha256_file
from vit_models import prepare_swag_224_state

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    seed_everything(a.seed); state,meta=prepare_swag_224_state(progress=True)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); torch.save({'seed':a.seed,'shared_vit_state':{k:v.cpu() for k,v in state.items()},'meta':meta},out)
    rep={'seed':a.seed,'sha256':sha256_file(out),'meta':meta}; out.with_suffix('.json').write_text(json.dumps(rep,indent=2)); print(json.dumps(rep,indent=2))
if __name__=='__main__': main()
