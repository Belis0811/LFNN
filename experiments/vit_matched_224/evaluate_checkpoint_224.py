#!/usr/bin/env python3
"""Strict evaluation of a supplied checkpoint; never loads donor/pretrained weights."""
import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common import sha256_file
from data_utils import make_val_dataset_from_recipe, preprocessing_config, verify_standard_val_distribution
from vit_models import BPViT, LFNNLocalViT, extract_final_logits

def load_strict(model,path,key=None):
    # Only tensor/state-dict checkpoints are supported. No pickle fallback.
    obj=torch.load(path,map_location='cpu',weights_only=True)
    if key is not None:
        if not isinstance(obj,dict) or key not in obj: raise ValueError(f'Checkpoint key missing: {key}')
        state=obj[key]
    else:
        candidates=[k for k in ['model','state_dict'] if isinstance(obj,dict) and k in obj]
        if len(candidates)>1: raise ValueError('Ambiguous checkpoint; specify --state-dict-key')
        state=obj[candidates[0]] if candidates else obj
    if not isinstance(state,dict) or not state: raise TypeError('Nonempty state dict required')
    if all(k.startswith('module.') for k in state): state={k[len('module.'):]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True)

def select_output(output,selector):
    if selector=='final': logits=extract_final_logits(output)
    elif selector=='tensor': logits=output
    elif selector.startswith('index:'): logits=output[int(selector.split(':',1)[1])]
    elif selector.startswith('key:'): logits=output[selector.split(':',1)[1]]
    else: raise ValueError('Selector must be final, tensor, index:N, or key:NAME')
    if not torch.is_tensor(logits) or logits.ndim!=2 or logits.shape[1]!=1000: raise ValueError('Expected [batch,1000] logits')
    if not torch.isfinite(logits).all(): raise ValueError('Nonfinite logits')
    return logits

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--method',choices=['bp','lfnn'],required=True)
    ap.add_argument('--checkpoint',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--output-dir',required=True)
    ap.add_argument('--batch-size',type=int,default=256); ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--protocol',choices=['matched','historical'],default='matched')
    ap.add_argument('--model-factory',help='Historical model constructor as module:function; must not download/preload weights')
    ap.add_argument('--factory-kwargs-json',help='JSON object passed to historical constructor')
    ap.add_argument('--recipe-json',help='Explicit preprocessing JSON matching the historical run')
    ap.add_argument('--output-selector',help='Historical final-head selector: tensor, index:0, or key:NAME')
    ap.add_argument('--state-dict-key'); a=ap.parse_args()
    out=Path(a.output_dir)
    if (out/'evaluation_summary.json').exists() or (out/'predictions.csv').exists(): raise FileExistsError('Use a fresh evaluation output directory')
    if a.protocol=='historical':
        if not all([a.model_factory,a.recipe_json,a.output_selector]): ap.error('Historical evaluation requires --model-factory, --recipe-json and --output-selector')
        if a.output_selector=='final': ap.error('Specify the historical head explicitly, not the generic final selector')
        mod,name=a.model_factory.split(':',1); module=importlib.import_module(mod); factory=getattr(module,name)
        kwargs=json.loads(Path(a.factory_kwargs_json).read_text()) if a.factory_kwargs_json else {}
        if not isinstance(kwargs,dict): ap.error('Factory kwargs must be a JSON object')
        model=factory(**kwargs); recipe=json.loads(Path(a.recipe_json).read_text()); selector=a.output_selector
        source=inspect.getsourcefile(factory)
        factory_info={'name':a.model_factory,'kwargs':kwargs,'source_file':source,'source_sha256':sha256_file(source) if source else None}
    else:
        if any([a.model_factory,a.recipe_json,a.output_selector,a.factory_kwargs_json]): ap.error('Custom model/recipe arguments require --protocol historical')
        model=BPViT() if a.method=='bp' else LFNNLocalViT(); recipe=preprocessing_config()['validation']; selector='final'
        factory_info={'name':'matched_builtin'}
    load_strict(model,a.checkpoint,a.state_dict_key); model.eval().cuda()
    ds=make_val_dataset_from_recipe(a.data_root,recipe); counts=verify_standard_val_distribution(ds)
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=True,drop_last=False)
    rows=[]; c1=c5=n=0; loss_sum=0.0
    for images,targets,paths in loader:
        images=images.cuda(non_blocking=True); targets=targets.cuda(non_blocking=True)
        logits=select_output(model(images),selector)
        loss_sum+=float(F.cross_entropy(logits,targets,reduction='sum')); top5=logits.topk(5,1).indices
        t1=top5[:,0].eq(targets); t5=top5.eq(targets[:,None]).any(1)
        c1+=int(t1.sum()); c5+=int(t5.sum()); n+=targets.numel()
        tc=targets.cpu(); pc=top5.cpu(); t1c=t1.cpu(); t5c=t5.cpu()
        for i,path in enumerate(paths):
            row={'image_path':path,'image_id':Path(path).name,'label':int(tc[i]),'top1_correct':int(t1c[i]),'top5_correct':int(t5c[i])}
            row.update({f'top5_class_{k}':int(pc[i,k]) for k in range(5)}); rows.append(row)
    if n!=50000: raise RuntimeError(n)
    out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame([{'class_index':k,'count':counts[k]} for k in sorted(counts)]).to_csv(out/'class_counts.csv',index=False)
    (out/'class_to_idx.json').write_text(json.dumps(ds.class_to_idx,indent=2,sort_keys=True))
    rep={'method':a.method,'protocol':a.protocol,'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':sha256_file(a.checkpoint),
         'evaluated_samples':n,'correct_top1':c1,'correct_top5':c5,'top1_accuracy':100*c1/n,'top5_accuracy':100*c5/n,'validation_loss':loss_sum/n,
         'input_resolution':224,'preprocessing':recipe,'model_factory':factory_info['name'],'factory_details':factory_info,
         'output_selector':selector,'checkpoint_load_strict':True,'state_dict_key':a.state_dict_key,
         'evaluator_sha256':sha256_file(__file__),'torch':str(torch.__version__),'command':sys.argv,
         'predictions_sha256':sha256_file(out/'predictions.csv')}
    (out/'evaluation_summary.json').write_text(json.dumps(rep,indent=2)); print(json.dumps(rep,indent=2))
if __name__=='__main__': main()
