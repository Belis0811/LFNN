#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from common import sha256_file
from data_utils import make_val_dataset, verify_standard_val_distribution
from vit_models import BPViT, LFNNLocalViT, extract_final_logits

def upd(h,t): h.update(t.detach().float().cpu().contiguous().numpy().tobytes())

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--shared-init',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--workers',type=int,default=8); ap.add_argument('--save-full-logits',action='store_true'); a=ap.parse_args()
    state=torch.load(a.shared_init,map_location='cpu',weights_only=True)['shared_vit_state']; bp=BPViT(state).eval().cuda(); lf=LFNNLocalViT(state).eval().cuda()
    ds=make_val_dataset(a.data_root); verify_standard_val_distribution(ds); loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=True,drop_last=False)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); bh=hashlib.sha256(); lh=hashlib.sha256(); rows=[]; b_all=[]; l_all=[]
    maxd=0.0; sumd=0.0; nel=0; exact=0; agree1=0; agree5=0; bc1=bc5=lc1=lc5=total=0
    for images,targets,paths in loader:
        images=images.cuda(non_blocking=True); targets=targets.cuda(non_blocking=True)
        b=extract_final_logits(bp(images)); l=extract_final_logits(lf(images)); d=(b-l).abs(); maxd=max(maxd,float(d.max())); sumd+=float(d.sum()); nel+=d.numel(); exact+=int(d.eq(0).sum())
        upd(bh,b); upd(lh,l)
        if a.save_full_logits: b_all.append(b.float().cpu().numpy()); l_all.append(l.float().cpu().numpy())
        b5=b.topk(5,1).indices; l5=l.topk(5,1).indices; agree1+=int(b5[:,0].eq(l5[:,0]).sum()); agree5+=int(torch.sort(b5,1).values.eq(torch.sort(l5,1).values).all(1).sum())
        bt1=b5[:,0].eq(targets); lt1=l5[:,0].eq(targets); bt5=b5.eq(targets[:,None]).any(1); lt5=l5.eq(targets[:,None]).any(1)
        bc1+=int(bt1.sum()); lc1+=int(lt1.sum()); bc5+=int(bt5.sum()); lc5+=int(lt5.sum()); total+=targets.numel()
        b5c=b5.cpu(); l5c=l5.cpu(); tc=targets.cpu()
        for i,p in enumerate(paths): rows.append({'image_path':p,'image_id':Path(p).name,'label':int(tc[i]),'bp_top1':int(b5c[i,0]),'lfnn_top1':int(l5c[i,0]),'bp_top5':' '.join(map(str,b5c[i].tolist())),'lfnn_top5':' '.join(map(str,l5c[i].tolist())),'top1_predictions_equal':int(b5c[i,0]==l5c[i,0]),'top5_sets_equal':int(set(b5c[i].tolist())==set(l5c[i].tolist()))})
    if total!=50000: raise RuntimeError(total)
    if a.save_full_logits:
        np.save(out/'bp_initial_final_logits.npy',np.concatenate(b_all,0)); np.save(out/'lfnn_initial_final_logits.npy',np.concatenate(l_all,0))
    pd.DataFrame(rows).to_csv(out/'initial_prediction_comparison.csv',index=False)
    rep={'shared_init_sha256':sha256_file(a.shared_init),'evaluated_samples':total,'same_preprocessed_batch_used_for_both_models':True,'bp_final_logits_sha256':bh.hexdigest(),'lfnn_final_logits_sha256':lh.hexdigest(),'final_logit_hashes_equal':bh.hexdigest()==lh.hexdigest(),'max_abs_final_logit_difference':maxd,'mean_abs_final_logit_difference':sumd/nel,'exactly_equal_logit_elements':exact,'total_logit_elements':nel,'top1_prediction_agreement_count':agree1,'top1_prediction_agreement_rate':agree1/total,'top5_set_agreement_count':agree5,'top5_set_agreement_rate':agree5/total,'bp_correct_top1':bc1,'lfnn_correct_top1':lc1,'bp_correct_top5':bc5,'lfnn_correct_top5':lc5,'bp_top1_accuracy':100*bc1/total,'lfnn_top1_accuracy':100*lc1/total,'bp_top5_accuracy':100*bc5/total,'lfnn_top5_accuracy':100*lc5/total}
    (out/'initial_logit_comparison.json').write_text(json.dumps(rep,indent=2)); print(json.dumps(rep,indent=2))
    if agree1!=total or agree5!=total or bc1!=lc1 or bc5!=lc5 or maxd>1e-6: raise SystemExit('matched-initialization check FAILED')
if __name__=='__main__': main()
