#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def read(path):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]; pre=next(r for r in rows if r['event']=='pre_training_evaluation'); ep=[r for r in rows if r['event']=='epoch_end'];
    if not ep or [r['epoch'] for r in ep]!=list(range(len(ep))): raise ValueError('Epoch records must be complete and ordered')
    return pre,ep[-1],max(ep,key=lambda r:r['val_top1'])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--seeds',nargs='+',type=int,required=True); a=ap.parse_args(); rows=[]
    for s in a.seeds:
        for m in ['bp','lfnn']:
            pre,final,best=read(Path(a.root)/f'seed_{s}'/m/'metrics.jsonl'); rows.append({'seed':s,'method':m,'pretrain_correct_top1':pre['correct_top1'],'pretrain_correct_top5':pre['correct_top5'],'pretrain_top1':pre['val_top1'],'pretrain_top5':pre['val_top5'],'final_correct_top1':final['correct_top1'],'final_correct_top5':final['correct_top5'],'final_top1':final['val_top1'],'final_top5':final['val_top5'],'best_epoch':best['epoch'],'best_top1':best['val_top1'],'best_top5_at_best_top1_epoch':best['val_top5']})
    df=pd.DataFrame(rows); df.to_csv(Path(a.root)/'paired_summary.csv',index=False); print(df.to_string(index=False)); bp=df[df.method=='bp'].set_index('seed'); lf=df[df.method=='lfnn'].set_index('seed'); d=np.array([lf.loc[s,'final_top1']-bp.loc[s,'final_top1'] for s in a.seeds]); print(f"\nBP final Top-1: {bp.final_top1.mean():.4f} ± {bp.final_top1.std(ddof=1):.4f}"); print(f"LFNN final Top-1: {lf.final_top1.mean():.4f} ± {lf.final_top1.std(ddof=1):.4f}"); print(f"Paired LFNN-BP: {d.mean():.4f} ± {d.std(ddof=1):.4f}")
if __name__=='__main__': main()
