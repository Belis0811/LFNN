#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('metrics_jsonl'); ap.add_argument('--output',required=True); a=ap.parse_args(); rec=[json.loads(x) for x in Path(a.metrics_jsonl).read_text().splitlines() if x.strip()]; rows=[]
    for r in rec:
        if r['event'] not in {'pre_training_evaluation','epoch_end'}: continue
        rows.append({'event':r['event'],'epoch':r['epoch'],'train_loss':r.get('train_loss'),'train_samples':r.get('train_samples'),'lr':r.get('lr'),'elapsed_sec':r.get('elapsed_sec'),'val_loss':r['val_loss'],'correct_top1':r['correct_top1'],'correct_top5':r['correct_top5'],'evaluated_samples':r['evaluated_samples'],'val_top1':r['val_top1'],'val_top5':r['val_top5']})
    pd.DataFrame(rows).to_csv(a.output,index=False); print(a.output)
if __name__=='__main__': main()
