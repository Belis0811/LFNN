#!/usr/bin/env python3
import argparse, json, pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('predictions_csv'); a=ap.parse_args(); df=pd.read_csv(a.predictions_csv); cols=[f'top5_class_{i}' for i in range(5)]; t1=df.top5_class_0.eq(df.label); t5=df[cols].eq(df.label,axis=0).any(axis=1)
    if 'top1_correct' in df and not (t1.astype(int).values==df.top1_correct.astype(int).values).all(): raise RuntimeError('Top-1 flags disagree with prediction columns')
    if 'top5_correct' in df and not (t5.astype(int).values==df.top5_correct.astype(int).values).all(): raise RuntimeError('Top-5 flags disagree with prediction columns')
    print(json.dumps({'evaluated_samples':len(df),'correct_top1':int(t1.sum()),'correct_top5':int(t5.sum()),'top1_accuracy':100*t1.mean(),'top5_accuracy':100*t5.mean()},indent=2))
if __name__=='__main__': main()
