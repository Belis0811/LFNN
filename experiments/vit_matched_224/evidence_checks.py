"""Portable consistency checks. Passing these does not authenticate an experiment."""
import csv
import hashlib
import json
import math
from pathlib import Path

METRIC_FIELDS=('event','epoch','train_loss','train_samples','lr','elapsed_sec',
               'val_loss','correct_top1','correct_top5','evaluated_samples','val_top1','val_top5')

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def hash_jsonable(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def require(condition,message):
    if not condition: raise ValueError(message)

def integer(value):
    n=int(value)
    require(not isinstance(value,bool) and float(value)==n,f'Invalid integer: {value}')
    return n

def equal_number(a,b,tol=1e-9):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<=tol

def check_metrics(row,n=50000,export=False):
    c1=integer(row['correct_top1']); c5=integer(row['correct_top5'])
    require(integer(row['evaluated_samples'])==n,'Wrong validation sample count')
    require(0<=c1<=c5<=n,'Invalid Top-1/Top-5 counts')
    keys=('top1_accuracy','top5_accuracy','validation_loss') if export else ('val_top1','val_top5','val_loss')
    require(equal_number(100*c1/n,row[keys[0]]),'Top-1 arithmetic mismatch')
    require(equal_number(100*c5/n,row[keys[1]]),'Top-5 arithmetic mismatch')
    require(math.isfinite(float(row[keys[2]])) and float(row[keys[2]])>=0,'Invalid validation loss')
    return c1,c5

def check_export(raw,export):
    require(len(raw)==len(export),'Raw/export row count mismatch')
    for i,(r,e) in enumerate(zip(raw,export)):
        for field in METRIC_FIELDS:
            require(field in e,f'Export missing {field}')
            if field not in r or r[field] is None:
                require(e[field]=='',f'Export row {i}: unexpected {field}')
            elif field=='event': require(r[field]==e[field],f'Export row {i}: event mismatch')
            else: require(equal_number(r[field],e[field]),f'Export row {i}: {field} mismatch')

def prediction_counts(path,labels=None,n=50000):
    seen=set(); c1=c5=0
    with open(path,newline='') as f:
        for row in csv.DictReader(f):
            name=row['image_id']; require(name not in seen,f'Duplicate prediction: {name}'); seen.add(name)
            y=integer(row['label']); top=[integer(row[f'top5_class_{k}']) for k in range(5)]
            require(0<=y<1000 and all(0<=x<1000 for x in top) and len(set(top))==5,'Invalid prediction classes')
            if labels is not None: require(name in labels and y==labels[name],f'Official label mismatch: {name}')
            a=int(top[0]==y); b=int(y in top)
            require(integer(row['top1_correct'])==a and integer(row['top5_correct'])==b,f'Prediction flags mismatch: {name}')
            c1+=a; c5+=b
    require(len(seen)==n,'Wrong number of unique prediction images')
    if labels is not None: require(seen==set(labels),'Prediction image set differs from official manifest')
    return c1,c5

def check_initial_predictions(path,labels,report):
    seen=set(); totals={'bp_correct_top1':0,'bp_correct_top5':0,'lfnn_correct_top1':0,'lfnn_correct_top5':0}
    with open(path,newline='') as f:
        for row in csv.DictReader(f):
            name=row['image_id']; require(name not in seen and name in labels,'Invalid/duplicate initial image'); seen.add(name)
            y=integer(row['label']); require(y==labels[name],f'Initial label mismatch: {name}')
            a=[integer(x) for x in row['bp_top5'].split()]; b=[integer(x) for x in row['lfnn_top5'].split()]
            require(len(a)==len(b)==5 and len(set(a))==len(set(b))==5,'Invalid initial Top-5')
            require(all(0<=x<1000 for x in a+b),'Initial class out of range')
            require(a[0]==b[0] and set(a)==set(b),'Initial predictions differ')
            require(integer(row['bp_top1'])==a[0] and integer(row['lfnn_top1'])==b[0],'Initial Top-1 columns differ')
            require(integer(row['top1_predictions_equal'])==integer(row['top5_sets_equal'])==1,'Incorrect initial agreement flags')
            for method,top in [('bp',a),('lfnn',b)]:
                totals[f'{method}_correct_top1']+=int(top[0]==y)
                totals[f'{method}_correct_top5']+=int(y in top)
    require(seen==set(labels),'Initial image coverage differs')
    for key,value in totals.items(): require(value==report[key],f'Initial report disagrees with predictions: {key}')
