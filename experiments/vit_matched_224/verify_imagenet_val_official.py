#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
import pandas as pd
from scipy.io import loadmat
from torchvision import datasets
from common import sha256_file
from collections import Counter
VAL_RE=re.compile(r'ILSVRC2012_val_(\d{8})\.JPEG$')

def txt(x):
    if isinstance(x,bytes): return x.decode()
    if isinstance(x,str): return x
    try: return str(x.item())
    except Exception: return str(x)

def parse_meta(path):
    mat=loadmat(path,squeeze_me=True,struct_as_record=False)
    synsets=mat['synsets']; out={}
    for s in synsets:
        try:
            iid=int(s.ILSVRC2012_ID); wnid=txt(s.WNID); children=int(s.num_children)
        except Exception:
            continue
        if children==0: out[iid]=wnid
    if not out: raise RuntimeError('could not parse leaf synsets from meta.mat')
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--val-root',required=True); ap.add_argument('--ground-truth',required=True); ap.add_argument('--meta-mat',required=True); ap.add_argument('--output-dir',required=True); a=ap.parse_args()
    gt=[int(x) for x in Path(a.ground_truth).read_text().split()]
    if len(gt)!=50000: raise RuntimeError(f'expected 50000 GT rows, got {len(gt)}')
    id2wnid=parse_meta(a.meta_mat); ds=datasets.ImageFolder(a.val_root)
    if len(ds)!=50000 or len(ds.classes)!=1000: raise RuntimeError(f'bad ImageFolder size: images={len(ds)} classes={len(ds.classes)}')
    if len(id2wnid)!=1000 or ds.classes!=sorted(id2wnid.values()): raise RuntimeError('Dataset classes do not match the 1000 official leaf synsets')
    rows=[]; bad=[]; seen=set()
    for path,label in ds.samples:
        name=Path(path).name; m=VAL_RE.fullmatch(name)
        if not m: raise RuntimeError(f'original ILSVRC val filename not preserved: {name}')
        number=int(m.group(1))
        if not 1<=number<=50000 or number in seen: raise RuntimeError(f'Duplicate/out-of-range image number: {number}')
        seen.add(number); official_id=gt[number-1]
        if official_id not in id2wnid: raise RuntimeError(f'official class ID {official_id} unresolved')
        wnid=id2wnid[official_id]
        if wnid not in ds.class_to_idx: raise RuntimeError(f'official synset folder missing: {wnid}')
        expected=ds.class_to_idx[wnid]; actual_dir=Path(path).parent.name; ok=(label==expected and actual_dir==wnid)
        r={'image_path':path,'image_id':name,'image_number':number,'official_ilsvrc_id':official_id,'official_wnid':wnid,'expected_imagefolder_label':expected,'actual_imagefolder_label':label,'actual_class_directory':actual_dir,'matches_official_ground_truth':bool(ok)}
        rows.append(r)
        if not ok: bad.append(r)
    if seen!=set(range(1,50001)): raise RuntimeError('Incomplete official validation image coverage')
    if set(Counter(y for _,y in ds.samples).values())!={50}: raise RuntimeError('Expected 50 images per class')
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).sort_values('image_number').to_csv(out/'official_validation_manifest.csv',index=False)
    pd.DataFrame(bad).to_csv(out/'official_validation_mismatches.csv',index=False)
    (out/'class_to_idx.json').write_text(json.dumps(ds.class_to_idx,indent=2,sort_keys=True))
    rep={'ground_truth_sha256':sha256_file(a.ground_truth),'meta_mat_sha256':sha256_file(a.meta_mat),'manifest_sha256':sha256_file(out/'official_validation_manifest.csv'),'class_to_idx_sha256':sha256_file(out/'class_to_idx.json'),'unique_image_numbers':len(seen),'validation_images':len(ds),'class_directories':len(ds.classes),'official_ground_truth_rows':len(gt),'matched_images':len(rows)-len(bad),'mismatched_images':len(bad),'all_images_match_official_ground_truth':len(bad)==0}
    (out/'official_validation_check.json').write_text(json.dumps(rep,indent=2)); print(json.dumps(rep,indent=2))
    if bad: raise SystemExit(1)
if __name__=='__main__': main()
