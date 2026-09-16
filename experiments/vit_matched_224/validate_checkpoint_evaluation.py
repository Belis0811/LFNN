#!/usr/bin/env python3
"""Recount a reload evaluation and check its labels against the verified manifest."""
import argparse
import csv
import json
from pathlib import Path
from evidence_checks import require,check_metrics,prediction_counts,sha256_file

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluation-dir',required=True);ap.add_argument('--validation-manifest-dir',required=True);ap.add_argument('--checkpoint');a=ap.parse_args()
    folder=Path(a.evaluation_dir);official=Path(a.validation_manifest_dir)
    summary=json.loads((folder/'evaluation_summary.json').read_text());report=json.loads((official/'official_validation_check.json').read_text())
    require(report['all_images_match_official_ground_truth'] is True,'Official validation check failed')
    require(sha256_file(official/'official_validation_manifest.csv')==report['manifest_sha256'],'Manifest hash mismatch')
    require(sha256_file(official/'class_to_idx.json')==report['class_to_idx_sha256'],'Official class-map hash mismatch')
    require(json.loads((folder/'class_to_idx.json').read_text())==json.loads((official/'class_to_idx.json').read_text()),'Class-map mismatch')
    with (official/'official_validation_manifest.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    labels={r['image_id']:int(r['expected_imagefolder_label']) for r in rows}
    require(len(rows)==len(labels)==50000,'Official image coverage incomplete/duplicated')
    require(summary['checkpoint_load_strict'] is True and summary['input_resolution']==224,'Evaluation protocol mismatch')
    require(sha256_file(folder/'predictions.csv')==summary['predictions_sha256'],'Prediction hash mismatch')
    counts=prediction_counts(folder/'predictions.csv',labels)
    require(counts==check_metrics(summary,export=True),'Recomputed counts disagree with evaluation')
    if a.checkpoint: require(sha256_file(a.checkpoint)==summary['checkpoint_sha256'],'Checkpoint hash mismatch')
    out={'passes_consistency_checks':True,'correct_top1':counts[0],'correct_top5':counts[1],'checkpoint_bytes_checked':bool(a.checkpoint),'scope':'Consistency checks, not independent authentication of training'}
    (folder/'evaluation_consistency_check.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
