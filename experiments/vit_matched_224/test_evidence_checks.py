"""Small artificial unit-test fixtures only; no experiment results are generated."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from evidence_checks import check_metrics, check_export, prediction_counts, check_initial_predictions, METRIC_FIELDS
from validate_evidence_bundle import validate

class Checks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def csv(self,rows,name='test_fixture.csv'):
        p=self.root/name
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader();w.writerows(rows)
        return p
    def predictions(self):
        rows=[]
        for i,(y,top) in enumerate([(0,[0,1,2,3,4]),(7,[6,7,8,9,10]),(20,[25,26,27,28,29])]):
            rows.append({'image_id':f'test_only_{i}','label':y,'top1_correct':int(top[0]==y),'top5_correct':int(y in top),**{f'top5_class_{k}':v for k,v in enumerate(top)}})
        return rows
    def metric(self):
        return dict(event='epoch_end',epoch=0,train_loss=1.0,train_samples=12,lr=.001,elapsed_sec=2.0,val_loss=1.5,correct_top1=1,correct_top5=2,evaluated_samples=3,val_top1=100/3,val_top5=200/3)
    def test_valid_counts(self): self.assertEqual(check_metrics(self.metric(),n=3),(1,2))
    def test_rejects_inconsistent_top5(self):
        r=self.metric();r['val_top5']=99
        with self.assertRaises(ValueError):check_metrics(r,n=3)
    def test_rejects_nonfinite(self):
        r=self.metric();r['val_loss']=float('nan')
        with self.assertRaises(ValueError):check_metrics(r,n=3)
    def test_export_every_field_matches(self):
        r=self.metric();e={k:str(v) for k,v in r.items()};check_export([r],[e])
        for key in METRIC_FIELDS:
            bad=e.copy();bad[key]='incorrect' if key=='event' else str(float(e[key])+1)
            with self.subTest(field=key),self.assertRaises(ValueError):check_export([r],[bad])
    def test_export_initial_empty_training_fields(self):
        r=self.metric()
        for key in ['train_loss','train_samples','lr','elapsed_sec']: del r[key]
        e={key:str(r[key]) if key in r else '' for key in METRIC_FIELDS};check_export([r],[e])
    def test_prediction_counts(self):
        rows=self.predictions();self.assertEqual(prediction_counts(self.csv(rows),{r['image_id']:r['label'] for r in rows},3),(1,2))
    def test_official_label_mismatch(self):
        rows=self.predictions();labels={r['image_id']:r['label'] for r in rows};labels['test_only_0']=4
        with self.assertRaises(ValueError):prediction_counts(self.csv(rows),labels,3)
    def test_duplicate_image(self):
        rows=self.predictions();rows[1]['image_id']=rows[0]['image_id']
        with self.assertRaises(ValueError):prediction_counts(self.csv(rows),n=3)
    def test_wrong_flags(self):
        rows=self.predictions();rows[1]['top1_correct']=1
        with self.assertRaises(ValueError):prediction_counts(self.csv(rows),n=3)
    def test_initial_top5_disagreement(self):
        rows=[dict(image_id='test_only',label=0,bp_top1=0,lfnn_top1=0,bp_top5='0 1 2 3 4',lfnn_top5='0 1 2 3 4',top1_predictions_equal=1,top5_sets_equal=1)]
        rep={f'{m}_correct_top{k}':1 for m in ['bp','lfnn'] for k in [1,5]}
        check_initial_predictions(self.csv(rows),{'test_only':0},rep)
        rows[0]['lfnn_top5']='0 1 2 3 5'
        with self.assertRaises(ValueError):check_initial_predictions(self.csv(rows),{'test_only':0},rep)
    def test_missing_bundle_fails_with_report(self):
        r=validate(self.root,[101]);self.assertFalse(r['passes_structural_validation']);self.assertGreater(r['problem_count'],0)
    def test_pandas_recomputation_cli(self):
        path=self.csv(self.predictions())
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('recompute_metrics.py')),str(path)],capture_output=True,text=True,check=True)
        r=json.loads(result.stdout);self.assertEqual((r['correct_top1'],r['correct_top5']),(1,2))
    def test_paired_summary_final_not_best(self):
        for method in ['bp','lfnn']:
            folder=self.root/'seed_101'/method;folder.mkdir(parents=True)
            pre={**self.metric(),'event':'pre_training_evaluation','epoch':-1}
            early={**self.metric(),'val_top1':90.0,'epoch':0};last={**self.metric(),'val_top1':80.0,'epoch':1}
            (folder/'metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in [pre,early,last])+'\n')
        subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_paired_runs.py')),'--root',str(self.root),'--seeds','101'],capture_output=True,text=True,check=True)
        with (self.root/'paired_summary.csv').open() as f: rows=list(csv.DictReader(f))
        self.assertTrue(all(float(r['final_top1'])==80 and int(r['best_epoch'])==0 for r in rows))

if __name__=='__main__': unittest.main(verbosity=2)
