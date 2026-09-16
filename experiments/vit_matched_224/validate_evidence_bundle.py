#!/usr/bin/env python3
"""Check measured-record consistency; never fill, correct, or generate result rows."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from evidence_checks import *

def read_json(path): return json.loads(Path(path).read_text())
def csv_rows(path):
    with open(path,newline='') as f: return list(csv.DictReader(f))
def read_metrics(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def validate(root,seeds,require_checkpoints=False):
    root=Path(root); problems=[]; warnings=[]
    def check(name,fn):
        try: return fn()
        except (ValueError,KeyError,IndexError,TypeError,OSError,ZeroDivisionError) as e:
            problems.append(f'{name}: {e}'); return None
    def provenance():
        prov=read_json(root/'provenance/provenance.json')
        hashes=prov['source_file_sha256']; require(bool(hashes),'Empty source manifest')
        for name,digest in hashes.items():
            require(Path(name).name==name,'Invalid source snapshot name')
            require(sha256_file(root/'provenance/source_snapshot'/name)==digest,f'Source hash mismatch: {name}')
        return hashes
    hashes=check('provenance',provenance)
    def official():
        folder=root/'validation_manifest'; rep=read_json(folder/'official_validation_check.json')
        require(rep['all_images_match_official_ground_truth'] is True,'Official mapping failed')
        require(rep['validation_images']==rep['matched_images']==rep['unique_image_numbers']==50000,'Official coverage failed')
        require(rep['mismatched_images']==0 and rep['class_directories']==1000,'Official class check failed')
        for file,key in [('official_validation_manifest.csv','manifest_sha256'),('class_to_idx.json','class_to_idx_sha256')]:
            require(sha256_file(folder/file)==rep[key],f'Official artifact hash mismatch: {file}')
        mapping=read_json(folder/'class_to_idx.json'); require(len(mapping)==1000 and set(mapping.values())==set(range(1000)),'Invalid class mapping')
        require([mapping[k] for k in sorted(mapping)]==list(range(1000)),'Class order is not sorted by WNID')
        labels={}; numbers=set()
        for r in csv_rows(folder/'official_validation_manifest.csv'):
            name=r['image_id']; number=integer(r['image_number']); label=integer(r['expected_imagefolder_label'])
            require(name==f'ILSVRC2012_val_{number:08d}.JPEG' and 1<=number<=50000,'Invalid official image number')
            require(name not in labels and number not in numbers,'Duplicate official image'); numbers.add(number)
            require(mapping[r['official_wnid']]==label==integer(r['actual_imagefolder_label']),'Official label mismatch')
            require(r['actual_class_directory']==r['official_wnid'] and r['matches_official_ground_truth']=='True','Official folder mismatch')
            labels[name]=label
        require(numbers==set(range(1,50001)) and Counter(labels.values())==Counter({k:50 for k in range(1000)}),'Official coverage/class counts failed')
        return labels,mapping
    off=check('official validation',official); labels,mapping=off if off else (None,None)
    configs={}
    for seed in seeds:
        def initial():
            folder=root/'initial_comparison'/f'seed_{seed}'; rep=read_json(folder/'initial_logit_comparison.json')
            pair=read_json(root/'shared_init'/f'seed_{seed}_pair_check.json')
            require(pair['seed']==seed and pair['shared_tensors_equal'] is True and pair['initial_final_prediction_functions_match'] is True,'Shared tensor/prediction setup failed')
            require(pair['shared_init_sha256']==rep['shared_init_sha256'],'Pair check/shared-init digest differs')
            init_path=root/'shared_init'/f'seed_{seed}.pt'
            if init_path.exists(): require(sha256_file(init_path)==rep['shared_init_sha256'],'Shared-init file hash mismatch')
            elif require_checkpoints: raise ValueError('Shared-init checkpoint missing')
            require(rep['evaluated_samples']==rep['top1_prediction_agreement_count']==rep['top5_set_agreement_count']==50000,'Initial predictions differ')
            require(rep['bp_correct_top1']==rep['lfnn_correct_top1'] and rep['bp_correct_top5']==rep['lfnn_correct_top5'],'Initial counts differ')
            require(math.isfinite(rep['max_abs_final_logit_difference']) and 0<=rep['max_abs_final_logit_difference']<=1e-6,'Initial logits differ/nonfinite')
            require(rep['same_preprocessed_batch_used_for_both_models'] is True,'Initial input pairing missing')
            if labels is not None: check_initial_predictions(folder/'initial_prediction_comparison.csv',labels,rep)
            return rep
        init=check(f'seed {seed} initialization',initial)
        for method in ['bp','lfnn']:
            def run_checks():
                run=root/f'seed_{seed}'/method
                for name in ['console.log','command.txt']: require((run/name).stat().st_size>0,f'Missing/empty {name}')
                cfg=read_json(run/'resolved_config.json'); configs[(seed,method)]=cfg
                require(cfg['method']==method and cfg['seed']==seed,'Run identity mismatch')
                require(cfg['input_resolution']==224 and cfg['epochs']==90,'Expected 224 / 90-epoch protocol')
                require(cfg['primary_endpoint']=='final epoch validation accuracy','Unexpected primary endpoint')
                require(cfg['follower_alignment_active'] is False,'This trainer implements only the all-leader local control')
                manifest=read_json(run/'parameter_manifest.json'); shared=manifest['shared_vit']; extra=manifest['method_specific']
                require(bool(shared) and hash_jsonable(shared)==cfg['shared_vit_manifest_sha256'],'Shared manifest mismatch')
                for group in [shared,extra]:
                    require(len({v['name'] for v in group})==len(group),'Duplicate parameter name')
                    for v in group: require(math.prod(v['shape'])==v['numel'] and v['requires_grad'] is True,'Parameter shape/count/trainability mismatch')
                require(sum(x['numel'] for x in shared)==cfg['shared_vit_parameter_count'],'Shared parameter count mismatch')
                require(sum(x['numel'] for x in extra)==cfg['method_specific_parameter_count'],'Extra parameter count mismatch')
                if hashes is not None:
                    require(bool(cfg['source_file_sha256']),'Missing executed source hashes')
                    for name,digest in cfg['source_file_sha256'].items(): require(hashes.get(name)==digest,f'Executed source/snapshot mismatch: {name}')
                if mapping is not None: require(hash_jsonable(mapping)==cfg['class_to_idx_sha256'],'Run class mapping mismatch')
                rows=read_metrics(run/'metrics.jsonl'); require(len(rows)==cfg['epochs']+1,'Wrong record count')
                require(rows[0]['event']=='pre_training_evaluation' and rows[0]['epoch']==-1,'Missing initial record')
                eps=rows[1:]; require([r['epoch'] for r in eps]==list(range(cfg['epochs'])),'Missing/repeated/reordered epoch')
                for r in rows: check_metrics(r)
                for r in eps:
                    require(r['event']=='epoch_end' and set(METRIC_FIELDS)<=set(r),'Incomplete epoch schema')
                    require(integer(r['train_samples'])>0,'Invalid training sample count')
                    for key in ['train_loss','lr','elapsed_sec']: require(math.isfinite(float(r[key])) and r[key]>=0,f'Invalid {key}')
                    expected_lr=cfg['eta_min']+(cfg['lr']-cfg['eta_min'])*(1+math.cos(math.pi*r['epoch']/cfg['T_max']))/2
                    require(equal_number(r['lr'],expected_lr,1e-12),'LR differs from resolved cosine schedule')
                check_export(rows,csv_rows(run/'reviewer_metrics.csv'))
                if init is not None:
                    require(cfg['shared_init_sha256']==init['shared_init_sha256'],'Shared initialization hash mismatch')
                    require(rows[0]['correct_top1']==init[f'{method}_correct_top1'] and rows[0]['correct_top5']==init[f'{method}_correct_top5'],'Pre-training metrics disagree with paired initialization check')
                summary=read_json(run/'run_summary.json'); best=max(eps,key=lambda r:r['val_top1']); final=eps[-1]
                require(summary['method']==method and summary['seed']==seed,'Summary identity mismatch')
                require(summary['primary_final_epoch']==final['epoch'] and summary['best_epoch']==best['epoch'] and equal_number(summary['best_top1'],best['val_top1']),'Summary endpoint mismatch')
                evdir=run/'final_checkpoint_re_evaluation'; ev=read_json(evdir/'evaluation_summary.json')
                check_metrics(ev,export=True)
                require(ev['method']==method and ev['checkpoint_load_strict'] is True and ev['input_resolution']==224,'Checkpoint evaluation protocol mismatch')
                require(ev['preprocessing']==cfg['preprocessing']['validation'] and ev['model_factory']=='matched_builtin' and ev['output_selector']=='final','Evaluation recipe/model mismatch')
                require(read_json(evdir/'class_to_idx.json')==mapping,'Evaluation class mapping mismatch')
                require(ev['checkpoint_sha256']==summary['last_checkpoint_sha256'],'Evaluated checkpoint hash differs from final checkpoint hash')
                require(sha256_file(evdir/'predictions.csv')==ev['predictions_sha256'],'Prediction file hash mismatch')
                counts=prediction_counts(evdir/'predictions.csv',labels)
                require(counts==(ev['correct_top1'],ev['correct_top5'])==(final['correct_top1'],final['correct_top5']),'Predictions/evaluation/final log counts disagree')
                # Different batch sizes can alter floating-point loss slightly; counts above remain exact.
                require(equal_number(ev['validation_loss'],final['val_loss'],1e-4),'Re-evaluation loss differs from final epoch')
                for filename,key in [('last.pt','last_checkpoint_sha256'),('best.pt','best_checkpoint_sha256')]:
                    path=run/filename
                    if path.exists(): require(sha256_file(path)==summary[key],f'Checkpoint hash mismatch: {filename}')
                    elif require_checkpoints: raise ValueError(f'Missing local {filename}')
                    else: warnings.append(f'seed {seed}/{method}: {filename} unavailable; checkpoint bytes not checked')
            check(f'seed {seed}/{method}',run_checks)
        def pair():
            bp=configs[(seed,'bp')]; lf=configs[(seed,'lfnn')]
            fields=['seed','input_resolution','epochs','global_batch_size','optimizer','lr','momentum','weight_decay','scheduler','T_max','eta_min','world_size','workers_per_rank','amp','preprocessing','train_drop_last','sampler_drop_last','shared_init_sha256','shared_vit_manifest_sha256','class_to_idx_sha256','source_file_sha256']
            for key in fields: require(bp[key]==lf[key],f'Paired setting differs: {key}')
            require(bp['local_loss_weights']==[1.0] and lf['local_loss_weights']==[0.25]*4,'Unexpected loss scaling')
            require(lf['worker_partition']==[3,3,3,3] and lf['leadership_fraction']==1.0,'Unexpected local worker setting')
        check(f'seed {seed} paired settings',pair)
    return {'root':str(root.resolve()),'problem_count':len(problems),'problems':problems,'warnings':warnings,
            'passes_structural_validation':not problems,'scope':'Artifact consistency only; does not authenticate training or independently evaluate withheld weights.'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--seeds',nargs='+',type=int,default=[101,202,303]); ap.add_argument('--require-checkpoints',action='store_true'); a=ap.parse_args()
    require(len(set(a.seeds))==len(a.seeds),'Duplicate seeds')
    rep=validate(a.root,a.seeds,a.require_checkpoints)
    (Path(a.root)/'evidence_validation_report.json').write_text(json.dumps(rep,indent=2)); print(json.dumps(rep,indent=2))
    if rep['problems']: raise SystemExit(1)
if __name__=='__main__': main()
