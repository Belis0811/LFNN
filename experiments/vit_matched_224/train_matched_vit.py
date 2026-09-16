#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import torch, torch.distributed as dist, torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from common import *
from data_utils import *
from vit_models import *

def train_loader(ds,seed,epoch,rank,world,global_bs,workers):
    if global_bs%world: raise ValueError('global batch must be divisible by world size')
    sampler=DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True,seed=seed,drop_last=False); sampler.set_epoch(epoch)
    g=torch.Generator().manual_seed(seed*1_000_000+epoch*1_000+rank)
    return DataLoader(ds,batch_size=global_bs//world,sampler=sampler,num_workers=workers,pin_memory=True,drop_last=True,worker_init_fn=seed_worker,generator=g,persistent_workers=False)

def val_loader(ds,rank,world,global_bs,workers):
    return DataLoader(ds,batch_size=max(1,global_bs//world),sampler=DistributedEvalSampler(ds,rank,world),num_workers=workers,pin_memory=True,drop_last=False,persistent_workers=False)

@torch.no_grad()
def evaluate(model,loader,device):
    # Bypass DDP forward collectives: ranks can have unequal validation batches.
    model=model.module if hasattr(model,'module') else model
    model.eval(); c1=c5=n=0; loss_sum=0.0
    for images,targets,_ in loader:
        images=images.to(device,non_blocking=True); targets=targets.to(device,non_blocking=True); logits=extract_final_logits(model(images)); loss_sum+=float(F.cross_entropy(logits,targets,reduction='sum'))
        top5=logits.topk(5,1).indices; c1+=int(top5[:,0].eq(targets).sum()); c5+=int(top5.eq(targets[:,None]).any(1).sum()); n+=targets.numel()
    c1,c5,n=all_reduce_long([c1,c5,n],device); loss_sum=all_reduce_float(loss_sum,device)
    return {'correct_top1':c1,'correct_top5':c5,'evaluated_samples':n,'val_top1':100*c1/n,'val_top5':100*c5/n,'val_loss':loss_sum/n}

def train_epoch(model,loader,opt,method,device,scaler,amp):
    model.train(); loss_sum=0.0; n=0
    for images,targets,_ in loader:
        images=images.to(device,non_blocking=True); targets=targets.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            output=model(images)
            if method=='bp': loss=F.cross_entropy(extract_final_logits(output),targets)
            else: loss=torch.stack([F.cross_entropy(z,targets) for z in unpack_local_logits(output)]).mean()
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); b=targets.numel(); loss_sum+=float(loss.detach())*b; n+=b
    loss_sum = all_reduce_float(loss_sum, device)
    n = all_reduce_long([n], device)[0]
    return loss_sum / n, n

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--method',choices=['bp','lfnn'],required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--shared-init',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--epochs',type=int,default=90); ap.add_argument('--batch-size',type=int,default=256); ap.add_argument('--lr',type=float,default=.001); ap.add_argument('--momentum',type=float,default=.9); ap.add_argument('--weight-decay',type=float,default=.0005); ap.add_argument('--eta-min',type=float,default=0.0); ap.add_argument('--workers',type=int,default=8); ap.add_argument('--amp',action='store_true'); a=ap.parse_args()
    if a.workers < 1: raise ValueError('Use workers >= 1 to preserve paired augmentation streams independently of model RNG')
    if a.epochs < 1: raise ValueError('epochs must be positive')
    rank,local,world=init_distributed(); seed_everything(a.seed); device=torch.device(f'cuda:{local}'); state=torch.load(a.shared_init,map_location='cpu',weights_only=True)['shared_vit_state']; raw=BPViT(state) if a.method=='bp' else LFNNLocalViT(state); raw.to(device)
    sm=shared_manifest(raw); mm=method_specific_manifest(raw); model=DDP(raw,device_ids=[local],output_device=local,find_unused_parameters=False) if world>1 else raw
    opt=torch.optim.SGD(model.parameters(),lr=a.lr,momentum=a.momentum,weight_decay=a.weight_decay); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=a.eta_min); scaler=torch.cuda.amp.GradScaler(enabled=a.amp)
    tr=make_train_dataset(a.data_root); va=make_val_dataset(a.data_root); counts=verify_standard_val_distribution(va);
    if tr.class_to_idx != va.class_to_idx: raise RuntimeError('train/val class_to_idx differ')
    vlo=val_loader(va,rank,world,a.batch_size,a.workers); out=Path(a.output_dir)
    if (out/'metrics.jsonl').exists() or (out/'last.pt').exists(): raise FileExistsError('Refusing to overwrite an existing run')
    if rank==0: out.mkdir(parents=True,exist_ok=True)
    barrier()
    cfg={'protocol_scope':'NEW_MATCHED_CONTROLLED_EXPERIMENT','workers_per_rank':a.workers,'amp':a.amp,'preprocessing':preprocessing_config(),'class_to_idx_sha256':hash_jsonable(va.class_to_idx),'train_drop_last':True,'sampler_drop_last':False,'loss_reduction':'mean over minibatch, then mean over four local heads' if a.method=='lfnn' else 'mean over minibatch','local_loss_weights':[0.25]*4 if a.method=='lfnn' else [1.0],'leadership_fraction':1.0 if a.method=='lfnn' else None,'worker_partition':[3,3,3,3] if a.method=='lfnn' else None,'follower_alignment_active':False,'source_file_sha256':{f.name:sha256_file(f) for f in sorted(Path(__file__).resolve().parent.glob('*.py'))},'not_reproduction_of_previous_recipe':True,'method':a.method,'seed':a.seed,'input_resolution':224,'epochs':a.epochs,'global_batch_size':a.batch_size,'optimizer':'SGD','lr':a.lr,'momentum':a.momentum,'weight_decay':a.weight_decay,'scheduler':'CosineAnnealingLR','T_max':a.epochs,'eta_min':a.eta_min,'world_size':world,'train_images':len(tr),'validation_images':len(va),'validation_class_count_min':min(counts.values()),'validation_class_count_max':max(counts.values()),'shared_vit_manifest_sha256':hash_jsonable(sm),'shared_vit_parameter_count':sum(x['numel'] for x in sm),'method_specific_parameter_count':sum(x['numel'] for x in mm),'shared_init_sha256':sha256_file(a.shared_init),'primary_endpoint':'final epoch validation accuracy','initialization':'shape-compatible SWAG E2E V1; newly initialized 224 positional embedding shared between methods','best_epoch_reported_separately':True,'released_output_ordering':'LFNN: (final_logits, auxiliary_1, auxiliary_2, auxiliary_3)'}
    if rank==0:
        (out/'resolved_config.json').write_text(json.dumps(cfg,indent=2)); (out/'parameter_manifest.json').write_text(json.dumps({'shared_vit':sm,'method_specific':mm},indent=2))
    pre=evaluate(model,vlo,device)
    if pre['evaluated_samples']!=50000: raise RuntimeError(pre['evaluated_samples'])
    if rank==0: (out/'metrics.jsonl').write_text(json.dumps({'event':'pre_training_evaluation','epoch':-1,**pre})+'\n')
    barrier(); best=-1.0; best_epoch=None
    for epoch in range(a.epochs):
        t0=time.time(); tl,tn=train_epoch(model,train_loader(tr,a.seed,epoch,rank,world,a.batch_size,a.workers),opt,a.method,device,scaler,a.amp); val=evaluate(model,vlo,device); elapsed=time.time()-t0
        if val['evaluated_samples']!=50000: raise RuntimeError(val['evaluated_samples'])
        if rank==0:
            rec={'event':'epoch_end','epoch':epoch,'train_loss':tl,'train_samples':tn,'lr':opt.param_groups[0]['lr'],'elapsed_sec':elapsed,**val}
            with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(rec)+'\n')
            print(json.dumps(rec),flush=True)
            m=model.module if hasattr(model,'module') else model; torch.save({'epoch':epoch,'method':a.method,'seed':a.seed,'model':m.state_dict(),'resolved_config':cfg},out/'last.pt')
            if val['val_top1']>best:
                best=val['val_top1']; best_epoch=epoch; torch.save({'epoch':epoch,'method':a.method,'seed':a.seed,'model':m.state_dict(),'resolved_config':cfg,'metrics':val},out/'best.pt')
        sched.step()
    barrier()
    if rank==0: (out/'run_summary.json').write_text(json.dumps({'method':a.method,'seed':a.seed,'primary_final_epoch':a.epochs-1,'best_epoch':best_epoch,'best_top1':best,'last_checkpoint_sha256':sha256_file(out/'last.pt'),'best_checkpoint_sha256':sha256_file(out/'best.pt')},indent=2))
    cleanup_distributed()
if __name__=='__main__': main()
