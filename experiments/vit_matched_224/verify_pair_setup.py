#!/usr/bin/env python3
import argparse, json, torch
from common import hash_jsonable, seed_everything, sha256_file
from vit_models import BPViT, LFNNLocalViT, extract_final_logits, shared_manifest, method_specific_manifest

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--shared-init',required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    seed_everything(a.seed); state=torch.load(a.shared_init,map_location='cpu',weights_only=True)['shared_vit_state']; bp=BPViT(state).eval(); lf=LFNNLocalViT(state).eval()
    bm,lm=shared_manifest(bp),shared_manifest(lf)
    if bm!=lm: raise RuntimeError('shared parameter manifests differ')
    for k,v in bp.vit.state_dict().items():
        if not torch.equal(v,lf.vit.state_dict()[k]): raise RuntimeError(f'shared tensor differs: {k}')
    g=torch.Generator().manual_seed(a.seed+999); x=torch.randn(2,3,224,224,generator=g)
    with torch.no_grad(): b=extract_final_logits(bp(x)); l=extract_final_logits(lf(x))
    md=float((b-l).abs().max().item())
    if md>1e-6: raise RuntimeError(f'initial final logits differ by {md}')
    rep={'seed':a.seed,'shared_init_sha256':sha256_file(a.shared_init),'shared_tensors_equal':True,'shared_manifest_sha256':hash_jsonable(bm),'shared_parameter_count':sum(x['numel'] for x in bm),'bp_method_specific':method_specific_manifest(bp),'lfnn_method_specific':method_specific_manifest(lf),'max_initial_final_logit_abs_diff':md,'initial_final_prediction_functions_match':True}
    open(a.output,'w').write(json.dumps(rep,indent=2)); print(json.dumps(rep,indent=2))
if __name__=='__main__': main()
