#!/usr/bin/env python3
"""Optional runtime check of block-local gradients; no ImageNet data or scores."""
import argparse
import json
import torch
import torch.nn.functional as F
from vit_models import LFNNLocalViT, unpack_local_logits

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args()
    torch.manual_seed(1);model=LFNNLocalViT().to(a.device).train()
    # Torchvision initializes its final head to zero; use a nonzero head solely
    # for this topology test so gradients into the last block are observable.
    torch.nn.init.normal_(model.vit.heads.head.weight,std=.01)
    x=torch.randn(1,3,224,224,device=a.device);target=torch.tensor([7],device=a.device)
    results=[]
    for head_index in range(4):
        model.zero_grad(set_to_none=True)
        logits=unpack_local_logits(model(x))[head_index]
        F.cross_entropy(logits,target).backward()
        observed=[]
        for block in range(4):
            params=[p for layer in list(model.vit.encoder.layers.children())[3*block:3*(block+1)] for p in layer.parameters()]
            active=any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in params)
            if active:observed.append(block)
        if observed!=[head_index]:raise RuntimeError(f'Loss {head_index} reaches blocks {observed}')
        token_params=[model.vit.class_token,model.vit.encoder.pos_embedding,*model.vit.conv_proj.parameters()]
        if (any(p.grad is not None for p in token_params)) != (head_index==0):raise RuntimeError('Tokenization gradient boundary failed')
        results.append({'loss_index':head_index,'blocks_with_nonzero_gradient':observed})
    print(json.dumps({'gradient_isolation_passed':True,'checks':results},indent=2))
if __name__=='__main__':main()
