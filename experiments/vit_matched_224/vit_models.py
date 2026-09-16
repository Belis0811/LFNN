import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def build_empty_vit224(): return vit_b_16(weights=None, image_size=224, num_classes=1000)

def prepare_swag_224_state(progress=True):
    model = build_empty_vit224(); target = model.state_dict()
    source = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.get_state_dict(progress=progress, check_hash=True)
    compatible, skipped = {}, {}
    for k,v in source.items():
        if k in target and tuple(v.shape)==tuple(target[k].shape): compatible[k]=v
        else: skipped[k]={'source_shape':list(v.shape), 'target_shape':list(target[k].shape) if k in target else None}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if unexpected: raise RuntimeError(f'unexpected SWAG keys: {unexpected}')
    allowed={'encoder.pos_embedding'}
    if set(missing) != allowed or set(skipped) != allowed:
        raise RuntimeError(f'expected only positional embedding mismatch; missing={missing}, skipped={list(skipped)}')
    return model.state_dict(), {'weights_enum':'ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1','skipped_keys':skipped,'missing_after_load':list(missing)}

class BPViT(nn.Module):
    def __init__(self, shared_state=None):
        super().__init__(); self.vit=build_empty_vit224()
        if shared_state is not None: self.vit.load_state_dict(shared_state, strict=True)
    def forward(self,x): return self.vit(x)

class AuxHead(nn.Module):
    def __init__(self, dim=768, classes=1000):
        super().__init__(); self.norm=nn.LayerNorm(dim); self.fc=nn.Linear(dim, classes)
    def forward(self,x): return self.fc(self.norm(x[:,0]))

class LFNNLocalViT(nn.Module):
    def __init__(self, shared_state=None):
        super().__init__(); self.vit=build_empty_vit224()
        if shared_state is not None: self.vit.load_state_dict(shared_state, strict=True)
        self.aux_heads=nn.ModuleList([AuxHead(self.vit.hidden_dim,1000) for _ in range(3)])
    def _tokenize(self,x):
        n=x.shape[0]; x=self.vit._process_input(x); cls=self.vit.class_token.expand(n,-1,-1)
        x=torch.cat([cls,x],dim=1); x=x+self.vit.encoder.pos_embedding; return self.vit.encoder.dropout(x)
    def forward(self,x):
        x=self._tokenize(x); layers=list(self.vit.encoder.layers.children())
        if len(layers)!=12: raise RuntimeError(f'expected 12 transformer layers, got {len(layers)}')
        aux=[]
        for b in range(4):
            for layer in layers[b*3:(b+1)*3]: x=layer(x)
            if b<3:
                aux.append(self.aux_heads[b](x)); x=x.detach()
        x=self.vit.encoder.ln(x); final=self.vit.heads(x[:,0])
        return final, aux[0], aux[1], aux[2]

def extract_final_logits(output):
    if torch.is_tensor(output): return output
    if isinstance(output,(tuple,list)): return output[0]
    if isinstance(output,dict):
        if 'final_logits' in output: return output['final_logits']
        if 'logits' in output: return output['logits']
    raise TypeError(type(output))

def unpack_local_logits(output):
    if not isinstance(output,(tuple,list)) or len(output)!=4: raise TypeError('expected (final, aux1, aux2, aux3)')
    final,a1,a2,a3=output; return [a1,a2,a3,final]

def shared_manifest(model):
    return [{'name':n,'shape':list(p.shape),'numel':p.numel(),'requires_grad':bool(p.requires_grad)} for n,p in model.vit.named_parameters()]

def method_specific_manifest(model):
    shared_ids={id(p) for p in model.vit.parameters()}
    return [{'name':n,'shape':list(p.shape),'numel':p.numel(),'requires_grad':bool(p.requires_grad)} for n,p in model.named_parameters() if id(p) not in shared_ids]
