"""Parameter and FLOP accounting per compression unit on a fixed frame.
FLOPs = 2 x multiply-accumulates of every Conv2d / ConvTranspose2d / Linear layer, measured with forward hooks
on one fixed frame of the tuning split (LiDAR point count varies per frame; image size is fixed)."""
import os
import torch
import torch.nn as nn
from .data import KittiFusionDataset, collate
from .evaluation import to_device
from .model import component_of


def profile_batch(root, use_image=True, split='tune', index=0):
    ds = KittiFusionDataset(root, os.path.join(root, 'ImageSets', split + '.txt'), training=False, use_image=use_image)
    return collate([ds[index]])


def _macs(m, inp, out):
    if isinstance(m, nn.Conv2d):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        return out.numel() * k
    if isinstance(m, nn.ConvTranspose2d):
        x = inp[0]
        return x.numel() * m.out_channels * m.kernel_size[0] * m.kernel_size[1] // m.groups
    if isinstance(m, nn.Linear):
        return out.numel() * m.in_features
    return 0


@torch.no_grad()
def profile_model(model, root, device='cuda'):
    model = model.to(device).eval()
    batch = to_device(profile_batch(root, model.use_image), device)
    flops, hooks = {}, []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            u = component_of(name + '.')
            hooks.append(mod.register_forward_hook(
                lambda m, i, o, u=u: flops.__setitem__(u, flops.get(u, 0) + 2 * _macs(m, i, o))))
    model(batch)
    for h in hooks:
        h.remove()
    params = {}
    for n, p in model.named_parameters():
        u = component_of(n); params[u] = params.get(u, 0) + p.numel()
    out = {u: {'params_M': params.get(u, 0) / 1e6, 'GFLOPs': flops.get(u, 0) / 1e9}
           for u in ['image', 'fusion', 'lidar', 'head']}
    out['total'] = {'params_M': sum(v['params_M'] for v in out.values()),
                    'GFLOPs': sum(v['GFLOPs'] for v in out.values())}
    out['n_points'] = int(batch['points'].shape[0])
    model.cpu()
    return out
