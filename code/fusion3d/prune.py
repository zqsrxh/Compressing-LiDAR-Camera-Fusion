"""Structured (channel) pruning per compression unit.

Each prunable group is a producer layer whose output channels are removed, together with its
BatchNorm and the matching input channels of every consumer. Channels at the boundary between
units are kept fixed, so each unit can be compressed alone:
  image : internal widths of every residual block in layer2-layer4 (stem and layer1 are frozen and kept),
          the shared FPN inner width and each FPN output width (consumers: the fusion lateral convs)
  fusion: the five lateral 3x3 convs (consumer: image-feature linear projection), and the fused
          point-feature width (img/pts projections jointly; consumer: first BEV conv)
  lidar : PFN layer 1 (consumer: PFN layer 2, both halves), every conv of the 2D BEV backbone
          including block outputs, and the up-sampling (deblock) outputs (consumer: head 1x1 convs)
  head  : not pruned (its outputs are the class/box/direction maps)
Importance criteria: 'l1' (filter L1 norm, Li et al. 2017), 'fpgm' (distance to the other filters,
geometric-median criterion, He et al. 2019), 'random'. Kept widths are rounded to multiples of 8.
"""
import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, BasicBlock


def importance(weight, criterion, out_dim=0, seed=0):
    W = weight.detach().float()
    if out_dim != 0:
        W = W.transpose(0, out_dim)
    W = W.reshape(W.shape[0], -1)
    if criterion == 'l1':
        return W.abs().sum(1)
    if criterion == 'fpgm':
        return torch.cdist(W[None], W[None])[0].sum(1)
    if criterion == 'random':
        g = torch.Generator().manual_seed(seed)
        return torch.rand(W.shape[0], generator=g)
    raise ValueError(criterion)


def n_keep(c, ratio, mult=8, min_keep=8):
    k = int(round(c * (1.0 - ratio) / mult)) * mult
    return int(min(c, max(min_keep, k)))


def _sel(t, idx, dim):
    return torch.index_select(t, dim, idx.to(t.device)).clone()


def _setp(layer, name, data):
    old = getattr(layer, name)
    setattr(layer, name, nn.Parameter(data, requires_grad=old.requires_grad))


def prune_out(layer, idx):
    if isinstance(layer, nn.ConvTranspose2d):
        _setp(layer, 'weight', _sel(layer.weight.data, idx, 1)); layer.out_channels = len(idx)
    elif isinstance(layer, nn.Conv2d):
        _setp(layer, 'weight', _sel(layer.weight.data, idx, 0)); layer.out_channels = len(idx)
    elif isinstance(layer, nn.Linear):
        _setp(layer, 'weight', _sel(layer.weight.data, idx, 0)); layer.out_features = len(idx)
    else:
        raise TypeError(type(layer))
    if layer.bias is not None:
        _setp(layer, 'bias', _sel(layer.bias.data, idx, 0))


def prune_in(layer, idx):
    if isinstance(layer, nn.ConvTranspose2d):
        _setp(layer, 'weight', _sel(layer.weight.data, idx, 0)); layer.in_channels = len(idx)
    elif isinstance(layer, nn.Conv2d):
        _setp(layer, 'weight', _sel(layer.weight.data, idx, 1)); layer.in_channels = len(idx)
    elif isinstance(layer, nn.Linear):
        _setp(layer, 'weight', _sel(layer.weight.data, idx, 1)); layer.in_features = len(idx)
    else:
        raise TypeError(type(layer))


def prune_bn(bn, idx):
    if isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
        _setp(bn, 'weight', _sel(bn.weight.data, idx, 0)); _setp(bn, 'bias', _sel(bn.bias.data, idx, 0))
        bn.running_mean = _sel(bn.running_mean, idx, 0); bn.running_var = _sel(bn.running_var, idx, 0)
        bn.num_features = len(idx)
    else:  # torchvision FrozenBatchNorm2d (buffers only)
        for name in ['weight', 'bias', 'running_mean', 'running_var']:
            setattr(bn, name, _sel(getattr(bn, name), idx, 0))


def _width(l):
    if isinstance(l, (nn.Conv2d, nn.ConvTranspose2d)):
        return l.out_channels
    return l.out_features


def _ident(i):
    return i


class Group:
    def __init__(self, name, unit, producers, bns, consumers):
        self.name, self.unit = name, unit
        self.producers, self.bns, self.consumers = producers, bns, consumers

    @property
    def channels(self):
        return _width(self.producers[0])

    def scores(self, criterion, seed=0):
        s = 0
        for l in self.producers:
            s = s + importance(l.weight, criterion, 1 if isinstance(l, nn.ConvTranspose2d) else 0, seed)
        return s

    def apply(self, keep):
        keep = torch.sort(keep)[0]
        for p in self.producers:
            prune_out(p, keep)
        for b in self.bns:
            prune_bn(b, keep)
        for layer, fn in self.consumers:
            prune_in(layer, fn(keep))


def _slot(widths, k, keep):
    off = sum(widths[:k])
    return torch.cat([torch.arange(off), keep + off, torch.arange(off + widths[k], sum(widths))]).long()


def build_groups(model):
    G = []
    if getattr(model, 'use_image', False):
        img = model.image
        for ln in ['layer2', 'layer3', 'layer4']:
            for bi, blk in enumerate(getattr(img, ln)):
                if isinstance(blk, Bottleneck):
                    G.append(Group(f'image.{ln}.{bi}.conv1', 'image', [blk.conv1], [blk.bn1], [(blk.conv2, _ident)]))
                    G.append(Group(f'image.{ln}.{bi}.conv2', 'image', [blk.conv2], [blk.bn2], [(blk.conv3, _ident)]))
                elif isinstance(blk, BasicBlock):
                    G.append(Group(f'image.{ln}.{bi}.conv1', 'image', [blk.conv1], [blk.bn1], [(blk.conv2, _ident)]))
        fu = model.fusion
        fpn = img.fpn
        inner = [b[0] for b in fpn.inner_blocks]
        layer = [b[0] for b in fpn.layer_blocks]
        # FPN inner width is shared by all levels (top-down additions): one coupled group
        G.append(Group('image.fpn.inner', 'image', inner, [], [(l, _ident) for l in layer]))
        # FPN output width per level; P6 = maxpool(P5) feeds the fifth lateral conv
        for k, l in enumerate(layer):
            cons = [(fu.lateral_convs[k], _ident)]
            if k == len(layer) - 1 and len(fu.lateral_convs) > len(layer):
                cons.append((fu.lateral_convs[k + 1], _ident))
            G.append(Group(f'image.fpn.out{k}', 'image', [l], [], cons))
        widths = [c.out_channels for c in fu.lateral_convs]
        for k, conv in enumerate(fu.lateral_convs):
            G.append(Group(f'fusion.lateral.{k}', 'fusion', [conv], [],
                           [(fu.img_transform[0], (lambda kk: (lambda keep: _slot(widths, kk, keep)))(k))]))
        first_bev = model.backbone.blocks[0][1]
        G.append(Group('fusion.out', 'fusion', [fu.img_transform[0], fu.pts_transform[0]],
                       [fu.img_transform[1], fu.pts_transform[1]], [(first_bev, _ident)]))
    pf = model.pillar
    c1 = pf.pfn1.linear.out_features
    G.append(Group('lidar.pfn1', 'lidar', [pf.pfn1.linear], [pf.pfn1.norm],
                   [(pf.pfn2.linear, lambda keep: torch.cat([keep, keep + c1]))]))
    bb, head = model.backbone, model.head
    nblk = len(bb.blocks)
    dwidths = [d[0].out_channels for d in bb.deblocks]
    for bi, blk in enumerate(bb.blocks):
        convs = [m for m in blk if isinstance(m, nn.Conv2d)]
        bns = [m for m in blk if isinstance(m, nn.BatchNorm2d)]
        for ci, (cv, bn) in enumerate(zip(convs, bns)):
            if ci < len(convs) - 1:
                cons = [(convs[ci + 1], _ident)]
            else:
                cons = [(bb.deblocks[bi][0], _ident)]
                if bi + 1 < nblk:
                    cons.append(([m for m in bb.blocks[bi + 1] if isinstance(m, nn.Conv2d)][0], _ident))
            G.append(Group(f'lidar.block{bi}.conv{ci}', 'lidar', [cv], [bn], cons))
    for di, db in enumerate(bb.deblocks):
        fn = (lambda dd: (lambda keep: _slot(dwidths, dd, keep)))(di)
        G.append(Group(f'lidar.deblock{di}', 'lidar', [db[0]], [db[1]],
                       [(head.conv_cls, fn), (head.conv_box, fn), (head.conv_dir_cls, fn)]))
    return G


def plan_uniform(model, ratios, criterion, seed=0):
    """ratios: dict unit -> ratio. Returns dict group name -> kept channel indices."""
    plan = {}
    for g in build_groups(model):
        r = ratios.get(g.unit, 0.0)
        if r <= 0:
            continue
        k = n_keep(g.channels, r)
        if k < g.channels:
            plan[g.name] = torch.topk(g.scores(criterion, seed), k).indices
    return plan


def plan_global(model, target_ratio, criterion='l1', units=('image', 'fusion', 'lidar'), max_ratio=0.9):
    """Global ranking of layer-normalised importance (score / layer mean) over all prunable groups."""
    gs = [g for g in build_groups(model) if g.unit in units]
    allv = []
    for gi, g in enumerate(gs):
        s = g.scores(criterion)
        s = s / s.mean().clamp(min=1e-12)
        allv += [(float(v), gi) for v in s]
    allv.sort()
    cnt = {}
    for v, gi in allv[:int(target_ratio * len(allv))]:
        cnt[gi] = cnt.get(gi, 0) + 1
    plan = {}
    for gi, g in enumerate(gs):
        if gi not in cnt:
            continue
        k = n_keep(g.channels, min(max_ratio, cnt[gi] / g.channels))
        if k < g.channels:
            plan[g.name] = torch.topk(g.scores(criterion), k).indices
    return plan


def apply_plan(model, plan):
    """Apply a plan; groups are rebuilt after each step so slot offsets reflect current widths."""
    for name, keep in plan.items():
        g = {x.name: x for x in build_groups(model)}[name]
        g.apply(keep.cpu())
    return model
