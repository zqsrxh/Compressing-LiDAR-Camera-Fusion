"""One-shot pruning sensitivity per unit (Li et al. 2017 style: prune one unit, no fine-tuning), with
BatchNorm statistics re-estimated on train_sub frames (no gradient updates). Evaluated on the tuning split.
"""
import argparse, copy, json, os
import numpy as np
import torch
import torch.nn as nn
from fusion3d.data import KittiFusionDataset, make_loader
from fusion3d.evaluation import run_inference, evaluate_annos, summarize, to_device
from fusion3d.prune import plan_uniform, apply_plan
from fusion3d.profile import profile_model


@torch.no_grad()
def recalibrate_bn(model, loader, n_batches):
    bns = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    for m in bns:
        m.reset_running_stats(); m.momentum = None
    model.train()
    for k, b in enumerate(loader):
        if k >= n_batches:
            break
        model.forward_features(to_device(b, 'cuda'))
    for m in bns:
        m.momentum = 0.01
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--units', default='image,fusion,lidar')
    ap.add_argument('--ratios', default='0.25,0.5,0.7')
    ap.add_argument('--criteria', default='l1,fpgm')
    ap.add_argument('--bn-batches', type=int, default=25)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--split', default='tune')
    a = ap.parse_args()
    base = torch.load(a.model, map_location='cpu', weights_only=False)
    S = os.path.join(a.root, 'ImageSets')
    tune = KittiFusionDataset(a.root, os.path.join(S, a.split + '.txt'), training=False, use_image=base.use_image)
    cal = KittiFusionDataset(a.root, os.path.join(S, 'train_sub.txt'), training=False, use_image=base.use_image)
    tl = make_loader(tune, 4, False, a.workers)
    cl = make_loader(cal, 8, True, a.workers, seed=0)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    def run(name, m):
        if name in res:
            return
        m = m.cuda()
        annos = run_inference(m, tl, 'cuda')
        _, d = evaluate_annos(annos, a.root, tune.ids)
        r = summarize(d); r['profile'] = profile_model(m, a.root)
        res[name] = r
        json.dump(res, open(a.out, 'w'), indent=1)
        print(name, round(r['mAP3d_mod_R40'], 2), round(r['profile']['total']['GFLOPs'], 1), flush=True)
    run('base', copy.deepcopy(base))
    m = copy.deepcopy(base).cuda(); recalibrate_bn(m, cl, a.bn_batches); run('base_bnrecal', m)
    for crit in a.criteria.split(','):
        for u in a.units.split(','):
            if u in ('image', 'fusion') and not base.use_image:
                continue
            for r in [float(x) for x in a.ratios.split(',')]:
                name = f'{u}_{crit}_{r}'
                if name in res:
                    continue
                m = copy.deepcopy(base)
                apply_plan(m, plan_uniform(m, {u: r}, crit))
                m = m.cuda(); recalibrate_bn(m, cl, a.bn_batches)
                run(name, m)


if __name__ == '__main__':
    main()
