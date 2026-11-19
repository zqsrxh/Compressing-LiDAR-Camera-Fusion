"""Prune a trained detector according to a spec and save the pruned (not yet fine-tuned) model.

  --method uniform --criterion l1 --ratios image=0.5,fusion=0,lidar=0
  --method global  --criterion l1 --target 0.4
Also writes params / GFLOPs per unit (FLOPs counted with torch.utils.flop_counter on a fixed tune frame).
"""
import argparse, copy, json, os
import torch
from fusion3d.prune import plan_uniform, plan_global, apply_plan
from fusion3d.profile import profile_model


def parse_ratios(s):
    out = {}
    for kv in s.split(','):
        if kv:
            k, v = kv.split('='); out[k] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--method', default='uniform', choices=['uniform', 'global', 'global_match', 'alloc'])
    ap.add_argument('--match-ratio', type=float, default=0.5, help='global_match: match GFLOPs of uniform pruning at this ratio')
    ap.add_argument('--alloc', default='', help='alloc: allocation.json from alloc.py')
    ap.add_argument('--budget', default='0.5')
    ap.add_argument('--criterion', default='l1', choices=['l1', 'fpgm', 'random'])
    ap.add_argument('--ratios', default='')
    ap.add_argument('--target', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    m = torch.load(a.model, map_location='cpu', weights_only=False)
    before = profile_model(m, a.root)
    extra = {}
    if a.method == 'uniform':
        plan = plan_uniform(m, parse_ratios(a.ratios), a.criterion, a.seed)
    elif a.method == 'alloc':
        ratios = json.load(open(a.alloc))['budgets'][a.budget]['ratios']
        extra['ratios'] = ratios
        plan = plan_uniform(m, ratios, a.criterion, a.seed)
    elif a.method == 'global':
        plan = plan_global(m, a.target, a.criterion)
    else:  # global_match: bisection on the global ratio until GFLOPs match uniform pruning at match_ratio
        units = ['image', 'fusion', 'lidar'] if m.use_image else ['lidar']
        mu = copy.deepcopy(m); apply_plan(mu, plan_uniform(mu, {u: a.match_ratio for u in units}, a.criterion))
        target_g = profile_model(mu, a.root)['total']['GFLOPs']
        lo, hi = 0.02, 0.9
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            mg = copy.deepcopy(m); apply_plan(mg, plan_global(mg, mid, a.criterion))
            g = profile_model(mg, a.root)['total']['GFLOPs']
            if g > target_g:
                lo = mid
            else:
                hi = mid
        plan = plan_global(m, hi, a.criterion)
        extra.update(global_ratio=hi, target_GFLOPs=target_g)
    apply_plan(m, plan)
    after = profile_model(m, a.root)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    torch.save(m, a.out)
    rec = {'args': vars(a), 'extra': extra, 'n_groups_pruned': len(plan),
           'widths': {k: int(len(v)) for k, v in plan.items()}, 'before': before, 'after': after}
    json.dump(rec, open(a.out.replace('.pt', '_prune.json'), 'w'), indent=1)
    print(json.dumps({'before': before['total'], 'after': after['total']}))


if __name__ == '__main__':
    main()
