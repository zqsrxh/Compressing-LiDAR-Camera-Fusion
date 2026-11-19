"""Check that every pruning plan yields a runnable model, and print per-unit params/FLOPs."""
import sys, json, copy, os
import torch
from fusion3d.model import Detector
from fusion3d.prune import plan_uniform, plan_global, apply_plan, build_groups
from fusion3d.profile import profile_model, profile_batch
from fusion3d.evaluation import to_device

def main():
    root = sys.argv[1]
    base = Detector(use_image=True, pretrained=False)
    print('n_groups', {u: sum(1 for g in build_groups(base) if g.unit == u) for u in ['image', 'fusion', 'lidar']})
    res = {'base': profile_model(copy.deepcopy(base), root)}
    specs = [('image0.5_l1', dict(ratios={'image': 0.5}), 'l1'), ('fusion0.5_l1', dict(ratios={'fusion': 0.5}), 'l1'),
             ('lidar0.5_fpgm', dict(ratios={'lidar': 0.5}), 'fpgm'), ('all0.5_l1', dict(ratios={'image': .5, 'fusion': .5, 'lidar': .5}), 'l1'),
             ('global0.4', dict(target=0.4), 'l1')]
    for name, spec, crit in specs:
        m = copy.deepcopy(base)
        plan = plan_global(m, spec['target'], crit) if 'target' in spec else plan_uniform(m, spec['ratios'], crit)
        apply_plan(m, plan)
        m = m.cuda().train()
        bb = to_device(profile_batch(root, True, split='train_sub', index=3), 'cuda')
        loss, _ = m(bb); loss.backward()
        m.eval()
        with torch.no_grad():
            p = m(bb)
        res[name] = profile_model(m, root)
        res[name]['loss'] = float(loss.detach())
        print(name, json.dumps(res[name]['total']), flush=True)
    json.dump(res, open('prune_test.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))

if __name__ == '__main__':
    main()
