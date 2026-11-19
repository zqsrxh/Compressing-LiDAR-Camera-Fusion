"""Measured latency lookup table (TensorRT FP16, batch 1) for pruned variants of one base model.
Variants: each unit alone at each ratio, uniform ratios over all units, and global-L1 targets.
Architectures (and therefore latency) do not depend on the seed, so one base model suffices.
Must run with the GPU otherwise idle."""
import argparse, json, os, copy
from types import SimpleNamespace
import torch
from fusion3d.prune import plan_uniform, plan_global, apply_plan
from fusion3d.profile import profile_model
import deploy_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ratios', default='0.25,0.5,0.7')
    ap.add_argument('--global-targets', default='0.3,0.4,0.5,0.6')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--device-label', default='gb10')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lut_path = os.path.join(a.out, f'latency_lut_{a.device_label}.json')
    lut = json.load(open(lut_path)) if os.path.exists(lut_path) else {}
    base = torch.load(a.model, map_location='cpu', weights_only=False)
    ratios = [float(x) for x in a.ratios.split(',')]
    variants = [('base', {})]
    units = ['image', 'fusion', 'lidar'] if base.use_image else ['lidar']
    for u in units:
        for r in ratios:
            variants.append((f'{u}_{r}', {'ratios': {u: r}}))
    for r in ratios:
        variants.append((f'uniform_{r}', {'ratios': {u: r for u in units}}))
    for t in [float(x) for x in a.global_targets.split(',') if x]:
        variants.append((f'global_{t}', {'target': t}))
    for name, spec in variants:
        if name in lut:
            continue
        m = copy.deepcopy(base)
        if 'ratios' in spec:
            apply_plan(m, plan_uniform(m, spec['ratios'], 'l1'))
        elif 'target' in spec:
            apply_plan(m, plan_global(m, spec['target'], 'l1'))
        mp = os.path.join(a.out, f'lut_{name}.pt'); torch.save(m, mp)
        prof = profile_model(m, a.root)
        ns = SimpleNamespace(root=a.root, model=mp, out=a.out, tag=f'lut_{name}', int8='', calib='entropy', calib_n=0,
                             calib_bev=0, split='tune', torch='fp32', n=a.n, warmup=50, device_label=a.device_label)
        deploy_run.cmd_build(ns)
        deploy_run.cmd_bench(ns)
        b = json.load(open(os.path.join(a.out, f'bench_lut_{name}_{a.device_label}.json')))
        lut[name] = {'median_ms': b['median_ms'], 'p95_ms': b['p95_ms'], 'stages': b['stages_median_ms'],
                     'GFLOPs': prof['total']['GFLOPs'], 'params_M': prof['total']['params_M'],
                     'unit_GFLOPs': {u: prof[u]['GFLOPs'] for u in ['image', 'fusion', 'lidar', 'head']}}
        json.dump(lut, open(lut_path, 'w'), indent=1)
        print(name, json.dumps(lut[name]), flush=True)
        for ext in ['_A.engine', '_B.engine', '_fp32_A.onnx', '_fp32_B.onnx', '_A.onnx', '_B.onnx']:
            p = os.path.join(a.out, f'lut_{name}{ext}')
            if os.path.exists(p) and ext.endswith('onnx'):
                os.remove(p)


if __name__ == '__main__':
    main()
