#!/usr/bin/env python3
"""make_tables2.py RV LR OUT : deployed (TensorRT) val accuracy and GB10 latency summary."""
import json, glob, os, sys
import numpy as np
RV, LR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(OUT, exist_ok=True)
CLS = ['Car', 'Pedestrian', 'Cyclist']
def m3(p):
    s = json.load(open(p))['summary']; return s.get('mAP3d_mod_R40', float(np.mean([s[c + '_3d/moderate_R40'] for c in CLS])))
cfgs = {'lidar': [f'{RV}/deploy_lidar_s{s}' for s in range(3)], 'fusion_r50': [f'{RV}/deploy_fusion_r50_s{s}' for s in range(3)]}
for m in ['alloc0.5', 'uniL1_0.5', 'uniFPGM_0.5', 'globalL1_0.5']: cfgs['ft_' + m] = [f'{RV}/deploy_ft_{m}_s{s}' for s in range(3)]
torchdirs = {'lidar': [f'{LR}/lidar_s{s}' for s in range(3)], 'fusion_r50': [f'{RV}/fusion_r50_s{s}' for s in range(3)]}
for m in ['alloc0.5', 'uniL1_0.5', 'uniFPGM_0.5', 'globalL1_0.5']: torchdirs['ft_' + m] = [f'{RV}/ft_{m}_s{s}' for s in range(3)]
D = {}
for k, ds in cfgs.items():
    r = {'torch': [], 'fp16': [], 'int8safe': []}
    for s, d in enumerate(ds):
        tv = f'{torchdirs[k][s]}/eval_val.json'
        r['torch'].append(m3(tv) if os.path.exists(tv) else None)
        for t in ['fp16', 'int8safe']:
            p = f'{d}/deploy_eval_{t}_val.json'; r[t].append(m3(p) if os.path.exists(p) else None)
    D[k] = r
B = {}
for f in glob.glob(f'{RV}/deploy_*/bench_*gb10_p*.json'):
    j = json.load(open(f)); cfg = os.path.basename(os.path.dirname(f))[7:]
    B.setdefault((cfg, j['tag']), []).append(j)
L = {}
for (cfg, tag), js in B.items():
    L[f'{cfg}|{tag}'] = {'n_pass': len(js), 'median_ms': float(np.mean([j['median_ms'] for j in js])), 'median_ms_passes': [j['median_ms'] for j in js],
        'p95_ms': float(np.mean([j['p95_ms'] for j in js])), 'stages': {k: float(np.mean([j['stages_median_ms'][k] for j in js])) for k in js[0]['stages_median_ms']},
        'gpu_state_before': [j.get('gpu_state_before') for j in js]}
json.dump({'deploy_val': D, 'gb10': L}, open(f'{OUT}/summary2.json', 'w'), indent=1)
for k, r in D.items(): print(k, {t: [None if v is None else round(v, 2) for v in vs] for t, vs in r.items()})
for k, v in sorted(L.items()): print(k, v['n_pass'], round(v['median_ms'], 2), [round(x, 2) for x in v['median_ms_passes']], round(v['p95_ms'], 2), {a: round(b, 2) for a, b in v['stages'].items()})
