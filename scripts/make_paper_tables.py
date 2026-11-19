#!/usr/bin/env python3
"""make_paper_tables.py RESULTS_DIR OUT_DIR : regenerates the numeric content of the paper tables from the released JSON files.
RESULTS_DIR is the `results/` folder of this archive."""
import json, os, sys, glob
import numpy as np
R, O = sys.argv[1], sys.argv[2]; os.makedirs(O, exist_ok=True)
S = json.load(open(f'{R}/tables/summary.json')); S2 = json.load(open(f'{R}/tables/summary2.json'))
G = json.load(open(f'{R}/tables/gb10x.json')); B3 = json.load(open(f'{R}/tables/bench3080_all.json'))
def msd(v): v = [x for x in v if x is not None]; return f"{np.mean(v):.2f} +- {np.std(v, ddof=1):.2f}"
out = []
out.append('# Main table (val mAP3D moderate R40, PyTorch FP32)')
for k, r in S['models'].items():
    out.append(f"{k}: {msd(r['val'])} | Car {msd(r['cls_val']['Car'])} | Ped {msd(r['cls_val']['Pedestrian'])} | Cyc {msd(r['cls_val']['Cyclist'])} | GFLOPs {np.mean(r['GFLOPs']) if r.get('GFLOPs') else '-'}")
out.append('# Paired tests (allocation minus baseline, Holm over 3)')
for c in S['paired']: out.append(json.dumps(c))
out.append('# One-shot sensitivity (tune drop, mean +- sd over seeds)')
for k, v in S['sensitivity'].items(): out.append(f"{k}: {v['mean']:.2f} +- {v['sd'] if v['sd'] is None else round(v['sd'], 2)}")
out.append('# Allocation'); out.append(json.dumps(S['allocation']['budgets']))
out.append('# INT8 per unit, fusion model (tune drop vs FP16 engine)')
for k, v in S['int8_units_fusion'].items(): out.append(f"{k}: {np.round(v['vals'], 2).tolist()} mean {v['mean']:.2f}")
out.append('# Deployed val accuracy (torch / fp16 / int8safe); LiDAR-only INT8 = head, see results/runs/deploy_lidar_s*/deploy_eval_int8head_val.json')
for k, r in S2['deploy_val'].items(): out.append(f"{k}: " + ' | '.join(f"{t} {msd(v)}" for t, v in r.items()))
out.append('# GB10 latency (pinned, 5 passes): mean of pass medians, p95, stages')
for k, v in sorted(G.items()): out.append(f"{k}: median {v['median']:.2f} p95 {v['p95']:.2f} stages {json.dumps({a: round(b, 2) for a, b in v['stages'].items()})}")
out.append('# RTX 3080 latency (3 passes)')
agg = {}
for k, v in B3.items():
    if 'median_ms' in v: agg.setdefault('|'.join(k.split('|')[:2]), []).append(v)
for k, js in sorted(agg.items()):
    out.append(f"{k}: median {np.mean([j['median_ms'] for j in js]):.2f} p95 {np.mean([j['p95_ms'] for j in js]):.2f} stages {json.dumps({s: round(float(np.mean([j['stages_median_ms'][s] for j in js])), 2) for s in js[0]['stages_median_ms']})}")
open(f'{O}/paper_numbers.txt', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))

# ---- additional raw data (results/extra)
X = f'{R}/extra'
def m3(p):
    s = json.load(open(p))['summary']; return s.get('mAP3d_mod_R40', float(np.mean([s[c + '_3d/moderate_R40'] for c in ['Car', 'Pedestrian', 'Cyclist']])))
out = ['# LiDAR-only per-unit INT8 (tune drop vs FP16 engine); FP16 tune from run1, units from run2 (fixed head matching)']
for s in range(3):
    fp = m3(f'{X}/lidar_int8_units/run1/lidar_s{s}/deploy_eval_fp16_tune.json')
    row = {u: round(fp - m3(f'{X}/lidar_int8_units/run2_fixed_head_regex/lidar_s{s}/deploy_eval_int8_{u}_tune.json'), 2)
           for u in ['lidar_entropy', 'lidar_max', 'head_entropy']}
    out.append(f'seed {s}: fp16 {fp:.2f} ' + json.dumps(row))
for s in range(3):
    out.append(f'LiDAR-only INT8-head val seed {s}: {m3(f"{R}/runs/deploy_lidar_s{s}/deploy_eval_int8head_val.json"):.2f}')
out.append('# overfitting diagnostic'); out.append(open(f'{X}/diagnostics/fusion_r18_s0/diag.json').read()); out.append(open(f'{X}/diagnostics/lidar/diag_lidar.json').read())
out.append('# split class shares'); out.append(open(f'{X}/split_class_shares.json').read())
out.append('# LiDAR-only profile'); out.append(open(f'{X}/lidar_only_profile.json').read())
out.append('# evaluator check (devkit vs python)'); out.append(open(f'{X}/evaluator_check/compare_devkit_vs_python.txt').read()[-1500:])
open(f'{O}/paper_numbers.txt', 'a').write('\n'.join(out) + '\n'); print('\n'.join(out))

out=['# per-class Welch vs LiDAR-only (Holm over classes)']
for k,v in json.load(open(f'{R}/tables/per_class_welch_vs_lidar.json')).items():
    for r in v: out.append(f"{k} {r['class']}: diff {r['diff']:+.2f} CI [{r['ci_low']:.2f}, {r['ci_high']:.2f}] p {r['p']:.4f} p_holm {r['p_holm']:.4f}")
open(f'{O}/paper_numbers.txt','a').write('\n'.join(out)+'\n'); print('\n'.join(out))
