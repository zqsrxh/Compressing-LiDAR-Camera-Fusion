#!/usr/bin/env python3
"""make_tables.py RV LIDAR_RUNS OUT : builds summary.json and LaTeX tables from all result files."""
import json, os, sys, glob, re
import numpy as np
from scipy import stats
RV, LR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(OUT, exist_ok=True)
CLS = ['Car', 'Pedestrian', 'Cyclist']
def J(p):
    try: return json.load(open(p))
    except Exception: return None
def summ(p):
    j = J(p); return j['summary'] if j else None
def ci95(x):
    x = np.asarray(x, float); return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float('nan')
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); prev = 0
    for k, i in enumerate(o): prev = max(prev, (m - k) * p[i]); adj[i] = min(1, prev)
    return adj
S = {}
# ---------------- base models (val and tune)
cfgs = {'LiDAR-only': [f'{LR}/lidar_s{s}' for s in range(3)], 'Fusion (R50)': [f'{RV}/fusion_r50_s{s}' for s in range(3)]}
for m in ['alloc0.5', 'uniL1_0.5', 'uniFPGM_0.5', 'globalL1_0.5']:
    cfgs['ft_' + m] = [f'{RV}/ft_{m}_s{s}' for s in range(3)]
rows = {}
for name, dirs in cfgs.items():
    r = {'val': [], 'tune': [], 'cls_val': {c: [] for c in CLS}, 'seeds': []}
    for s, d in enumerate(dirs):
        v = summ(f'{d}/eval_val.json'); t = summ(f'{d}/eval_tune.json')
        if v is None: continue
        r['seeds'].append(s); r['val'].append(v['mAP3d_mod_R40']); r['tune'].append(t['mAP3d_mod_R40'] if t else float('nan'))
        for c in CLS: r['cls_val'][c].append(v[f'{c}_3d/moderate_R40'])
        pr = J(f'{d}/pruned_prune.json')
        if pr:
            r.setdefault('GFLOPs', []).append(pr['after']['total']['GFLOPs']); r.setdefault('params_M', []).append(pr['after']['total']['params_M'])
            r.setdefault('GFLOPs_unit', []).append({u: pr['after'][u]['GFLOPs'] for u in ['image', 'fusion', 'lidar', 'head']})
            if pr.get('extra', {}).get('ratios'): r['ratios'] = pr['extra']['ratios']
            if pr.get('extra', {}).get('global_ratio'): r.setdefault('global_ratio', []).append(pr['extra']['global_ratio'])
    rows[name] = r
S['models'] = rows
# paired tests: alloc vs each baseline (val mAP3d)
comps = []
a = rows.get('ft_alloc0.5', {})
for b in ['ft_uniL1_0.5', 'ft_uniFPGM_0.5', 'ft_globalL1_0.5']:
    bb = rows.get(b, {})
    common = sorted(set(a.get('seeds', [])) & set(bb.get('seeds', [])))
    if len(common) < 2: continue
    x = np.array([a['val'][a['seeds'].index(s)] for s in common]); y = np.array([bb['val'][bb['seeds'].index(s)] for s in common])
    d = x - y; t, p = stats.ttest_rel(x, y)
    comps.append({'baseline': b, 'n': len(common), 'mean_diff': float(d.mean()), 'ci_low': float(d.mean() - ci95(d)), 'ci_high': float(d.mean() + ci95(d)), 't': float(t), 'p': float(p)})
if comps:
    adj = holm([c['p'] for c in comps])
    for c, q in zip(comps, adj): c['p_holm'] = float(q)
S['paired'] = comps
# ---------------- pruning sensitivity
sens = {}
for s in range(3):
    sj = J(f'{RV}/sens_r50_s{s}/sensitivity.json')
    if not sj: continue
    b = sj['base_bnrecal']['mAP3d_mod_R40']
    for k, v in sj.items():
        m = re.match(r'(image|fusion|lidar)_(l1|fpgm)_([\d.]+)$', k)
        if m: sens.setdefault(k, []).append(b - v['mAP3d_mod_R40'])
    sens.setdefault('_base_bnrecal', []).append(b); sens.setdefault('_base_raw', []).append(sj['base']['mAP3d_mod_R40'])
S['sensitivity'] = {k: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else None, 'n': len(v), 'vals': v} for k, v in sens.items()}
S['allocation'] = J(f'{RV}/alloc/allocation.json')
# ---------------- INT8 per unit (tune)
q = {}
for s in range(3):
    d = f'{RV}/int8_r50_s{s}'; fp = summ(f'{d}/deploy_eval_fp16_tune.json')
    if not fp: continue
    q.setdefault('fp16', []).append(fp['mAP3d_mod_R40'])
    for u in ['image', 'fusion', 'lidar', 'head']:
        e = summ(f'{d}/deploy_eval_int8_{u}_tune.json')
        if e: q.setdefault(u, []).append(fp['mAP3d_mod_R40'] - e['mAP3d_mod_R40'])
S['int8_units_fusion'] = {k: {'mean': float(np.mean(v)), 'vals': v} for k, v in q.items()}
json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1)
def ms(v): return f"{np.mean(v):.2f} $\\pm$ {np.std(v, ddof=1):.2f}" if len(v) > 1 else (f"{v[0]:.2f}" if v else '--')
# table: base + compressed
L = ['\\begin{tabular}{lrrrrr}', '\\hline', 'Model & mAP$_{3D}$ (val) & Car & Pedestrian & Cyclist & GFLOPs \\\\', '\\hline']
for name, r in rows.items():
    if not r['val']: continue
    g = ms(r['GFLOPs']) if r.get('GFLOPs') else '--'
    L.append(f"{name.replace('_', ' ')} & {ms(r['val'])} & {ms(r['cls_val']['Car'])} & {ms(r['cls_val']['Pedestrian'])} & {ms(r['cls_val']['Cyclist'])} & {g} \\\\")
L += ['\\hline', '\\end{tabular}']
open(f'{OUT}/tab_main.tex', 'w').write('\n'.join(L) + '\n')
print(json.dumps({k: (np.round(v['val'], 2).tolist() if v['val'] else None) for k, v in rows.items()}))
print(json.dumps(comps, indent=1))
print(json.dumps(S['int8_units_fusion']))
