"""Aggregate results: base models, sensitivity, allocation, compressed models; paired statistics."""
import json, os, glob, re, sys
import numpy as np
import pandas as pd
from scipy import stats

CLS = ['Car', 'Pedestrian', 'Cyclist']


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def summary_row(run_dir, split='val'):
    e = load(os.path.join(run_dir, f'eval_{split}.json'))
    if e is None:
        return None
    s = e['summary']
    r = {'mAP3d': s['mAP3d_mod_R40'], 'mAPbev': s['mAPbev_mod_R40']}
    for c in CLS:
        for d in ['easy', 'moderate', 'hard']:
            r[f'{c}_3d_{d}'] = s[f'{c}_3d/{d}_R40']
    return r


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); prev = 0
    for k, i in enumerate(order):
        prev = max(prev, (m - k) * pvals[i]); adj[i] = min(1.0, prev)
    return adj


def main(runs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(os.path.join(runs, '*'))):
        name = os.path.basename(d)
        m = re.match(r'(.+)_s(\d)$', name)
        if not m:
            continue
        cfg, seed = m.group(1), int(m.group(2))
        if cfg.startswith('sens'):
            continue
        r = summary_row(d)
        if r is None:
            continue
        r.update(config=cfg, seed=seed)
        pr = load(os.path.join(d, 'pruned_prune.json'))
        if pr:
            r.update(params_M=pr['after']['total']['params_M'], GFLOPs=pr['after']['total']['GFLOPs'],
                     **{f'GFLOPs_{u}': pr['after'][u]['GFLOPs'] for u in ['image', 'fusion', 'lidar', 'head']})
            if pr.get('extra', {}).get('ratios'):
                r['ratios'] = json.dumps(pr['extra']['ratios'])
        tl = os.path.join(d, 'train_log.jsonl')
        if os.path.exists(tl):
            ep = [json.loads(l) for l in open(tl) if l.startswith('{"epoch')]
            if ep:
                r['epoch_s_median'] = float(np.median([e['epoch_s'] for e in ep]))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'all_runs_val.csv'), index=False)
    num = [c for c in df.columns if c not in ('config', 'seed', 'ratios')]
    agg = df.groupby('config')[num].agg(['mean', 'std', 'count'])
    agg.to_csv(os.path.join(out_dir, 'config_summary_val.csv'))
    ci = df.groupby('config')['mAP3d'].apply(lambda x: ci95(x.values)).rename('mAP3d_ci95')
    ci.to_csv(os.path.join(out_dir, 'config_ci95.csv'))
    # paired comparisons (paired by base seed)
    comps = []
    for rb, others in [('0.5', ['uniL1_0.5', 'uniFPGM_0.5', 'globalL1_0.5']), ('0.7', ['uniL1_0.7'])]:
        ours = df[df.config == f'ft_alloc{rb}'].set_index('seed')['mAP3d']
        for o in others:
            b = df[df.config == f'ft_{o}'].set_index('seed')['mAP3d']
            s = sorted(set(ours.index) & set(b.index))
            if len(s) < 2:
                continue
            diff = ours.loc[s].values - b.loc[s].values
            t, p = stats.ttest_rel(ours.loc[s].values, b.loc[s].values)
            comps.append({'budget': rb, 'ours': f'ft_alloc{rb}', 'baseline': f'ft_{o}', 'n': len(s),
                          'mean_diff': float(diff.mean()), 'sd_diff': float(diff.std(ddof=1)),
                          'ci95_low': float(diff.mean() - ci95(diff)), 'ci95_high': float(diff.mean() + ci95(diff)),
                          't': float(t), 'df': len(s) - 1, 'p': float(p)})
    if comps:
        cd = pd.DataFrame(comps)
        for rb in cd.budget.unique():
            m = cd.budget == rb
            cd.loc[m, 'p_holm'] = holm(cd.loc[m, 'p'].values)
        cd.to_csv(os.path.join(out_dir, 'paired_tests.csv'), index=False)
    # sensitivity
    srows = []
    for p in sorted(glob.glob(os.path.join(runs, 'sens_*', 'sensitivity.json'))):
        seed = int(re.search(r'_s(\d)', p).group(1)); sj = json.load(open(p))
        b = sj['base_bnrecal']['mAP3d_mod_R40']
        for k, v in sj.items():
            m = re.match(r'(image|fusion|lidar)_(l1|fpgm)_([\d.]+)$', k)
            if m:
                srows.append({'seed': seed, 'unit': m.group(1), 'criterion': m.group(2), 'ratio': float(m.group(3)),
                              'mAP3d': v['mAP3d_mod_R40'], 'drop': b - v['mAP3d_mod_R40'], 'GFLOPs': v['profile']['total']['GFLOPs'],
                              'base_bnrecal': b, 'base_raw': sj['base']['mAP3d_mod_R40']})
    if srows:
        sd = pd.DataFrame(srows); sd.to_csv(os.path.join(out_dir, 'sensitivity_long.csv'), index=False)
        sd.groupby(['criterion', 'unit', 'ratio'])['drop'].agg(['mean', 'std', 'count']).to_csv(os.path.join(out_dir, 'sensitivity_summary.csv'))
    print(df.groupby('config')['mAP3d'].agg(['mean', 'std', 'count']).round(2).to_string())
    if comps:
        print(pd.DataFrame(comps).round(3).to_string())


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
