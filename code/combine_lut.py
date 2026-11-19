"""Average several latency-table passes (same variants) into one table: mean of per-pass medians."""
import json, sys
import numpy as np
out_path, paths = sys.argv[1], sys.argv[2:]
tabs = [json.load(open(p)) for p in paths]
keys = [k for k in tabs[0] if all(k in t for t in tabs)]
out = {}
for k in keys:
    e = dict(tabs[0][k])
    e['median_ms'] = float(np.mean([t[k]['median_ms'] for t in tabs]))
    e['p95_ms'] = float(np.mean([t[k]['p95_ms'] for t in tabs]))
    e['stages'] = {s: float(np.mean([t[k]['stages'][s] for t in tabs])) for s in e['stages']}
    e['passes_median_ms'] = [t[k]['median_ms'] for t in tabs]
    e['passes_net_ms'] = [sum(v for s, v in t[k]['stages'].items() if s != 'post') for t in tabs]
    out[k] = e
json.dump(out, open(out_path, 'w'), indent=1)
for k, e in out.items():
    print(k, round(e['median_ms'], 2), [round(x, 2) for x in e['passes_net_ms']])
