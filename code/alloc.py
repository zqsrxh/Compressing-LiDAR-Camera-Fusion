"""Sensitivity-guided allocation of pruning ratios per unit.

Inputs: one-shot sensitivity results (one JSON per base seed, from sensitivity.py) and the measured GB10
TensorRT FP16 latency lookup table (latency_lut.py). For a latency budget equal to the measured latency of
uniform pruning at ratio r_b, all combinations of unit ratios from {0, 0.25, 0.5, 0.7} are enumerated; the
predicted network latency is the base latency minus the per-unit savings measured with that unit pruned alone.
Among combinations whose predicted latency is within the budget (1% tolerance), the one with the smallest
sum of seed-averaged accuracy losses (L1 criterion, tuning split) is selected."""
import argparse, itertools, json
import numpy as np

UNITS = ['image', 'fusion', 'lidar']
GRID = [0.0, 0.25, 0.5, 0.7]


def net_ms(entry):
    return sum(v for k, v in entry['stages'].items() if k != 'post')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sens', nargs='+', required=True)
    ap.add_argument('--lut', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--criterion', default='l1')
    ap.add_argument('--budgets', default='0.25,0.5,0.7')
    ap.add_argument('--tol', type=float, default=0.01)
    ap.add_argument('--budget-mode', default='predicted', choices=['predicted', 'measured'],
                    help='predicted: budget = additive prediction for uniform pruning (same currency as the search); '
                         'measured: budget = measured latency of uniform pruning')
    a = ap.parse_args()
    sens = [json.load(open(p)) for p in a.sens]
    lut = json.load(open(a.lut))
    base_ms = net_ms(lut['base'])
    dap, dlat = {}, {}
    for u in UNITS:
        dap[(u, 0.0)], dlat[(u, 0.0)] = 0.0, 0.0
        for r in GRID[1:]:
            vals = [s['base_bnrecal']['mAP3d_mod_R40'] - s[f'{u}_{a.criterion}_{r}']['mAP3d_mod_R40'] for s in sens]
            dap[(u, r)] = float(np.mean(vals))
            dlat[(u, r)] = base_ms - net_ms(lut[f'{u}_{r}'])
    out = {'base_net_ms': base_ms, 'unit_table': {f'{u}_{r}': {'dAP_mean': dap[(u, r)], 'dlat_ms': dlat[(u, r)]}
                                                  for u in UNITS for r in GRID},
           'per_seed_dAP': {f'{u}_{r}': [s['base_bnrecal']['mAP3d_mod_R40'] - s[f'{u}_{a.criterion}_{r}']['mAP3d_mod_R40']
                                          for s in sens] for u in UNITS for r in GRID[1:]},
           'budgets': {}}
    for rb in [float(x) for x in a.budgets.split(',')]:
        uni_pred = base_ms - sum(dlat[(u, rb)] for u in UNITS)
        budget = uni_pred if a.budget_mode == 'predicted' else net_ms(lut[f'uniform_{rb}'])
        best = None
        for combo in itertools.product(GRID, repeat=3):
            pred = base_ms - sum(dlat[(u, r)] for u, r in zip(UNITS, combo))
            if pred > budget * (1 + a.tol):
                continue
            loss = sum(dap[(u, r)] for u, r in zip(UNITS, combo))
            key = (loss, pred)
            if best is None or key < best[0]:
                best = (key, combo, pred, loss)
        if best is None:
            out['budgets'][str(rb)] = {'feasible': False, 'budget_ms': budget, 'uniform_pred_ms': uni_pred}
            continue
        out['budgets'][str(rb)] = {'budget_mode': a.budget_mode, 'budget_ms': budget, 'measured_uniform_ms': net_ms(lut[f'uniform_{rb}']), 'uniform_pred_ms': uni_pred,
                                   'ratios': dict(zip(UNITS, best[1])), 'pred_ms': best[2], 'pred_dAP_sum': best[3],
                                   'uniform_pred_dAP_sum': sum(dap[(u, rb)] for u in UNITS)}
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps(out['budgets'], indent=1))


if __name__ == '__main__':
    main()
