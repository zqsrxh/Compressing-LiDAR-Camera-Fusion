#!/usr/bin/env python3
"""fusion3d pod orchestrator, runs on the DGX Spark. Talks to pods over SSH only (no RunPod API here).
pods.json: {label: {"host":..,"port":..,"role":"pilotA|pilotB|seed2", "seed": null|int}}
Writes state.json, orchestrator.log, decision.json, NEED_POD_SEED2, POD_DONE_<label>; downloads to runs_v5/."""
import json, os, subprocess, time, shlex, datetime
H = os.path.expanduser('~/.claude-science-scratch/fusion3d/pods'); os.chdir(H)
KEY = os.path.expanduser('~/.ssh/runpod_fusion')
O = ['-i', KEY, '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', '-o', 'ConnectTimeout=20',
     '-o', 'LogLevel=ERROR', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=30']
B = '/workspace/fusion3d'
DL = os.path.expanduser('~/.claude-science-scratch/fusion3d/runs_v5'); os.makedirs(DL, exist_ok=True)
PASS, FALLBACK = 55.6, 54.0
def log(m):
    line = f"[{datetime.datetime.utcnow().isoformat(timespec='seconds')}Z] {m}"
    print(line, flush=True); open('orchestrator.log', 'a').write(line + '\n')
def ssh(p, cmd, t=120):
    try:
        r = subprocess.run(['ssh', *O, '-p', str(p['port']), 'root@' + p['host'], cmd], capture_output=True, text=True, timeout=t, stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired:
        return 124, ''
def scp_from(p, remote, local, t=1800):
    os.makedirs(os.path.dirname(local), exist_ok=True)
    try:
        r = subprocess.run(['scp', *O, '-P', str(p['port']), f"root@{p['host']}:{remote}", local + '.part'], capture_output=True, text=True, timeout=t, stdin=subprocess.DEVNULL)
        if r.returncode == 0: os.replace(local + '.part', local); return True
    except subprocess.TimeoutExpired: pass
    return False
def scp_to(p, local, remote, t=600):
    r = subprocess.run(['scp', *O, '-P', str(p['port']), local, f"root@{p['host']}:{remote}"], capture_output=True, text=True, timeout=t, stdin=subprocess.DEVNULL)
    return r.returncode == 0
def jload(f, d):
    try: return json.load(open(f))
    except Exception: return d
def save(st): json.dump(st, open('state.json.tmp', 'w'), indent=1); os.replace('state.json.tmp', 'state.json')

st = jload('state.json', {'phase': 'pilot', 'downloaded': [], 'sens_synced': False, 'launched': [], 'done_pods': []})
while True:
    pods = jload('pods.json', {})
    # ------------------------------------------------ deploy pods that need launching (new seed2 pod)
    for lab, p in pods.items():
        if p.get('launch_cmd') and lab not in st['launched']:
            rc, out = ssh(p, 'true', 30)
            if rc != 0: continue
            r = subprocess.run(['bash', 'pod_launch.sh', p['host'], str(p['port']), p['launch_cmd']], capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL)
            log(f'{lab}: launched ({p["launch_cmd"]}): ' + r.stdout.strip().replace('\n', ' | ')[-300:])
            st['launched'].append(lab); save(st)
    # ------------------------------------------------ pilot phase
    if st['phase'] == 'pilot':
        res = {}
        for lab in ('pilotA', 'pilotB'):
            if lab not in pods: continue
            p = pods[lab]
            rc, out = ssh(p, f"cd {B}; echo FILES=$(ls runs/fusion_r50_s0 2>/dev/null | tr '\\n' ' '); echo LAST=$(tail -n1 runs/fusion_r50_s0/train_log.jsonl 2>/dev/null); "
                             f"echo MAP=$(python3 -c \"import json;print(json.load(open('runs/fusion_r50_s0/eval_tune.json'))['summary']['mAP3d_mod_R40'])\" 2>/dev/null); "
                             f"echo SETUP=$(tail -c 200 setup.log 2>/dev/null | tr '\\r\\n' '  ' | tail -c 80)")
            if rc != 0: res[lab] = ('unreachable', None); continue
            kv = {}
            for ln in out.split('\n'):
                if '=' in ln:
                    k_, v_ = ln.split('=', 1); kv[k_.strip()] = v_.strip()
            files = kv.get('FILES', ''); last = kv.get('LAST', '')
            m = None
            try: m = float(kv.get('MAP', ''))
            except Exception: pass
            failed = 'FAILED' in files
            if not last: last = 'setup: ' + kv.get('SETUP', '')
            res[lab] = ('done' if (m is not None and 'DONE' in files) else 'failed' if failed else 'running', m)
            st.setdefault('pilot_last', {})[lab] = last[:300]
        save(st)
        if int(time.time()) % 600 < 70:
            log('pilot status ' + json.dumps({k: v for k, v in res.items()}) + ' last=' + json.dumps(st.get('pilot_last', {}))[:600])
        if len(res) == 2 and all(v[0] in ('done', 'failed') for v in res.values()):
            ok = {k: v[1] for k, v in res.items() if v[0] == 'done' and v[1] is not None}
            log('pilot results ' + json.dumps(ok))
            win = max(ok, key=ok.get) if ok else None
            score = ok.get(win) if win else None
            drop = {'pilotA': '0.0', 'pilotB': '0.5'}
            if win and score >= FALLBACK:
                dec = {'winner': win, 'tune_mAP': score, 'drop': drop[win], 'tier': 'pass' if score >= PASS else 'fallback', 'all': ok}
                json.dump(dec, open('decision.json', 'w'), indent=1); log('DECISION ' + json.dumps(dec))
                other = 'pilotB' if win == 'pilotA' else 'pilotA'
                rc, out = ssh(pods[win], f"cd {B} && bash start_full.sh 0 {drop[win]} 2>&1 | tail -4", 300); log(f'{win} -> seed 0: ' + out.replace('\n', ' | '))
                rc, out = ssh(pods[other], f"cd {B} && bash start_full.sh 1 {drop[win]} 2>&1 | tail -4", 300); log(f'{other} -> seed 1: ' + out.replace('\n', ' | '))
                pods[win]['seed'] = 0; pods[other]['seed'] = 1
                json.dump(pods, open('pods.json', 'w'), indent=1)
                open('NEED_POD_SEED2', 'w').write(drop[win] + '\n')
                st['phase'] = 'full'; st['drop'] = drop[win]; save(st)
            else:
                dec = {'winner': None, 'all': ok, 'stop': True}
                json.dump(dec, open('decision.json', 'w'), indent=1); log('DECISION STOP ' + json.dumps(dec))
                for lab in ('pilotA', 'pilotB'):
                    scp_from(pods[lab], f'{B}/runs/fusion_r50_s0/train_log.jsonl', f'{DL}/pilot_{lab}/train_log.jsonl')
                    scp_from(pods[lab], f'{B}/runs/fusion_r50_s0/eval_tune.json', f'{DL}/pilot_{lab}/eval_tune.json')
                open('POD_DONE_pilotA', 'w').write('stop'); open('POD_DONE_pilotB', 'w').write('stop')
                st['phase'] = 'stopped'; save(st)
    # ------------------------------------------------ full phase
    if st['phase'] == 'full':
        seedpods = {p['seed']: (lab, p) for lab, p in pods.items() if p.get('seed') is not None}
        # incremental downloads
        for s, (lab, p) in seedpods.items():
            if lab in st['done_pods']: continue
            rc, out = ssh(p, f"cd {B}/runs 2>/dev/null && for d in */; do [ -f $d/DONE ] && echo ${{d%/}}; done")
            if rc != 0: continue
            for task in out.split():
                if task in st['downloaded']: continue
                ok = True
                if task.startswith('valfusion_'):
                    ok &= scp_from(p, f'{B}/runs/{task[3:]}/eval_val.json', f'{DL}/{task[3:]}/eval_val.json')
                for f in ['model_final.pt', 'eval_tune.json', 'eval_val.json', 'train_log.jsonl', 'sensitivity.json', 'allocation.json', 'pruned_prune.json']:
                    rc2, ex = ssh(p, f"test -f {B}/runs/{task}/{f} && echo y")
                    if ex.strip() == 'y':
                        ok &= scp_from(p, f'{B}/runs/{task}/{f}', f'{DL}/{task}/{f}')
                if ok:
                    st['downloaded'].append(task); save(st); log(f'downloaded {task} from {lab}')
        # sensitivity sync
        if not st['sens_synced'] and len(seedpods) == 3:
            have = [s for s in range(3) if os.path.exists(f'{DL}/sens_r50_s{s}/sensitivity.json')]
            if len(have) == 3:
                for s, (lab, p) in seedpods.items():
                    for t in range(3):
                        if t == s: continue
                        ssh(p, f"mkdir -p {B}/runs/sens_r50_s{t}")
                        scp_to(p, f'{DL}/sens_r50_s{t}/sensitivity.json', f'{B}/runs/sens_r50_s{t}/sensitivity.json')
                        ssh(p, f"touch {B}/runs/sens_r50_s{t}/DONE")
                st['sens_synced'] = True; save(st); log('sensitivity files synced to all pods')
        # completion
        for s, (lab, p) in seedpods.items():
            if lab in st['done_pods']: continue
            rc, out = ssh(p, f"bash {B}/pod_status.sh 2>/dev/null | grep -E 'DONE|FAILED|ALL_DONE|FINISHED_WITH_FAILURES|WORKERS_ALIVE|RUNNING' | head -6")
            if int(time.time()) % 900 < 70: log(f'{lab} seed{s}: ' + out.replace('\n', ' | ')[:400])
            if 'ALL_DONE' in out or 'FINISHED_WITH_FAILURES' in out:
                rc, o2 = ssh(p, f"bash {B}/pack_results.sh 2>&1 | tail -2", 900)
                if scp_from(p, f'{B}/results_fusion3d.tgz', f'{DL}/results_fusion3d_v5_seed{s}.tgz', 3600):
                    open(f'POD_DONE_{lab}', 'w').write(out); st['done_pods'].append(lab); save(st)
                    log(f'{lab} seed{s} finished and packed: ' + out.replace('\n', ' | ')[:300])
        if len(st['done_pods']) == 3:
            st['phase'] = 'complete'; save(st); log('ALL PODS COMPLETE')
    if st['phase'] in ('complete', 'stopped'):
        break
    time.sleep(60)
