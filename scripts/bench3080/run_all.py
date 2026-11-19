
import subprocess, sys, json, time, os
W=sys.argv[1]; env={**os.environ,'PYTHONPATH':os.path.join(W,'repo')}
cfgs=[('fusion_r50_s0','bench3080/models/fusion_r50_s0.pt',['fp16','int8safe']),('ft_alloc0.5_s0','bench3080/models/ft_alloc0.5_s0.pt',['fp16','int8safe']),
      ('ft_uniL1_0.5_s0','bench3080/models/ft_uniL1_0.5_s0.pt',['fp16','int8safe']),('ft_uniFPGM_0.5_s0','bench3080/models/ft_uniFPGM_0.5_s0.pt',['fp16','int8safe']),
      ('ft_globalL1_0.5_s0','bench3080/models/ft_globalL1_0.5_s0.pt',['fp16','int8safe']),('lidar','bench3080/models/lidar_s0.pt',['lidar_s0_fp16','lidar_s0_int8head'])]
def temp():
    return int(subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip())
out={}
for p in (1,2,3):
    for cfg,mp,tags in cfgs:
        for tag in tags:
            while temp()>55: time.sleep(10)
            r=subprocess.run([sys.executable,'repo/deploy_run.py','bench','--root','kitti_local','--model',mp,'--out','bench3080/'+cfg,'--tag',tag,'--n','300','--device-label',f'rtx3080_p{p}'],capture_output=True,text=True,env=env,cwd=W)
            o=[l for l in r.stdout.splitlines() if l.startswith('BENCH')]
            out[f'{cfg}|{tag}|p{p}']=json.loads(o[-1][6:]) if o else {'err':r.stderr[-500:]}
            json.dump(out,open('bench3080/bench3080_all.json','w'),indent=1)
print('DONE', len(out))
