#!/bin/bash
P=$HOME/.claude-science-scratch/fusion3d; RV=$P/runs_v5
while pgrep -f "[d]eploy_ft_all.sh" > /dev/null; do sleep 30; done
cd $P/code; . $P/venv-deploy/bin/activate; export PYTHONPATH=$P/code
for s in 0 1 2; do
  M=$P/runs/lidar_s$s/model_final.pt; O=$RV/deploy_lidar_s$s
  [ -f $O/deploy_eval_int8head_val.json ] && continue
  python -u deploy_run.py build --root $P/data/kitti --model $M --out $O --tag int8head --int8 head >> $O/log.txt 2>&1
  python -u deploy_run.py eval --root $P/data/kitti --model $M --out $O --tag int8head --split val >> $O/log.txt 2>&1
  rm -f $O/int8head_fp32_*.onnx
done
M=$P/runs/lidar_s0/model_final.pt; O=$RV/deploy_lidar_s0
for pass in 1 2 3; do python -u deploy_run.py bench --root $P/data/kitti --model $M --out $O --tag int8head --n 300 --device-label gb10_p$pass >> $O/bench_log.txt 2>&1; done
ls $RV/deploy_lidar_s*/deploy_eval_int8head_val.json
