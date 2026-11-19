#!/bin/bash
# bench_gb10_v2.sh : GB10 latency, process pinned to the Cortex-X925 cores (5-9,15-19), 5 rounds interleaved over all configs
P=$HOME/.claude-science-scratch/fusion3d; RV=$P/runs_v5
while pgrep -f "[l]idar_int8head.sh" > /dev/null; do sleep 30; done
cd $P/code; . $P/venv-deploy/bin/activate; export PYTHONPATH=$P/code
for r in 1 2 3 4 5; do
  for cfg in fusion_r50_s0 ft_alloc0.5_s0 ft_uniL1_0.5_s0 ft_uniFPGM_0.5_s0 ft_globalL1_0.5_s0; do
    for t in fp16 int8safe; do
      taskset -c 5-9,15-19 python -u deploy_run.py bench --root $P/data/kitti --model $RV/$cfg/model_final.pt --out $RV/deploy_$cfg --tag $t --n 300 --device-label gb10x_r$r >> $RV/deploy_$cfg/bench_log.txt 2>&1
    done
  done
  for t in fp16 int8head; do
    taskset -c 5-9,15-19 python -u deploy_run.py bench --root $P/data/kitti --model $P/runs/lidar_s0/model_final.pt --out $RV/deploy_lidar_s0 --tag $t --n 300 --device-label gb10x_r$r >> $RV/deploy_lidar_s0/bench_log.txt 2>&1
  done
done
ls $RV/deploy_*/bench_*gb10x* | wc -l
