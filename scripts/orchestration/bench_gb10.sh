#!/bin/bash
# bench_gb10.sh : latency of seed-0 deployed configs on the GB10 (idle GPU, cool start, 300 tune frames after 50 warm-up)
P=$HOME/.claude-science-scratch/fusion3d; RV=$P/runs_v5
cd $P/code; . $P/venv-deploy/bin/activate; export PYTHONPATH=$P/code
run() { M=$1; O=$2; T=$3
  [ -f $O/${T}_B.engine ] || return 0
  for pass in 1 2 3; do python -u deploy_run.py bench --root $P/data/kitti --model $M --out $O --tag $T --n 300 --device-label gb10_p$pass >> $O/bench_log.txt 2>&1; done; }
for cfg in fusion_r50_s0 ft_alloc0.5_s0 ft_uniL1_0.5_s0 ft_uniFPGM_0.5_s0 ft_globalL1_0.5_s0; do
  for t in fp16 int8safe; do run $RV/$cfg/model_final.pt $RV/deploy_$cfg $t; done
done
run $P/runs/lidar_s0/model_final.pt $RV/deploy_lidar_s0 fp16
ls $RV/deploy_*/bench_*gb10* | wc -l
