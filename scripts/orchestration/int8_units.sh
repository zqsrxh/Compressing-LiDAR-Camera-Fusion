#!/bin/bash
# int8_units.sh MODEL OUTDIR : FP16 + per-unit INT8 (image, fusion, lidar, head) on the tuning subset
set -e
P=$HOME/.claude-science-scratch/fusion3d; M=$1; O=$2; mkdir -p $O
cd $P/code; . $P/venv-deploy/bin/activate; export PYTHONPATH=$P/code
for u in fp16 image fusion lidar head; do
  if [ $u = fp16 ]; then a=""; t=fp16; else a="--int8 $u"; t=int8_$u; fi
  [ -f $O/deploy_eval_${t}_tune.json ] && continue
  python -u deploy_run.py build --root $P/data/kitti --model $M --out $O --tag $t $a >> $O/log.txt 2>&1
  python -u deploy_run.py eval --root $P/data/kitti --model $M --out $O --tag $t --split tune >> $O/log.txt 2>&1
  rm -f $O/${t}_*.engine $O/${t}_fp32_*.onnx $O/${t}_fp32_*.data
done
ls $O
