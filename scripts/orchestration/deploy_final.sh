#!/bin/bash
# deploy_final.sh MODEL OUTDIR : FP16 and INT8-safe (image,fusion) engines; val accuracy; ONNX kept for the RTX 3080
set -e
P=$HOME/.claude-science-scratch/fusion3d; M=$1; O=$2; mkdir -p $O
cd $P/code; . $P/venv-deploy/bin/activate; export PYTHONPATH=$P/code
for t in fp16 int8safe; do
  if [ $t = fp16 ]; then a=""; else a="--int8 image,fusion"; fi
  [ -f $O/deploy_eval_${t}_val.json ] && continue
  python -u deploy_run.py build --root $P/data/kitti --model $M --out $O --tag $t $a >> $O/log.txt 2>&1
  python -u deploy_run.py eval --root $P/data/kitti --model $M --out $O --tag $t --split val >> $O/log.txt 2>&1
  rm -f $O/${t}_fp32_*.onnx $O/${t}_fp32_*.data
done
ls $O
