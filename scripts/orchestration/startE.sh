#!/bin/bash
# bash startE.sh SEED : full phase with recipe E (COCO frozen image branch, modality dropout 0.5, PointPillars schedule)
set -e
SEED=$1; B=/workspace/fusion3d; NEW=/workspace/v6
pkill -f '[p]od_worker.sh' || true; sleep 2
[ -d $NEW/code ] || { mkdir -p $NEW && tar xzf /workspace/v6.tgz --no-same-owner -C $NEW; }
rm -rf $B/code && cp -r $NEW/code $B/code
cp $NEW/seed_tasks.py $NEW/pod_worker.sh $NEW/pod_start.sh $NEW/pod_status.sh $NEW/pack_results.sh $B/
KEEP=0
if [ "$SEED" = 0 ] && [ "$(cat $B/runs/fusion_r50_s0/PILOT_MULT 2>/dev/null)" = "E" ]; then KEEP=1; fi
if [ $KEEP = 0 ] && [ -d $B/runs ] && [ "$(ls -A $B/runs 2>/dev/null)" ]; then pkill -f '[t]rain.py --root' || true; sleep 2; mv $B/runs $B/runs_old_$(date -u +%H%M%S); fi
mkdir -p $B/runs
cp /workspace/tasks_full_E.tsv $B/tasks_full.tsv
cd $B && python seed_tasks.py $SEED
echo KEPT_PILOT=$KEEP; grep -c -- '--img-dropout 0.5' $B/tasks.tsv
bash $B/pod_start.sh
echo FULL_E_STARTED seed $SEED
