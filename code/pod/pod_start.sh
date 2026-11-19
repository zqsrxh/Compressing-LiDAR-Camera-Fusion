#!/bin/bash
# Start one detached worker per GPU (safe to re-run: existing workers are left alone).
B=${B:-/workspace/fusion3d}
N=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((N-1))); do
  if pgrep -f "pod_worker.sh $g\$" >/dev/null; then echo "worker $g already running"; continue; fi
  nohup bash $B/pod_worker.sh $g > $B/worker_gpu$g.log 2>&1 &
  echo "started worker $g"
done
