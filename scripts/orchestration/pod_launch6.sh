#!/bin/bash
# pod_launch.sh HOST PORT CMD  -> copy bundle v5, extract, run "setup (retry once) && CMD" in background on the pod
H=$1; PT=$2; CMD=$3
O="-i $HOME/.ssh/runpod_fusion -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR"
B=$HOME/.claude-science-scratch/fusion3d/pods/fusion3d_pod_bundle_v6.tgz
for i in $(seq 1 20); do ssh $O -p $PT root@$H true 2>/dev/null && break; sleep 10; done
scp $O -P $PT $B root@$H:/workspace/bundle.tgz
ssh $O -p $PT root@$H "md5sum /workspace/bundle.tgz; mkdir -p /workspace/fusion3d && tar xzf /workspace/bundle.tgz --no-same-owner -C /workspace/fusion3d && cd /workspace/fusion3d && setsid nohup bash -c 'bash pod_setup.sh > setup.log 2>&1 || bash pod_setup.sh >> setup.log 2>&1; $CMD > launch.log 2>&1' < /dev/null > /dev/null 2>&1 & sleep 2; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; ls /workspace/fusion3d | head -30 | tr '\n' ' '"
