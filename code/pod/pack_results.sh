#!/bin/bash
# Pack results (no checkpoints, no dataset). Model weights of every run are included (final .pt files).
B=${B:-/workspace/fusion3d}
cd $B
tar czf results_fusion3d.tgz --exclude='*/code/*' --exclude='checkpoint.pt' --exclude='*.lock*' runs queue.log tasks.tsv worker_gpu*.log
md5sum results_fusion3d.tgz > results_fusion3d.tgz.md5
ls -la results_fusion3d.tgz; cat results_fusion3d.tgz.md5
