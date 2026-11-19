# Branch-wise compression of a LiDAR-camera fusion 3D detector

Code, split files and scripts for the study "Where to compress a LiDAR-camera fusion 3D detector:
branch-wise sensitivity to pruning and INT8 quantization with measured latency on x86 and ARM64 GPUs".

No KITTI data are redistributed. All derived files are regenerated from the official KITTI download
(KITTI is licensed CC BY-NC-SA 3.0; register and accept the terms at https://www.cvlibs.net/datasets/kitti/).

## Contents
| Path | Purpose |
|---|---|
| `fusion3d/model.py` | MVX-Net-style point-level fusion detector with a PointPillars LiDAR branch (pure PyTorch) |
| `fusion3d/data.py` | KITTI loader; points keep their original pixel coordinates so 3D augmentation keeps the point-to-pixel match |
| `fusion3d/prune.py` | structured channel pruning per unit (image / fusion / LiDAR), L1 and FPGM criteria, global L1 |
| `fusion3d/deploy.py`, `deploy_run.py` | BN folding, ONNX export, FP16 and per-unit INT8 (NVIDIA ModelOpt Q/DQ) TensorRT engines, accuracy and latency |
| `fusion3d/pcdet_lite/` | pure-Python parts of OpenPCDet (Apache-2.0) with compiled CUDA ops replaced (see NOTICE) |
| `prepare.py` | field-of-view point cache and the sequence-disjoint tuning split |
| `train.py`, `evaluate.py` | training (80-epoch OpenPCDet PointPillars schedule) and evaluation (KITTI R40) |
| `sensitivity.py`, `alloc.py`, `compress.py` | one-shot unit sensitivity, sensitivity-guided allocation, pruning |
| `latency_lut.py`, `combine_lut.py` | TensorRT latency lookup table used by the allocation |
| `analyze.py` | aggregation and paired statistics |
| `splits/` | `train.txt`/`val.txt` (3712/3769) and our `train_sub.txt`/`tune.txt` (3153/559) |
| `pod/` | scripts that ran the training queue on a 4-GPU cloud machine |

## Reproduction
1. Download KITTI 3D object (`data_object_velodyne`, `data_object_image_2`, `data_object_calib`, `data_object_label_2`)
   into `data/kitti/training/`, copy `splits/*.txt` to `data/kitti/ImageSets/`, then `python prepare.py --root data/kitti --fov-only`.
2. Train: `python train.py --root data/kitti --out runs/fusion_r50_s0 --use-image 1 --img-depth 50 --seed 0`
   (LiDAR-only: `--use-image 0`; ResNet-18 image branch: `--img-depth 18`).
3. Sensitivity and allocation: `python sensitivity.py ...`, `python alloc.py ...` (see `pod/tasks.tsv` for the exact commands).
4. Pruning and fine-tuning: `python compress.py ...` then `python train.py --init pruned.pt --epochs 20 --lr 0.0015 ...`.
5. Evaluation: `python evaluate.py --root data/kitti --model runs/<run>/model_final.pt --split val --out eval_val.json`.
6. Deployment: `python deploy_run.py build|eval|bench ...` (TensorRT >= 11, NVIDIA ModelOpt >= 0.48).
The exact command of every reported run is in `pod/tasks.tsv` and in the `args` field of each result JSON.

## Software
PyTorch 2.14.1 (GB10, CUDA 13.0) and the versions recorded in each result file; TensorRT 11.4.0.106;
NVIDIA Model Optimizer 0.48.0; numba; OpenCV.

## License
Apache License 2.0 (this repository includes code adapted from OpenPCDet, Apache-2.0, and the KITTI Python
evaluation of second.pytorch, MIT; see NOTICE). Trained weights, if released, are for non-commercial research use
because they derive from KITTI.
