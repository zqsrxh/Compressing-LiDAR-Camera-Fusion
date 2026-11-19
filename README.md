# Where to compress a LiDAR-camera fusion detector

Code, split files, raw results and analysis scripts for the article
"Where to compress a LiDAR-camera fusion detector: unit-wise sensitivity, latency-budgeted pruning
and INT8 deployment on two GPU platforms" .

No KITTI data are redistributed. All derived files are regenerated from the official KITTI download
(KITTI is licensed CC BY-NC-SA 3.0; register and accept the terms at https://www.cvlibs.net/datasets/kitti/).

## Layout
| Path | Purpose |
|---|---|
| `code/fusion3d/model.py` | MVX-Net-style point-level fusion detector with a PointPillars LiDAR branch (pure PyTorch), modality dropout |
| `code/fusion3d/data.py` | KITTI loader; points keep their original pixel coordinates so 3D augmentation keeps the point-to-pixel match |
| `code/fusion3d/prune.py` | structured channel pruning per unit (image / fusion / LiDAR), L1 and FPGM criteria, global L1 |
| `code/fusion3d/deploy.py`, `code/deploy_run.py` | BN folding, ONNX export, FP16 and per-unit INT8 (NVIDIA ModelOpt Q/DQ) TensorRT engines, accuracy and latency |
| `code/fusion3d/pcdet_lite/` | pure-Python parts of OpenPCDet (Apache-2.0) with compiled CUDA ops replaced (see `code/NOTICE`) |
| `code/prepare.py` | field-of-view point cache and the drive-disjoint tuning split |
| `code/train.py`, `code/evaluate.py` | training and evaluation (KITTI R40) |
| `code/sensitivity.py`, `code/alloc.py`, `code/compress.py` | one-shot unit sensitivity, sensitivity-guided allocation (`--budget-mode predicted`), pruning |
| `code/latency_lut.py`, `code/combine_lut.py` | TensorRT latency lookup table used by the allocation |
| `code/splits/` | `train.txt`/`val.txt` (3712/3769) and `train_sub.txt`/`tune.txt` (3153/559) with MD5 checksums in `split_info.json` |
| `scripts/orchestration/tasks_full_E.tsv` | the exact command of every fusion, sensitivity, allocation and fine-tuning run |
| `scripts/orchestration/*.sh`, `orchestrator.py` | cloud-GPU queue, deployment, INT8 and benchmark scripts |
| `scripts/orchestration/make_tables*.py`, `scripts/make_paper_tables.py` | aggregation, paired t tests with Holm correction, LaTeX tables |
| `scripts/bench3080/` | RTX 3080 engine build (from the same ONNX graphs) and 3-pass benchmark |
| `results/runs/` | per-run training logs, tuning and validation evaluations, sensitivity tables, allocation, pruning profiles, deployed evaluations, engine build records, GB10 benchmarks |
| `results/lidar_runs/` | LiDAR-only runs and the GB10 latency lookup table |
| `results/tables/` | `summary.json`, `summary2.json`, `gb10x.json` (pinned GB10 latency), `bench3080_all.json`, `recipe_ablation.json` |
| `paper/` | LaTeX sources of the tables and figures as used in the article |

## Reproduction
1. Download KITTI 3D object (`data_object_velodyne`, `data_object_image_2`, `data_object_calib`, `data_object_label_2`)
   into `data/kitti/training/`, copy `code/splits/*.txt` to `data/kitti/ImageSets/`, then
   `python code/prepare.py --root data/kitti --fov-only`.
2. LiDAR-only model: `python code/train.py --root data/kitti --out runs/lidar_s0 --use-image 0 --seed 0`.
3. Fusion model (recipe used in the article): `python code/train.py --root data/kitti --out runs/fusion_r50_s0 --amp none --epochs 80 --lr 0.003 --sched onecycle --img-lr-mult 1.0 --img-init coco --freeze-img-backbone 1 --freeze-img-fpn 1 --img-dropout 0.5 --use-image 1 --img-depth 50 --seed 0`.
4. Sensitivity, allocation, pruning and fine-tuning: see `scripts/orchestration/tasks_full_E.tsv`
   (the allocation line was run with `--budgets 0.5 --budget-mode predicted`).
5. Deployment and latency: `scripts/orchestration/deploy_final.sh MODEL OUTDIR`, `bench_gb10_v2.sh`
   (GB10, pinned to the Cortex-X925 cores) and `scripts/bench3080/run_all.py` (RTX 3080).
6. Tables: `python scripts/orchestration/make_tables.py RUNS LIDAR_RUNS OUT` and `make_tables2.py`.
Every result JSON also records its full command-line arguments in its `args` field.

## Software
Training of fusion models and fine-tuning: PyTorch 2.4.1 (CUDA 12.4) on A100-SXM4-80GB.
LiDAR-only training, INT8 analysis and GB10 latency: PyTorch 2.14.1 (CUDA 13.0), TensorRT 11.4.0.106, NVIDIA Model Optimizer 0.48.0.
RTX 3080 latency: PyTorch 2.14.1 (CUDA 13.4), TensorRT 11.4.0.106.

## License
Code: Apache-2.0 (see `code/LICENSE` and `code/NOTICE`). Results and split files: CC BY 4.0.
