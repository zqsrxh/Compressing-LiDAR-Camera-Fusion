#!/bin/bash
# One-time pod setup: python deps, KITTI 3D object download, FOV cache. Idempotent.
set -e
command -v unzip >/dev/null || (apt-get update -qq && apt-get install -y -qq unzip)
B=${B:-/workspace/fusion3d}
cd $B
python -c "import torch, torchvision; print('torch', torch.__version__, torchvision.__version__, torch.cuda.device_count())"
pip install -q numba opencv-python-headless pyyaml scipy 2>&1 | tail -1
python - <<'EOF'
import torchvision
torchvision.models.resnet50(weights='IMAGENET1K_V1'); torchvision.models.resnet18(weights='IMAGENET1K_V1')
print('imagenet weights cached')
EOF
mkdir -p $B/data/kitti/zips && cd $B/data/kitti
for f in data_object_calib.zip data_object_label_2.zip data_object_image_2.zip data_object_velodyne.zip; do
  if [ ! -f $f.done ]; then
    wget -q -c -O zips/$f https://s3.eu-central-1.amazonaws.com/avg-kitti/$f
    unzip -q -o zips/$f 'training/*' -d . && rm -f zips/$f && touch $f.done
  fi
done
mkdir -p ImageSets && tar xzf $B/imagesets.tgz --no-same-owner -C ImageSets
(cd ImageSets && md5sum -c $B/imagesets.md5)
ls training/velodyne | wc -l; ls training/image_2 | wc -l
cd $B/code && python prepare.py --root $B/data/kitti --fov-only
rm -rf $B/data/kitti/training/velodyne        # FOV cache (velodyne_fov) is all the code reads
df -h /workspace | tail -1
touch $B/SETUP_DONE
echo SETUP_DONE
