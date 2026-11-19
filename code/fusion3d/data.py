"""KITTI 3D object dataset for LiDAR-camera fusion.

Every LiDAR point keeps the pixel coordinate (u, v) it projects to in the ORIGINAL image.
3D augmentation (flip, rotation, scaling) changes only x, y, z and the boxes, so the
point-to-pixel correspondence stays exact while the image itself is left unaugmented.
"""
import os
import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
from torch.utils.data import Dataset

from .pcdet_lite import box_utils, calibration_kitti, object3d_kitti, augmentor_utils, common_utils

CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']
PC_RANGE = np.array([0, -39.68, -3, 69.12, 39.68, 1], dtype=np.float32)
IMG_W, IMG_H = 1280, 384              # padded network input size (divisible by 32)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255


def read_split(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


class KittiFusionDataset(Dataset):
    def __init__(self, root, split_file, training, use_image=True, class_names=CLASS_NAMES):
        self.root = root
        self.ids = read_split(split_file)
        self.training = training
        self.use_image = use_image
        self.class_names = class_names
        self.d = os.path.join(root, 'training')

    def __len__(self):
        return len(self.ids)

    # ------------------------------------------------------------------ io
    def get_calib(self, idx):
        return calibration_kitti.Calibration(os.path.join(self.d, 'calib', idx + '.txt'))

    def get_points(self, idx):
        return np.fromfile(os.path.join(self.d, 'velodyne', idx + '.bin'), dtype=np.float32).reshape(-1, 4)

    def get_image(self, idx):
        img = cv2.imread(os.path.join(self.d, 'image_2', idx + '.png'))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_labels(self, idx):
        return object3d_kitti.get_objects_from_label(os.path.join(self.d, 'label_2', idx + '.txt'))

    # ------------------------------------------------------------------ main
    def __getitem__(self, i):
        idx = self.ids[i]
        calib = self.get_calib(idx)
        img = self.get_image(idx)
        H0, W0 = img.shape[:2]
        cache = os.path.join(self.d, 'velodyne_fov', idx + '.bin')
        if os.path.exists(cache):                      # written by prepare.py (same FOV rule as below)
            a = np.fromfile(cache, dtype=np.float32).reshape(-1, 6)
            pts, uv = a[:, :4].copy(), a[:, 4:6].copy()
        else:
            pts = self.get_points(idx)
            # field-of-view filter (as OpenPCDet FOV_POINTS_ONLY) and pixel coordinates
            pts_rect = calib.lidar_to_rect(pts[:, :3])
            uv, depth = calib.rect_to_img(pts_rect)
            fov = (uv[:, 0] >= 0) & (uv[:, 0] < W0) & (uv[:, 1] >= 0) & (uv[:, 1] < H0) & (depth >= 0)
            pts, uv = pts[fov], uv[fov].astype(np.float32)

        objs = [o for o in self.get_labels(idx) if o.cls_type in self.class_names]
        if len(objs):
            loc = np.array([o.loc for o in objs], dtype=np.float32)
            dims = np.array([[o.l, o.h, o.w] for o in objs], dtype=np.float32)
            ry = np.array([o.ry for o in objs], dtype=np.float32)
            boxes_cam = np.concatenate([loc, dims, ry[:, None]], 1)
            gt_boxes = box_utils.boxes3d_kitti_camera_to_lidar(boxes_cam, calib).astype(np.float32)
            gt_cls = np.array([self.class_names.index(o.cls_type) + 1 for o in objs], dtype=np.int64)
        else:
            gt_boxes = np.zeros((0, 7), np.float32)
            gt_cls = np.zeros((0,), np.int64)

        if self.training:
            gt_boxes, pts3 = augmentor_utils.random_flip_along_x(gt_boxes, pts[:, :3].copy())
            gt_boxes, pts3 = augmentor_utils.global_rotation(gt_boxes, pts3, rot_range=[-0.78539816, 0.78539816])
            gt_boxes, pts3 = augmentor_utils.global_scaling(gt_boxes, pts3, [0.95, 1.05])
            pts = np.concatenate([pts3, pts[:, 3:4]], 1).astype(np.float32)

        # range filtering (points and boxes), as OpenPCDet mask_points_and_boxes_outside_range
        m = common_utils.mask_points_by_range(pts, PC_RANGE)
        pts, uv = pts[m], uv[m]
        if self.training and len(gt_boxes):
            keep = box_utils.mask_boxes_outside_range_numpy(gt_boxes, PC_RANGE, min_num_corners=1)
            gt_boxes, gt_cls = gt_boxes[keep], gt_cls[keep]
            perm = np.random.permutation(len(pts))
            pts, uv = pts[perm], uv[perm]
        if len(gt_boxes):
            gt_boxes[:, 6] = common_utils.limit_period(gt_boxes[:, 6], offset=0.5, period=2 * np.pi)

        out = {'frame_id': idx, 'points': pts, 'gt_boxes': np.concatenate([gt_boxes, gt_cls[:, None].astype(np.float32)], 1),
               'image_shape': np.array([H0, W0], np.int32), 'calib': calib}
        if self.use_image:
            s = min(IMG_W / W0, IMG_H / H0)
            nw, nh = int(round(W0 * s)), int(round(H0 * s))
            im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            im = (im - MEAN) / STD
            canvas = np.zeros((IMG_H, IMG_W, 3), np.float32)
            canvas[:nh, :nw] = im
            out['image'] = canvas.transpose(2, 0, 1)
            out['uv'] = uv * np.array([nw / W0, nh / H0], np.float32)
        return out


def collate(batch):
    B = len(batch)
    pts = [torch.from_numpy(np.concatenate([np.full((len(b['points']), 1), i, np.float32), b['points']], 1)) for i, b in enumerate(batch)]
    M = max(1, max(len(b['gt_boxes']) for b in batch))
    gt = torch.zeros(B, M, 8)
    for i, b in enumerate(batch):
        if len(b['gt_boxes']):
            gt[i, :len(b['gt_boxes'])] = torch.from_numpy(b['gt_boxes'])
    out = {'batch_size': B, 'points': torch.cat(pts, 0), 'gt_boxes': gt,
           'frame_id': [b['frame_id'] for b in batch], 'calib': [b['calib'] for b in batch],
           'image_shape': np.stack([b['image_shape'] for b in batch])}
    if 'image' in batch[0]:
        out['images'] = torch.from_numpy(np.stack([b['image'] for b in batch]))
        out['uv'] = torch.cat([torch.from_numpy(b['uv']) for b in batch], 0)
    return out


def worker_init(worker_id):
    """Seed numpy per worker from torch's per-worker seed (deterministic given the loader generator)."""
    np.random.seed(torch.initial_seed() % 2 ** 32)


def make_loader(ds, batch_size, shuffle, workers, seed=0, drop_last=False, persistent=False):
    from torch.utils.data import DataLoader
    g = torch.Generator(); g.manual_seed(seed)
    kw = dict(num_workers=workers, collate_fn=collate, worker_init_fn=worker_init, generator=g, pin_memory=True)
    if workers > 0:
        kw.update(multiprocessing_context='forkserver', persistent_workers=persistent)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kw)
