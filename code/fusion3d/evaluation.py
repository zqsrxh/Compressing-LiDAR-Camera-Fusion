"""Prediction export (KITTI format) and official-protocol evaluation (R40)."""
import os
import numpy as np
import torch
from .pcdet_lite import box_utils
from .pcdet_lite.kitti_eval import eval as kitti_eval
from .pcdet_lite.kitti_eval import kitti_common
from .data import CLASS_NAMES


def to_kitti_annos(preds, batch):
    annos = []
    for b, p in enumerate(preds):
        calib, img_shape = batch['calib'][b], batch['image_shape'][b]
        boxes = p['pred_boxes'].detach().float().cpu().numpy()
        scores = p['pred_scores'].detach().float().cpu().numpy()
        labels = p['pred_labels'].detach().cpu().numpy()
        n = len(boxes)
        a = {'name': np.zeros(n, dtype='<U10'), 'truncated': np.zeros(n), 'occluded': np.zeros(n),
             'alpha': np.zeros(n), 'bbox': np.zeros((n, 4)), 'dimensions': np.zeros((n, 3)),
             'location': np.zeros((n, 3)), 'rotation_y': np.zeros(n), 'score': np.zeros(n),
             'boxes_lidar': np.zeros((n, 7)), 'frame_id': batch['frame_id'][b]}
        if n:
            cam = box_utils.boxes3d_lidar_to_kitti_camera(boxes, calib)
            img = box_utils.boxes3d_kitti_camera_to_imageboxes(cam, calib, image_shape=img_shape)
            a.update(name=np.array(CLASS_NAMES)[labels - 1], alpha=-np.arctan2(-boxes[:, 1], boxes[:, 0]) + cam[:, 6],
                     bbox=img, dimensions=cam[:, 3:6], location=cam[:, 0:3], rotation_y=cam[:, 6], score=scores,
                     boxes_lidar=boxes)
        annos.append(a)
    return annos


def write_kitti_txt(annos, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for a in annos:
        with open(os.path.join(out_dir, a['frame_id'] + '.txt'), 'w') as f:
            for k in range(len(a['name'])):
                bb, d, l = a['bbox'][k], a['dimensions'][k], a['location'][k]
                f.write('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n' % (
                    a['name'][k], a['alpha'][k], bb[0], bb[1], bb[2], bb[3], d[1], d[2], d[0],
                    l[0], l[1], l[2], a['rotation_y'][k], a['score'][k]))


def evaluate_annos(dt_annos, root, frame_ids):
    gt = kitti_common.get_label_annos(os.path.join(root, 'training', 'label_2'), [int(i) for i in frame_ids])
    order = {a['frame_id']: a for a in dt_annos}
    dt = [order[i] for i in frame_ids]
    res = kitti_eval.get_official_eval_result(gt, dt, CLASS_NAMES)
    if isinstance(res, tuple):
        txt, d = res
    else:
        txt, d = res, {}
    return txt, d


def summarize(d):
    """Key numbers: AP3D R40 (easy/mod/hard) per class and the 3-class moderate mean."""
    out = {}
    for c in CLASS_NAMES:
        for diff in ['easy', 'moderate', 'hard']:
            for kind in ['3d', 'bev']:
                k = f'{c}_{kind}/{diff}_R40'
                if k in d:
                    out[k] = float(d[k])
    out['mAP3d_mod_R40'] = float(np.mean([out[f'{c}_3d/moderate_R40'] for c in CLASS_NAMES]))
    out['mAPbev_mod_R40'] = float(np.mean([out[f'{c}_bev/moderate_R40'] for c in CLASS_NAMES]))
    return out


@torch.no_grad()
def run_inference(model, loader, device, amp=None):
    model.eval()
    annos = []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp == 'bf16'):
            preds = model(batch)
        annos += to_kitti_annos(preds, batch)
    return annos


def to_device(batch, device):
    out = dict(batch)
    for k in ['points', 'images', 'uv', 'gt_boxes']:
        if k in out:
            out[k] = out[k].to(device, non_blocking=True)
    return out
