"""Sanity checks run once on the target machine (results recorded in tests_report.json)."""
import os, sys, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from fusion3d.pcdet_lite import iou3d_nms_utils
from fusion3d.pcdet_lite.kitti_eval.rotate_iou_cpu import rotate_iou_cpu_eval
from fusion3d.data import KittiFusionDataset, collate, make_loader
from fusion3d.model import Detector
from fusion3d.evaluation import evaluate_annos, summarize, to_device
from fusion3d.pcdet_lite import box_utils

def main():
    root = sys.argv[1]
    rep = {}
    def log(*a):
        print(*a, flush=True)
    # 1) torch rotated BEV IoU vs the official-eval numba implementation
    rng = np.random.RandomState(0)
    N = 300
    b = np.concatenate([rng.uniform(-5, 5, (N, 2)), rng.uniform(-1, 1, (N, 1)), rng.uniform(0.3, 5, (N, 3)),
                        rng.uniform(-np.pi, np.pi, (N, 1))], 1).astype(np.float32)
    t = iou3d_nms_utils.boxes_iou_bev(torch.from_numpy(b).cuda(), torch.from_numpy(b).cuda()).cpu().numpy()
    # eval format: [x, y, w, l, angle] in a frame where the official code is used with lidar boxes as [x, y, dx, dy, -heading]
    be = b[:, [0, 1, 3, 4, 6]].copy(); be[:, 4] = -be[:, 4]   # official code rotates by -angle
    ref = rotate_iou_cpu_eval(be, be, -1)
    rep['iou_max_abs_diff'] = float(np.abs(t - ref).max())
    rep['iou_mean_abs_diff'] = float(np.abs(t - ref).mean())
    rep['iou_diag_min'] = float(np.diag(t).min())
    log('step1', rep)
    # 2) evaluation with ground truth as predictions -> AP should be ~100
    ds = KittiFusionDataset(root, os.path.join(root, 'ImageSets', 'tune.txt'), training=False, use_image=False)
    ids = ds.ids[:200]
    from fusion3d.pcdet_lite.kitti_eval import kitti_common
    gt = kitti_common.get_label_annos(os.path.join(root, 'training', 'label_2'), [int(i) for i in ids])
    dt = []
    for i, a in zip(ids, gt):
        a2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in a.items()}
        a2['score'] = np.random.RandomState(len(dt)).uniform(0.5, 1.0, len(a['name'])); a2['frame_id'] = i
        dt.append(a2)
    _, d = evaluate_annos(dt, root, ids)
    rep['gt_as_pred'] = summarize(d)
    log('step2', rep.get('gt_as_pred'))
    # 3) dataset + model forward/backward (fusion and lidar-only), timing
    for use_image in [1, 0]:
        ds = KittiFusionDataset(root, os.path.join(root, 'ImageSets', 'train_sub.txt'), training=True, use_image=bool(use_image))
        dl = make_loader(ds, 8, True, 8)
        m = Detector(use_image=bool(use_image)).cuda().train()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
        times = []
        for k, batch in enumerate(dl):
            if k == 12: break
            torch.cuda.synchronize(); t0 = time.time()
            batch = to_device(batch, 'cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(int(os.environ.get('AMP', '0')))):
                loss, tb = m(batch)
            opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize(); times.append(time.time() - t0)
        rep[f'train_iter_s_img{use_image}'] = float(np.median(times[2:]))
        rep[f'loss_img{use_image}'] = float(loss)
        rep[f'mem_gb_img{use_image}'] = torch.cuda.max_memory_allocated() / 1e9
        rep[f'params_M_img{use_image}'] = sum(p.numel() for p in m.parameters()) / 1e6
        m.eval()
        with torch.no_grad():
            preds = m(to_device(batch, 'cuda'))
        rep[f'n_pred_img{use_image}'] = [int(len(p['pred_boxes'])) for p in preds]
        del m, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    print(json.dumps(rep, indent=1))
    json.dump(rep, open('tests_report.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
