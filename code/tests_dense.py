"""Equivalence and TensorRT check for the dense point part (three-engine deployment)."""
import sys, os, json, time
import numpy as np
import torch
from fusion3d.model import Detector, scatter_max
from fusion3d.deploy import fold_bn, PartA, TRTModule
from fusion3d.dense import DensePointPart, export_point_part, build_point_engine
from fusion3d.data import KittiFusionDataset, collate
from fusion3d.evaluation import to_device


def ref_bev(det, b):
    points, uv, uniq, inv, coords = det.lidar_points(b['points'], b.get('uv'))
    pf = det.pillar.point_features(points, inv, uniq.numel(), coords)
    if det.use_image:
        lat = list(PartA(det)(b['images']))
        pf = det.fusion.fuse(det.fusion.sample(lat, uv, points[:, 0].long(), 1, b['images'].shape[-2:]), pf)
    return det.scatter_bev(scatter_max(pf, inv, uniq.numel()), uniq, 1)


def main():
    root = sys.argv[1]
    out = {}
    torch.manual_seed(0)
    for use_image in [True, False]:
        tag = f'img{int(use_image)}'
        det = fold_bn(Detector(use_image=use_image, pretrained=False)).cuda().eval()
        pp = DensePointPart(det).cuda().eval()
        ds = KittiFusionDataset(root, os.path.join(root, 'ImageSets', 'tune.txt'), training=False, use_image=use_image)
        p = export_point_part(det, f'point_{tag}.onnx')
        out[f'build_s_{tag}'] = build_point_engine(p, f'point_{tag}.engine', use_image=use_image)
        eng = TRTModule(f'point_{tag}.engine')
        A = PartA(det).eval() if use_image else None
        d_torch, d_trt, t_trt, npts = [], [], [], []
        for i in range(40):
            b = to_device(collate([ds[i]]), 'cuda')
            with torch.no_grad():
                ref = ref_bev(det, b)
                kw = {'points': b['points'][:, 1:5]}
                if use_image:
                    lat = A(b['images'])
                    kw['uv'] = b['uv']; kw.update({f'l{k}': lat[k] for k in range(5)})
                got = pp(**kw)
                scale = float(ref.abs().max()) + 1e-6
                d_torch.append(float((ref - got).abs().max()) / scale)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                res = eng(**kw)[0]
                torch.cuda.synchronize(); t_trt.append((time.perf_counter() - t0) * 1000)
                d_trt.append(float((ref - res).abs().max()) / scale)
                npts.append(int(b['points'].shape[0]))
        out[f'torch_dense_rel_maxdiff_{tag}'] = max(d_torch)
        out[f'trt_point_rel_maxdiff_{tag}'] = max(d_trt)
        out[f'trt_point_ms_median_{tag}'] = float(np.median(t_trt[5:]))
        out[f'n_points_range_{tag}'] = [min(npts), max(npts)]
        print(json.dumps(out), flush=True)
    json.dump(out, open('dense_test.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
