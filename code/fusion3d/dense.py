"""Single-graph deployment form of the detector (batch size 1).

Identical in function to Detector.forward_features + head convolutions, but every pillar-level reduction
scatters into the dense 496 x 432 pillar grid (plus one dummy row for out-of-range points) instead of a
compacted list of occupied pillars. This removes the data-dependent torch.unique step, so the whole network,
including the point-wise layers, can be exported to ONNX and compiled into one TensorRT engine with a dynamic
number of points. Empty pillars are zero in both forms, so outputs are the same (checked by tests_dense.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import GRID
from .deploy import PartA, PartB


class DenseDetector(nn.Module):
    def __init__(self, det):
        super().__init__()
        self.det = det
        self.use_image = det.use_image
        self.pillar = det.pillar
        if det.use_image:
            self.partA = PartA(det)
            self.fusion = det.fusion
        self.partB = PartB(det)
        self.nx, self.ny = int(GRID[0]), int(GRID[1])
        self.register_buffer('vsize', det.pillar.vsize.clone(), persistent=False)
        self.register_buffer('pcr', det.pillar.pcr.clone(), persistent=False)

    def forward(self, points, uv=None, images=None):
        # points: (N, 4) x, y, z, r in the LiDAR frame; uv: (N, 2) pixel coordinates in the padded image
        NP = self.nx * self.ny
        xyz = points[:, :3]
        cf = torch.floor((xyz - self.pcr[:3]) / self.vsize)
        inr = (cf[:, 0] >= 0) & (cf[:, 0] < self.nx) & (cf[:, 1] >= 0) & (cf[:, 1] < self.ny) & (cf[:, 2] >= 0) & (cf[:, 2] < 1)
        key = (cf[:, 1] * self.nx + cf[:, 0]).long()
        key = torch.where(inr, key, torch.full_like(key, NP))
        w = inr.to(xyz.dtype)
        sums = xyz.new_zeros((NP + 1, 3)).index_add(0, key, xyz * w[:, None])
        cnt = xyz.new_zeros((NP + 1,)).index_add(0, key, w)
        mean = sums / cnt.clamp(min=1)[:, None]
        f_cluster = xyz - mean[key]
        f_center = xyz - (cf * self.vsize + self.vsize / 2 + self.pcr[:3])
        f = torch.cat([points, f_cluster, f_center], 1)
        x1 = self.pillar.pfn1(f)
        m1 = x1.new_zeros((NP + 1, x1.shape[1])).scatter_reduce(0, key[:, None].expand(-1, x1.shape[1]), x1, reduce='amax', include_self=True)
        pf = self.pillar.pfn2(torch.cat([x1, m1[key]], 1))
        if self.use_image:
            lat = self.partA(images)
            H, W = images.shape[-2], images.shape[-1]
            g = torch.stack([uv[:, 0] / W * 2 - 1, uv[:, 1] / H * 2 - 1], -1).view(1, 1, -1, 2)
            per = [F.grid_sample(l, g, mode='bilinear', padding_mode='zeros', align_corners=False).view(l.shape[1], -1).t()
                   for l in lat]
            pf = self.fusion.fuse(torch.cat(per, 1), pf)
        pf = pf * w[:, None]
        vox = pf.new_zeros((NP + 1, pf.shape[1])).scatter_reduce(0, key[:, None].expand(-1, pf.shape[1]), pf, reduce='amax', include_self=True)
        bev = vox[:NP].t().reshape(1, pf.shape[1], self.ny, self.nx)
        return self.partB(bev)


class DensePointPart(nn.Module):
    """Point-wise part between the image engine and the BEV engine, in dense form (exported as its own engine).
    Inputs: points (N, 4), uv (N, 2) and the five lateral maps from the image engine; output: BEV pseudo-image."""
    def __init__(self, det, img_hw=(384, 1280)):
        super().__init__()
        self.use_image = det.use_image
        self.pillar = det.pillar
        if det.use_image:
            self.fusion = det.fusion
        self.nx, self.ny = int(GRID[0]), int(GRID[1])
        self.H, self.W = img_hw
        self.register_buffer('vsize', det.pillar.vsize.clone(), persistent=False)
        self.register_buffer('pcr', det.pillar.pcr.clone(), persistent=False)

    def forward(self, points, uv=None, l0=None, l1=None, l2=None, l3=None, l4=None):
        NP = self.nx * self.ny
        xyz = points[:, :3]
        cf = torch.floor((xyz - self.pcr[:3]) / self.vsize)
        inr = (cf[:, 0] >= 0) & (cf[:, 0] < self.nx) & (cf[:, 1] >= 0) & (cf[:, 1] < self.ny) & (cf[:, 2] >= 0) & (cf[:, 2] < 1)
        key = (cf[:, 1] * self.nx + cf[:, 0]).long()
        key = torch.where(inr, key, torch.full_like(key, NP))
        w = inr.to(xyz.dtype)
        sums = xyz.new_zeros((NP + 1, 3)).index_add(0, key, xyz * w[:, None])
        cnt = xyz.new_zeros((NP + 1,)).index_add(0, key, w)
        mean = sums / cnt.clamp(min=1)[:, None]
        f = torch.cat([points, xyz - mean[key], xyz - (cf * self.vsize + self.vsize / 2 + self.pcr[:3])], 1)
        x1 = self.pillar.pfn1(f)
        m1 = x1.new_zeros((NP + 1, x1.shape[1])).scatter_reduce(0, key[:, None].expand(-1, x1.shape[1]), x1, reduce='amax', include_self=True)
        pf = self.pillar.pfn2(torch.cat([x1, m1[key]], 1))
        if self.use_image:
            g = torch.stack([uv[:, 0] / self.W * 2 - 1, uv[:, 1] / self.H * 2 - 1], -1).view(1, 1, -1, 2)
            per = [F.grid_sample(l, g, mode='bilinear', padding_mode='zeros', align_corners=False).view(l.shape[1], -1).t()
                   for l in (l0, l1, l2, l3, l4)]
            pf = self.fusion.fuse(torch.cat(per, 1), pf)
        pf = pf * w[:, None]
        vox = pf.new_zeros((NP + 1, pf.shape[1])).scatter_reduce(0, key[:, None].expand(-1, pf.shape[1]), pf, reduce='amax', include_self=True)
        return vox[:NP].t().reshape(1, pf.shape[1], self.ny, self.nx)


def export_point_part(det, path, n_opt=20000):
    pp = DensePointPart(det).eval().cuda()
    pts = torch.zeros(n_opt, 4, device='cuda')
    if det.use_image:
        with torch.no_grad():
            lat = PartA(det).eval()(torch.zeros(1, 3, 384, 1280, device='cuda'))
        args = (pts, torch.zeros(n_opt, 2, device='cuda')) + tuple(lat)
        names = ['points', 'uv'] + [f'l{i}' for i in range(5)]
        dyn = {'points': {0: 'n'}, 'uv': {0: 'n'}}
    else:
        args, names, dyn = (pts,), ['points'], {'points': {0: 'n'}}
    torch.onnx.export(pp, args, path, input_names=names, output_names=['canvas'], dynamic_axes=dyn, opset_version=18, dynamo=False)
    return path


def build_point_engine(onnx_path, engine_path, n_min=1000, n_opt=20000, n_max=80000, use_image=True):
    import tensorrt as trt, time
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')          # ScatterElements with max reduction uses a standard plugin
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(open(onnx_path, 'rb').read()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    prof = builder.create_optimization_profile()
    prof.set_shape('points', (n_min, 4), (n_opt, 4), (n_max, 4))
    if use_image:
        prof.set_shape('uv', (n_min, 2), (n_opt, 2), (n_max, 2))
    cfg.add_optimization_profile(prof)
    t0 = time.time(); ser = builder.build_serialized_network(net, cfg)
    if ser is None:
        raise RuntimeError('point engine build failed')
    open(engine_path, 'wb').write(ser)
    return time.time() - t0
