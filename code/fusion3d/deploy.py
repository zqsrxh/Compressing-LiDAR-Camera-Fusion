"""Deployment: BN folding, ONNX export of the two dense sub-graphs, TensorRT engines, timed pipeline.

Sub-graph A (fusion models only): image -> ResNet/FPN -> five fusion lateral convs  (input 'images')
Sub-graph B: BEV pseudo-image -> 2D backbone -> head 1x1 convs                        (input 'canvas')
Point-wise operations (pillar feature net, bilinear sampling, fusion projections, scatter) and box
decoding/NMS run in PyTorch on the GPU between the engines.
"""
import copy, time
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models.resnet import Bottleneck, BasicBlock

from .model import scatter_max


# ----------------------------------------------------------------------------- BN folding
def _bn_params(bn):
    eps = getattr(bn, 'eps', 1e-5)
    scale = bn.weight / torch.sqrt(bn.running_var + eps)
    shift = bn.bias - bn.running_mean * scale
    return scale.detach(), shift.detach()


@torch.no_grad()
def _fold(conv, bn):
    s, t = _bn_params(bn)
    w = conv.weight.data
    if isinstance(conv, nn.ConvTranspose2d):
        conv.weight.data = w * s.view(1, -1, 1, 1)
    elif isinstance(conv, nn.Conv2d):
        conv.weight.data = w * s.view(-1, 1, 1, 1)
    else:  # Linear
        conv.weight.data = w * s.view(-1, 1)
    b = conv.bias.data if conv.bias is not None else torch.zeros_like(s)
    conv.bias = nn.Parameter(b * s + t)


def is_bn(m):
    return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, FrozenBatchNorm2d))


@torch.no_grad()
def fold_bn(model):
    model = copy.deepcopy(model).eval()
    for m in model.modules():
        if isinstance(m, (Bottleneck, BasicBlock)):
            for i in (1, 2, 3):
                if hasattr(m, f'conv{i}') and hasattr(m, f'bn{i}') and is_bn(getattr(m, f'bn{i}')):
                    _fold(getattr(m, f'conv{i}'), getattr(m, f'bn{i}')); setattr(m, f'bn{i}', nn.Identity())
            if m.downsample is not None and is_bn(m.downsample[1]):
                _fold(m.downsample[0], m.downsample[1]); m.downsample[1] = nn.Identity()
        if isinstance(m, nn.Sequential):
            mods = list(m)
            for i in range(len(mods) - 1):
                if isinstance(mods[i], (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) and is_bn(mods[i + 1]):
                    _fold(mods[i], mods[i + 1]); m[i + 1] = nn.Identity()
    # PFN layers (Linear + BN1d)
    for pfn in [model.pillar.pfn1, model.pillar.pfn2]:
        if is_bn(pfn.norm):
            _fold(pfn.linear, pfn.norm); pfn.norm = nn.Identity()
    return model


# ----------------------------------------------------------------------------- sub-graphs
class PartA(nn.Module):
    def __init__(self, det):
        super().__init__()
        self.image, self.lateral = det.image, det.fusion.lateral_convs

    def forward(self, images):
        feats = self.image(images)
        return tuple(conv(f) for conv, f in zip(self.lateral, feats))


class PartB(nn.Module):
    def __init__(self, det):
        super().__init__()
        self.backbone, self.head = det.backbone, det.head

    def forward(self, canvas):
        x = self.backbone({'spatial_features': canvas})['spatial_features_2d']
        return self.head.conv_cls(x), self.head.conv_box(x), self.head.conv_dir_cls(x)


def export_onnx(det, out_prefix, img_hw=(384, 1280)):
    det = det.eval().cuda()
    paths = {}
    if det.use_image:
        a = PartA(det).eval()
        x = torch.zeros(1, 3, *img_hw, device='cuda')
        p = out_prefix + '_A.onnx'
        torch.onnx.export(a, (x,), p, input_names=['images'], output_names=[f'lat{i}' for i in range(5)],
                          opset_version=17, dynamo=False)
        paths['A'] = p
    b = PartB(det).eval()
    from .model import GRID
    c = torch.zeros(1, det.backbone.blocks[0][1].in_channels, int(GRID[1]), int(GRID[0]), device='cuda')
    p = out_prefix + '_B.onnx'
    torch.onnx.export(b, (c,), p, input_names=['canvas'], output_names=['cls', 'box', 'dir'], opset_version=17, dynamo=False)
    paths['B'] = p
    return paths


# ----------------------------------------------------------------------------- TensorRT
def build_engine(onnx_path, engine_path, workspace_gb=8):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, logger)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    t0 = time.time()
    ser = builder.build_serialized_network(net, cfg)
    if ser is None:
        raise RuntimeError('engine build failed: ' + onnx_path)
    with open(engine_path, 'wb') as f:
        f.write(ser)
    return time.time() - t0


class TRTModule:
    def __init__(self, engine_path):
        import tensorrt as trt
        self.trt = trt
        lg = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(lg, '')
        rt = trt.Runtime(lg)
        self.engine = rt.deserialize_cuda_engine(open(engine_path, 'rb').read())
        self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self._dt = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
                    trt.bfloat16: torch.bfloat16, trt.int8: torch.int8}
        self.in_dtype = {n: self._dt[self.engine.get_tensor_dtype(n)] for n in self.inputs}
        self._out = {}

    def __call__(self, **inputs):
        keep = []
        for n in self.inputs:
            t = inputs[n].to(self.in_dtype[n]).contiguous(); keep.append(t)
            self.ctx.set_input_shape(n, tuple(t.shape)); self.ctx.set_tensor_address(n, t.data_ptr())
        outs = []
        for n in self.outputs:
            shape = tuple(self.ctx.get_tensor_shape(n))
            key = (n, shape)
            if key not in self._out:
                self._out[key] = torch.empty(shape, dtype=self._dt[self.engine.get_tensor_dtype(n)], device='cuda')
            t = self._out[key]
            self.ctx.set_tensor_address(n, t.data_ptr()); outs.append(t)
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return [o.float() for o in outs]


# ----------------------------------------------------------------------------- pipeline
class Pipeline:
    """Runs a (BN-folded) detector with optional TensorRT engines for sub-graphs A and B.
    mode: 'torch' (eager, fp32 or fp16 autocast) or 'trt'. Batch size 1."""
    def __init__(self, det, engine_a=None, engine_b=None, torch_fp16=False):
        self.det = det.eval().cuda()
        self.A = TRTModule(engine_a) if engine_a else None
        self.B = TRTModule(engine_b) if engine_b else None
        self.fp16 = torch_fp16
        if not self.A and det.use_image:
            self.partA = PartA(self.det)
        if not self.B:
            self.partB = PartB(self.det)

    @torch.no_grad()
    def __call__(self, batch, timer=None):
        det = self.det
        ev = [] if timer is not None else None
        def mark():
            if ev is not None:
                e = torch.cuda.Event(enable_timing=True); e.record(); ev.append(e)
        mark()
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.fp16):
            points, uv, uniq, inv, coords = det.lidar_points(batch['points'], batch.get('uv'))
            pf = det.pillar.point_features(points, inv, uniq.numel(), coords)
            mark()
            if det.use_image:
                lat = self.A(images=batch['images']) if self.A else list(self.partA(batch['images']))
                mark()
                img_pts = det.fusion.sample(lat, uv, points[:, 0].long(), 1, batch['images'].shape[-2:])
                pf = det.fusion.fuse(img_pts.to(pf.dtype), pf)
            else:
                mark()
            vox = scatter_max(pf, inv, uniq.numel())
            bev = det.scatter_bev(vox, uniq, 1)
            mark()
            if self.B:
                cls, box, dirc = self.B(canvas=bev)
            else:
                cls, box, dirc = [t.float() for t in self.partB(bev)]
            mark()
        h = det.head
        if h.anchors[0].device != cls.device:
            h.anchors = [a.to(cls.device) for a in h.anchors]
        bcls, bbox = h.generate_predicted_boxes(1, cls.permute(0, 2, 3, 1).contiguous(), box.permute(0, 2, 3, 1).contiguous(),
                                                dirc.permute(0, 2, 3, 1).contiguous())
        preds = det.post_process({'batch_size': 1, 'batch_cls_preds': bcls, 'batch_box_preds': bbox})
        mark()
        if ev is not None:
            torch.cuda.synchronize()
            names = ['pillar', 'imageA', 'fuse_scatter', 'bevB', 'post']
            timer.append({n: ev[i].elapsed_time(ev[i + 1]) for i, n in enumerate(names)})
        return preds
