"""Quantize / build TensorRT engines, evaluate deployed accuracy, and measure latency.

build : python deploy_run.py build --model M.pt --out DIR --tag T --int8 image,fusion   (empty --int8 = all FP16)
eval  : python deploy_run.py eval  --model M.pt --out DIR --tag T --split tune
bench : python deploy_run.py bench --model M.pt --out DIR --tag T [--torch fp32|fp16] --n 300
INT8 is applied per unit by quantizing only the ONNX nodes of that unit (Q/DQ inserted by NVIDIA ModelOpt,
entropy or max calibration on frames of train_sub); all other layers run in FP16 (strongly typed network).
"""
import argparse, json, os, re, time, platform, subprocess
import numpy as np
import torch
from fusion3d.data import KittiFusionDataset, collate
from fusion3d.evaluation import to_device, to_kitti_annos, evaluate_annos, summarize
from fusion3d.deploy import fold_bn, export_onnx, build_engine, Pipeline, PartA, PartB

UNIT_NODES = {'A': {'image': r'^/image/', 'fusion': r'^/lateral'},
              'B': {'lidar': r'^/backbone/', 'head': r'^/conv_(cls|box|dir_cls)/'}}


def gpu_state():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,clocks.sm,power.draw', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        return [x.strip() for x in out.split(',')]
    except Exception:
        return None


def wait_cool(limit=65, timeout_s=1800):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        s = gpu_state()
        if s is None or not s[0].isdigit() or int(s[0]) <= limit:
            return s
        time.sleep(15)
    return gpu_state()


def ds_of(root, split, use_image, training=False):
    return KittiFusionDataset(root, os.path.join(root, 'ImageSets', split + '.txt'), training=training, use_image=use_image)


@torch.no_grad()
def calib_arrays(det, root, n_img, n_bev, seed=0):
    ds = ds_of(root, 'train_sub', det.use_image)
    idx = np.random.RandomState(seed).permutation(len(ds))[:max(n_img, n_bev)]
    imgs, bevs = [], []
    pipe = Pipeline(det)
    for k, i in enumerate(idx):
        b = to_device(collate([ds[int(i)]]), 'cuda')
        if det.use_image and k < n_img:
            imgs.append(b['images'].cpu().numpy())
        if k < n_bev:
            points, uv, uniq, inv, coords = det.lidar_points(b['points'], b.get('uv'))
            pf = det.pillar.point_features(points, inv, uniq.numel(), coords)
            if det.use_image:
                lat = list(pipe.partA(b['images']))
                pf = det.fusion.fuse(det.fusion.sample(lat, uv, points[:, 0].long(), 1, b['images'].shape[-2:]), pf)
            from fusion3d.model import scatter_max
            bev = det.scatter_bev(scatter_max(pf, inv, uniq.numel()), uniq, 1)
            bevs.append(bev.cpu().numpy())
    out = {}
    if imgs:
        out['A'] = {'images': np.concatenate(imgs)}
    out['B'] = {'canvas': np.concatenate(bevs)}
    return out


def onnx_nodes(path):
    import onnx
    return [n.name for n in onnx.load(path, load_external_data=False).graph.node]


def quantize_part(fp32_onnx, out_onnx, calib, units_int8, part, method):
    """units_int8: units of this part to quantize; the rest of the part stays FP16."""
    from modelopt.onnx.quantization import quantize
    names = onnx_nodes(fp32_onnx)
    keep_fp = [u for u in UNIT_NODES[part] if u not in units_int8]
    excl = [n for n in names if any(re.search(UNIT_NODES[part][u], n) for u in keep_fp)]
    quantize(onnx_path=fp32_onnx, quantize_mode='int8', calibration_data=calib, calibration_method=method,
             nodes_to_exclude=excl if excl else None, output_path=out_onnx, high_precision_dtype='fp16',
             calibration_eps=['cpu'])
    return len(excl), len(names)


def fp16_part(fp32_onnx, out_onnx):
    from modelopt.onnx.autocast import convert_to_mixed_precision
    import onnx
    m = convert_to_mixed_precision(onnx_path=fp32_onnx, low_precision_type='fp16', keep_io_types=True)
    onnx.save(m, out_onnx)


def cmd_build(a):
    det = fold_bn(torch.load(a.model, map_location='cpu', weights_only=False)).cuda().eval()
    os.makedirs(a.out, exist_ok=True)
    pre = os.path.join(a.out, a.tag)
    paths = export_onnx(det, pre + '_fp32')
    units = [u for u in a.int8.split(',') if u]
    info = {'tag': a.tag, 'int8_units': units, 'model': a.model, 'calib_method': a.calib}
    calib = None
    if units:
        calib = calib_arrays(det, a.root, a.calib_n if det.use_image else 0, a.calib_bev)
    for part, p in paths.items():
        mine = [u for u in units if u in UNIT_NODES[part]]
        q = pre + f'_{part}.onnx'
        if mine:
            ne, nn_ = quantize_part(p, q, calib[part], mine, part, a.calib)
            info[f'{part}_excluded_nodes'] = ne; info[f'{part}_nodes'] = nn_
        else:
            fp16_part(p, q)
        info[f'{part}_build_s'] = build_engine(q, pre + f'_{part}.engine')
    info['host'] = platform.node(); info['trt'] = __import__('tensorrt').__version__
    json.dump(info, open(pre + '_build.json', 'w'), indent=1)
    print('BUILD', json.dumps(info))


def engines(a, det):
    pre = os.path.join(a.out, a.tag)
    ea = pre + '_A.engine' if det.use_image else None
    return ea, pre + '_B.engine'


@torch.no_grad()
def cmd_eval(a):
    det = fold_bn(torch.load(a.model, map_location='cpu', weights_only=False)).cuda().eval()
    ea, eb = engines(a, det) if a.tag != 'torch' else (None, None)
    pipe = Pipeline(det, ea, eb, torch_fp16=(a.torch == 'fp16'))
    ds = ds_of(a.root, a.split, det.use_image)
    annos = []
    for i in range(len(ds)):
        b = to_device(collate([ds[i]]), 'cuda')
        annos += to_kitti_annos(pipe(b), b)
    txt, d = evaluate_annos(annos, a.root, ds.ids)
    r = summarize(d)
    name = f"deploy_eval_{a.tag}{'_' + a.torch if a.tag == 'torch' else ''}_{a.split}.json"
    json.dump({'tag': a.tag, 'split': a.split, 'summary': r, 'full': {k: float(v) for k, v in d.items()}},
              open(os.path.join(a.out, name), 'w'), indent=1)
    print('EVAL', json.dumps(r))


@torch.no_grad()
def cmd_bench(a):
    det = fold_bn(torch.load(a.model, map_location='cpu', weights_only=False)).cuda().eval()
    ea, eb = engines(a, det) if a.tag != 'torch' else (None, None)
    pipe = Pipeline(det, ea, eb, torch_fp16=(a.torch == 'fp16'))
    ds = ds_of(a.root, 'tune', det.use_image)
    batches = [collate([ds[i]]) for i in range(min(a.n, len(ds)))]
    state_before = wait_cool()
    for b in batches[:a.warmup]:
        pipe(to_device(b, 'cuda'))
    torch.cuda.synchronize()
    timer, walls = [], []
    for b in batches:
        bb = to_device(b, 'cuda'); torch.cuda.synchronize()
        t0 = time.perf_counter()
        pipe(bb, timer=timer)
        torch.cuda.synchronize(); walls.append((time.perf_counter() - t0) * 1000)
    tot = np.array(walls)
    stages = {k: float(np.median([t[k] for t in timer])) for k in timer[0]}
    rec = {'tag': a.tag, 'torch_mode': a.torch, 'n': len(walls), 'warmup': a.warmup, 'median_ms': float(np.median(tot)),
           'p95_ms': float(np.percentile(tot, 95)), 'mean_ms': float(tot.mean()), 'stages_median_ms': stages,
           'gpu': torch.cuda.get_device_name(0), 'host': platform.node(), 'torch': torch.__version__,
           'peak_mem_mb': torch.cuda.max_memory_allocated() / 2**20,
           'gpu_state_before': state_before, 'gpu_state_after': gpu_state()}
    name = f"bench_{a.tag}{'_' + a.torch if a.tag == 'torch' else ''}_{a.device_label}.json"
    json.dump(rec, open(os.path.join(a.out, name), 'w'), indent=1)
    print('BENCH', json.dumps(rec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'eval', 'bench'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='fp16')
    ap.add_argument('--int8', default='')
    ap.add_argument('--calib', default='entropy', choices=['entropy', 'max'])
    ap.add_argument('--calib-n', type=int, default=256)
    ap.add_argument('--calib-bev', type=int, default=64)
    ap.add_argument('--split', default='tune')
    ap.add_argument('--torch', default='fp32', choices=['fp32', 'fp16'])
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--device-label', default='gb10')
    a = ap.parse_args()
    {'build': cmd_build, 'eval': cmd_eval, 'bench': cmd_bench}[a.cmd](a)


if __name__ == '__main__':
    main()
