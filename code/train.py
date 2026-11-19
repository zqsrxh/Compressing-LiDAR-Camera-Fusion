"""Train (or fine-tune) the fusion / LiDAR-only detector on KITTI.

Schedule follows the OpenPCDet PointPillars KITTI config: 80 epochs, Adam one-cycle,
max lr 0.003, div factor 10, pct_start 0.4, decoupled weight decay 0.01, grad clip 10.
The final-epoch weights are used (no checkpoint selection on any evaluation split).
"""
import argparse, json, math, os, time, random, threading, glob, subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader

from fusion3d.data import KittiFusionDataset, collate, make_loader
from fusion3d.model import Detector
from fusion3d.evaluation import run_inference, evaluate_annos, summarize, to_device, write_kitti_txt


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def drop_page_cache(paths, stop):
    """GB10 unified memory: keep page cache from eating CUDA-usable memory (see host notes)."""
    while not stop.is_set():
        for p in paths:
            try:
                fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
            except Exception:
                pass
        stop.wait(60)


def gpu_temp():
    try:
        return int(subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0])
    except Exception:
        return -1


def thermal_guard(log, hi=int(os.environ.get('THERMAL_HI', 85)), lo=int(os.environ.get('THERMAL_LO', 75))):
    """Pause training while the GPU is hot (the GB10 host shut down once under sustained load)."""
    t = gpu_temp()
    if t < hi:
        return 0
    t0 = time.time()
    while gpu_temp() > lo:
        time.sleep(20)
    dt = time.time() - t0
    log.write(json.dumps({'thermal_pause_s': dt, 'temp_at_pause': t}) + '\n'); log.flush()
    return dt


def load_coco_image_branch(model):
    """Copy ResNet-50 + FPN weights of torchvision's COCO Faster R-CNN into the image branch (same module layout)."""
    import torchvision
    det = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1')
    src = det.state_dict(); dst = model.image.state_dict(); new = {}
    for k, v in src.items():
        if k.startswith('backbone.body.'):
            kk = k[len('backbone.body.'):]
            if kk.startswith('conv1.'): kk = 'stem.0.' + kk[len('conv1.'):]
            elif kk.startswith('bn1.'): kk = 'stem.1.' + kk[len('bn1.'):]
        elif k.startswith('backbone.fpn.'):
            kk = 'fpn.' + k[len('backbone.fpn.'):]
        else:
            continue
        if kk in dst and dst[kk].shape == v.shape:
            new[kk] = v
    missing = [k for k in dst if k not in new]
    if missing:
        raise RuntimeError(f'COCO init: {len(missing)} image-branch tensors not covered, e.g. {missing[:5]}')
    model.image.load_state_dict(new)
    return len(new)


def build_model(args):
    m = Detector(use_image=args.use_image, img_depth=args.img_depth, pretrained=not args.no_pretrained)
    if args.use_image and getattr(args, 'img_init', 'imagenet') == 'coco':
        assert args.img_depth == 50, 'COCO initialisation is available for ResNet-50 only'
        n = load_coco_image_branch(m); print('COCO image-branch init: tensors', n, flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--train-split', default='train_sub')
    ap.add_argument('--eval-split', default='tune')
    ap.add_argument('--use-image', type=int, default=1)
    ap.add_argument('--img-depth', type=int, default=50)
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.003)
    ap.add_argument('--img-lr-mult', type=float, default=1.0, help='learning-rate multiplier for the pretrained image branch')
    ap.add_argument('--sched', default='onecycle', choices=['onecycle', 'cosine'],
                    help='onecycle: OpenPCDet PointPillars schedule; cosine: MMDetection3D MVX-Net schedule (linear warm-up, cosine decay)')
    ap.add_argument('--warmup-iters', type=int, default=1000)
    ap.add_argument('--freeze-img-backbone', type=int, default=0,
                    help='1: keep ResNet stages frozen (only FPN, fusion, LiDAR and head train); 0: stages 2-4 trainable')
    ap.add_argument('--img-dropout', type=float, default=0.0, help='modality dropout probability for the image term')
    ap.add_argument('--img-init', default='imagenet', choices=['imagenet', 'coco'],
                    help='coco: ResNet-50 + FPN from torchvision Faster R-CNN trained on COCO 2D detection')
    ap.add_argument('--freeze-img-fpn', type=int, default=0, help='1: FPN frozen as well')
    ap.add_argument('--clip', type=float, default=10.0)
    ap.add_argument('--max-skips', type=int, default=20, help='abort after this many skipped non-finite batches')
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--amp', default='none', choices=['none', 'bf16'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--init', default='', help='checkpoint (full pickled model) to fine-tune, e.g. a pruned model')
    ap.add_argument('--max-iters', type=int, default=0, help='debug: stop after N iterations')
    ap.add_argument('--eval-every', type=int, default=0)
    ap.add_argument('--ckpt-every', type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = True        # fixed input sizes (image 1280x384, BEV 496x432)
    dev = torch.device('cuda')
    S = os.path.join(args.root, 'ImageSets')
    tr = KittiFusionDataset(args.root, os.path.join(S, args.train_split + '.txt'), training=True, use_image=bool(args.use_image))
    ev = KittiFusionDataset(args.root, os.path.join(S, args.eval_split + '.txt'), training=False, use_image=bool(args.use_image))
    tl = make_loader(tr, args.bs, True, args.workers, seed=args.seed, drop_last=True, persistent=True)
    el = make_loader(ev, 4, False, args.workers)
    if args.init:
        model = torch.load(args.init, map_location='cpu', weights_only=False)
    else:
        model = build_model(args)
    model.to(dev)
    if getattr(model, 'use_image', False):
        model.img_drop_p = args.img_dropout
        for part in [model.image.layer2, model.image.layer3, model.image.layer4]:
            for p_ in part.parameters():
                p_.requires_grad_(not args.freeze_img_backbone)
        for p_ in list(model.image.stem.parameters()) + list(model.image.layer1.parameters()):
            p_.requires_grad_(False)                  # frozen_stages=1 as in MVX-Net
        for p_ in model.image.fpn.parameters():
            p_.requires_grad_(not args.freeze_img_fpn)
    img_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('image.')]
    rest_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('image.')]
    groups = [{'params': rest_p, 'lr': args.lr}] + ([{'params': img_p, 'lr': args.lr * args.img_lr_mult}] if img_p else [])
    max_lrs = [g['lr'] for g in groups]
    params = rest_p + img_p
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.95, 0.99), weight_decay=args.wd)
    total = args.epochs * len(tl)
    if args.sched == 'onecycle':
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lrs, total_steps=total, pct_start=0.4,
                                                    anneal_strategy='cos', div_factor=10, final_div_factor=1e4,
                                                    cycle_momentum=True, base_momentum=0.85, max_momentum=0.95)
    else:   # MMDetection3D cosine.py: LinearLR(start 0.1, 1000 iters) combined with CosineAnnealing to 1e-5 over all epochs
        eta = 1e-5 / 0.003
        def lam(i):
            w = 0.1 + 0.9 * min(1.0, i / max(1, args.warmup_iters))
            return w * (eta + (1 - eta) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / total))))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    stop = threading.Event()
    files = glob.glob(os.path.join(args.root, 'training', '*', '*'))
    threading.Thread(target=drop_page_cache, args=(files, stop), daemon=True).start()
    log = open(os.path.join(args.out, 'train_log.jsonl'), 'a')
    it, t0, start_ep, skips = 0, time.time(), 0, 0
    ck = os.path.join(args.out, 'checkpoint.pt')
    if os.path.exists(ck):                        # resume after an interruption (recorded in the log)
        st = torch.load(ck, map_location='cpu', weights_only=False)
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
        start_ep, it = st['epoch'], st['iters']
        random.setstate(st['py_rng']); np.random.set_state(st['np_rng']); torch.set_rng_state(st['torch_rng'])
        torch.cuda.set_rng_state(st['cuda_rng'])
        tl = make_loader(tr, args.bs, True, args.workers, seed=args.seed * 1000 + start_ep, drop_last=True, persistent=True)
        log.write(json.dumps({'resumed_at_epoch': start_ep}) + '\n'); log.flush()
        print('RESUMED', start_ep, flush=True)
    for ep in range(start_ep, args.epochs):
        model.train()
        te = time.time(); losses = []; gns = []
        for batch in tl:
            batch = to_device(batch, dev)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.amp == 'bf16'):
                loss, tb = model(batch)
            opt.zero_grad(set_to_none=True)
            lv = float(loss.detach())
            if not np.isfinite(lv):                # skip the update (logged); abort if it keeps happening
                skips += 1; sched.step(); it += 1
                log.write(json.dumps({'skipped_nonfinite_iter': it, 'n_skipped': skips}) + '\n'); log.flush()
                if skips > args.max_skips:
                    raise RuntimeError(f'non-finite loss at iter {it} ({skips} skipped batches)')
                continue
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
            if not np.isfinite(gn):
                opt.zero_grad(set_to_none=True); skips += 1; sched.step(); it += 1
                log.write(json.dumps({'skipped_nonfinite_grad_iter': it, 'n_skipped': skips}) + '\n'); log.flush()
                if skips > args.max_skips:
                    raise RuntimeError(f'non-finite gradient at iter {it} ({skips} skipped batches)')
                continue
            opt.step(); sched.step(); it += 1
            losses.append(lv); gns.append(gn)
            if it % int(os.environ.get('THERMAL_EVERY', 50)) == 0:
                thermal_guard(log)
            if args.max_iters and it >= args.max_iters:
                break
        if args.ckpt_every and (ep + 1) % args.ckpt_every == 0 and ep + 1 < args.epochs:
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': ep + 1,
                        'iters': it, 'py_rng': random.getstate(), 'np_rng': np.random.get_state(),
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}, ck + '.tmp')
            os.replace(ck + '.tmp', ck)
        rec = {'epoch': ep + 1, 'iters': it, 'loss': float(np.mean(losses)), 'lr': sched.get_last_lr()[0], 'gpu_temp': gpu_temp(),
               'epoch_s': time.time() - te, 'elapsed_s': time.time() - t0,
               'max_mem_gb': torch.cuda.max_memory_allocated() / 1e9, 'grad_norm_mean': float(np.mean(gns)) if gns else None,
               'grad_norm_max': float(np.max(gns)) if gns else None, 'n_skipped_total': skips}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
        if args.max_iters and it >= args.max_iters:
            break
        if args.eval_every and (ep + 1) % args.eval_every == 0 and ep + 1 < args.epochs:
            annos = run_inference(model, el, dev, None)
            _, d = evaluate_annos(annos, args.root, ev.ids)
            r = summarize(d); r['epoch'] = ep + 1
            print('EVAL', json.dumps(r), flush=True); log.write(json.dumps({'eval': r}) + '\n'); log.flush()
    stop.set()
    if os.path.exists(ck):
        os.remove(ck)
    model.cpu()
    torch.save(model, os.path.join(args.out, 'model_final.pt'))
    torch.save(model.state_dict(), os.path.join(args.out, 'state_final.pth'))
    model.to(dev)
    annos = run_inference(model, el, dev, None)          # evaluation always in FP32
    txt, d = evaluate_annos(annos, args.root, ev.ids)
    r = summarize(d)
    json.dump({'split': args.eval_split, 'summary': r, 'full': {k: float(v) for k, v in d.items()}, 'args': vars(args)},
              open(os.path.join(args.out, f'eval_{args.eval_split}.json'), 'w'), indent=1)
    open(os.path.join(args.out, f'eval_{args.eval_split}.txt'), 'w').write(txt)
    print('FINAL', json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
