"""Evaluate a saved model on a split; writes summary JSON (and optionally KITTI-format txt files)."""
import argparse, json, os
import torch
from fusion3d.data import KittiFusionDataset, make_loader
from fusion3d.evaluation import run_inference, evaluate_annos, summarize, write_kitti_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--split', default='tune')
    ap.add_argument('--out', required=True)
    ap.add_argument('--write-txt', default='')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    m = torch.load(a.model, map_location='cpu', weights_only=False).cuda().eval()
    ds = KittiFusionDataset(a.root, os.path.join(a.root, 'ImageSets', a.split + '.txt'), training=False, use_image=m.use_image)
    annos = run_inference(m, make_loader(ds, 4, False, a.workers), 'cuda')
    if a.write_txt:
        write_kitti_txt(annos, a.write_txt)
    txt, d = evaluate_annos(annos, a.root, ds.ids)
    r = summarize(d)
    json.dump({'split': a.split, 'model': a.model, 'summary': r, 'full': {k: float(v) for k, v in d.items()}},
              open(a.out, 'w'), indent=1)
    print('FINAL', json.dumps(r))


if __name__ == '__main__':
    main()
