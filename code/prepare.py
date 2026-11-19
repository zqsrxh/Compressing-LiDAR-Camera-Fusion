"""One-off preprocessing.
1) Field-of-view point cache: training/velodyne_fov/<id>.bin with (x, y, z, r, u, v) float32.
2) Sequence-disjoint tuning split carved out of the standard 3712-frame train split, using the
   KITTI devkit mapping (train_rand.txt + train_mapping.txt) from frames to raw drives.
"""
import argparse, os, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from fusion3d.pcdet_lite import calibration_kitti
import cv2

def fov_one(args):
    root, idx = args
    d = os.path.join(root, 'training')
    out = os.path.join(d, 'velodyne_fov', idx + '.bin')
    if os.path.exists(out):
        return idx
    pts = np.fromfile(os.path.join(d, 'velodyne', idx + '.bin'), np.float32).reshape(-1, 4)
    calib = calibration_kitti.Calibration(os.path.join(d, 'calib', idx + '.txt'))
    H, W = cv2.imread(os.path.join(d, 'image_2', idx + '.png')).shape[:2]
    uv, depth = calib.rect_to_img(calib.lidar_to_rect(pts[:, :3]))
    m = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (depth >= 0)
    np.concatenate([pts[m], uv[m].astype(np.float32)], 1).astype(np.float32).tofile(out)
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--tune-frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--fov-only', action='store_true', help='only build the FOV cache (use shipped split files)')
    a = ap.parse_args()
    d = os.path.join(a.root, 'training')
    os.makedirs(os.path.join(d, 'velodyne_fov'), exist_ok=True)
    ids = sorted(f[:-4] for f in os.listdir(os.path.join(d, 'velodyne')) if f.endswith('.bin'))
    with ProcessPoolExecutor(16) as ex:
        n = sum(1 for _ in ex.map(fov_one, [(a.root, i) for i in ids], chunksize=32))
    print('fov cache', n)
    if a.fov_only:
        return
    # drive of every training frame
    dev = os.path.join(a.root, 'devkit', 'mapping')
    rand = [int(x) for x in open(os.path.join(dev, 'train_rand.txt')).read().strip().split(',')]
    mapping = [l.split() for l in open(os.path.join(dev, 'train_mapping.txt')).read().strip().splitlines()]
    drive = {f'{i:06d}': mapping[r - 1][1] for i, r in enumerate(rand)}
    S = os.path.join(a.root, 'ImageSets')
    train = [l.strip() for l in open(os.path.join(S, 'train.txt')) if l.strip()]
    val = [l.strip() for l in open(os.path.join(S, 'val.txt')) if l.strip()]
    drives = sorted({drive[i] for i in train})
    # class counts per drive (Car, Pedestrian, Cyclist), used to stratify the drive-level split
    cls = ['Car', 'Pedestrian', 'Cyclist']
    cnt = {dn: np.zeros(4) for dn in drives}
    for i in train:
        cnt[drive[i]][3] += 1
        for l in open(os.path.join(d, 'label_2', i + '.txt')):
            n = l.split()[0]
            if n in cls:
                cnt[drive[i]][cls.index(n)] += 1
    tot = sum(cnt.values())
    rng = np.random.RandomState(a.seed)
    best, best_err = None, 1e9
    for trial in range(20000):
        perm = list(rng.permutation(drives))
        sel, s = [], np.zeros(4)
        for dn in perm:
            if s[3] >= a.tune_frac * tot[3]:
                break
            sel.append(dn); s += cnt[dn]
        frac = s / tot
        err = np.abs(frac[:3] - frac[3]).max() + abs(frac[3] - a.tune_frac)
        if err < best_err:
            best, best_err = sel, err
    tune_drives = sorted(best)
    print('tune class fractions', (sum(cnt[x] for x in tune_drives) / tot).round(3).tolist())
    tune = [i for i in train if drive[i] in tune_drives]
    tsub = [i for i in train if drive[i] not in tune_drives]
    open(os.path.join(S, 'train_sub.txt'), 'w').write('\n'.join(tsub) + '\n')
    open(os.path.join(S, 'tune.txt'), 'w').write('\n'.join(tune) + '\n')
    val_drives = {drive[i] for i in val}
    info = {'n_train': len(train), 'n_train_sub': len(tsub), 'n_tune': len(tune), 'n_val': len(val),
            'n_train_drives': len(drives), 'tune_drives': sorted(tune_drives),
            'overlap_tune_val_drives': sorted(set(tune_drives) & val_drives),
            'overlap_trainsub_val_drives': sorted({drive[i] for i in tsub} & val_drives)}
    json.dump(info, open(os.path.join(S, 'split_info.json'), 'w'), indent=1)
    print(json.dumps({k: (v if not isinstance(v, list) or len(v) < 12 else len(v)) for k, v in info.items()}))

if __name__ == '__main__':
    main()
