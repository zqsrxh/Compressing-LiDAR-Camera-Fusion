# Rotated-box IoU for the KITTI evaluation, CPU (numba).
# Replaces the CUDA rotate_iou_gpu_eval of the second.pytorch / OpenPCDet python evaluator.
# The original vertex-sorting routine is numerically fragile on CPU for coincident boxes, so this
# version computes the exact intersection by Sutherland-Hodgman clipping of convex polygons in float64.
# Box format and conventions follow the original: (x, y, w, h, angle), corners rotated by -angle,
# criterion -1: IoU, 0: inter / area(query), 1: inter / area(box), 2: intersection area.
import math
import numba
import numpy as np


@numba.njit(cache=True)
def _corners(b, out):
    c = math.cos(b[4]); s = math.sin(b[4])
    xs = (-b[2] / 2, -b[2] / 2, b[2] / 2, b[2] / 2)
    ys = (-b[3] / 2, b[3] / 2, b[3] / 2, -b[3] / 2)
    for i in range(4):
        out[i, 0] = c * xs[i] + s * ys[i] + b[0]
        out[i, 1] = -s * xs[i] + c * ys[i] + b[1]


@numba.njit(cache=True)
def _signed_area(p, n):
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += p[i, 0] * p[j, 1] - p[j, 0] * p[i, 1]
    return 0.5 * a


@numba.njit(cache=True)
def _inter_area(b1, b2):
    P = np.zeros((4, 2)); Q = np.zeros((4, 2))
    _corners(b1, P); _corners(b2, Q)
    if _signed_area(P, 4) < 0:
        P = P[::-1].copy()
    if _signed_area(Q, 4) < 0:
        Q = Q[::-1].copy()
    poly = np.zeros((16, 2)); n = 4
    for i in range(4):
        poly[i] = P[i]
    tmp = np.zeros((16, 2))
    for e in range(4):
        ax, ay = Q[e, 0], Q[e, 1]
        bx, by = Q[(e + 1) % 4, 0], Q[(e + 1) % 4, 1]
        m = 0
        for i in range(n):
            cx, cy = poly[i, 0], poly[i, 1]
            dx, dy = poly[(i + 1) % n, 0], poly[(i + 1) % n, 1]
            sc = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            sd = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
            if sc >= 0:
                tmp[m, 0] = cx; tmp[m, 1] = cy; m += 1
            if (sc >= 0) != (sd >= 0):
                t = sc / (sc - sd)
                tmp[m, 0] = cx + t * (dx - cx); tmp[m, 1] = cy + t * (dy - cy); m += 1
        n = m
        for i in range(n):
            poly[i] = tmp[i]
        if n < 3:
            return 0.0
    return abs(_signed_area(poly, n))


@numba.njit(cache=True)
def _iou_one(qb, b, criterion):
    a1 = qb[2] * qb[3]
    a2 = b[2] * b[3]
    it = _inter_area(qb, b)
    if criterion == -1:
        return it / (a1 + a2 - it)
    elif criterion == 0:
        return it / a1
    elif criterion == 1:
        return it / a2
    return it


@numba.njit(cache=True, parallel=True)
def _kernel(boxes, query_boxes, iou, criterion):
    for n in numba.prange(boxes.shape[0]):
        for k in range(query_boxes.shape[0]):
            iou[n, k] = _iou_one(query_boxes[k], boxes[n], criterion)


def rotate_iou_cpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    box_dtype = boxes.dtype
    b = np.ascontiguousarray(boxes, dtype=np.float64)
    q = np.ascontiguousarray(query_boxes, dtype=np.float64)
    iou = np.zeros((b.shape[0], q.shape[0]), dtype=np.float64)
    if b.shape[0] and q.shape[0]:
        _kernel(b, q, iou, int(criterion))
    return iou.astype(box_dtype)
