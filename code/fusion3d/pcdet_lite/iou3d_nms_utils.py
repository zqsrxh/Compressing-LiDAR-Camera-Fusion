# Pure-PyTorch replacements for OpenPCDet's compiled iou3d_nms ops (BEV rotated IoU and NMS).
# Written for this project; boxes are (N, 7+) [x, y, z, dx, dy, dz, heading] in the LiDAR frame.
import numpy as np
import torch


def _bev_corners(boxes):
    """(N, 7) -> (N, 4, 2) BEV corners in counter-clockwise order."""
    x, y, dx, dy, h = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 6]
    c, s = torch.cos(h), torch.sin(h)
    tx = torch.stack([dx, -dx, -dx, dx], 1) * 0.5
    ty = torch.stack([dy, dy, -dy, -dy], 1) * 0.5
    cx = tx * c[:, None] - ty * s[:, None] + x[:, None]
    cy = tx * s[:, None] + ty * c[:, None] + y[:, None]
    return torch.stack([cx, cy], -1)


def _cross(o, a, b):
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])


def pairwise_bev_intersection(ca, cb, eps=1e-8):
    """Intersection area of P pairs of convex quads. ca, cb: (P, 4, 2) CCW corners -> (P,)."""
    P = ca.shape[0]
    if P == 0:
        return ca.new_zeros(0)
    a0, a1 = ca, ca.roll(-1, dims=1)          # edges of a: a0[i] -> a1[i]
    b0, b1 = cb, cb.roll(-1, dims=1)
    # vertices of a inside b (all cross products >= 0 for a CCW polygon)
    in_a = (_cross(b0[:, None, :, :], b1[:, None, :, :], ca[:, :, None, :]) >= -eps).all(-1)   # (P,4)
    in_b = (_cross(a0[:, None, :, :], a1[:, None, :, :], cb[:, :, None, :]) >= -eps).all(-1)
    # edge-edge intersections (P,4,4)
    p, r = a0[:, :, None, :], (a1 - a0)[:, :, None, :]
    q, s = b0[:, None, :, :], (b1 - b0)[:, None, :, :]
    rxs = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
    qp = q - p
    t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / (rxs + (rxs.abs() < eps).float() * eps)
    u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / (rxs + (rxs.abs() < eps).float() * eps)
    valid_e = (rxs.abs() > eps) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    pts_e = p + t[..., None] * r                                    # (P,4,4,2)
    pts = torch.cat([ca, cb, pts_e.reshape(P, 16, 2)], 1)           # (P,24,2)
    mask = torch.cat([in_a, in_b, valid_e.reshape(P, 16)], 1)       # (P,24)
    n = mask.sum(1)
    mf = mask.float()
    centroid = (pts * mf[..., None]).sum(1) / n.clamp(min=1)[:, None]
    ang = torch.atan2(pts[..., 1] - centroid[:, None, 1], pts[..., 0] - centroid[:, None, 0])
    ang = torch.where(mask, ang, torch.full_like(ang, 10.0))
    order = ang.argsort(1)
    sp = torch.gather(pts, 1, order[..., None].expand(-1, -1, 2))
    sm = torch.gather(mask, 1, order)
    first = sp[:, :1, :].expand_as(sp)
    sp = torch.where(sm[..., None], sp, first)                      # pad with the first vertex
    nxt = sp.roll(-1, dims=1)
    area = 0.5 * (sp[..., 0] * nxt[..., 1] - nxt[..., 0] * sp[..., 1]).sum(1).abs()
    return torch.where(n >= 3, area, torch.zeros_like(area))


def boxes_iou_bev(boxes_a, boxes_b, chunk=200000):
    """Rotated BEV IoU matrix (N, M)."""
    N, M = boxes_a.shape[0], boxes_b.shape[0]
    out = boxes_a.new_zeros((N, M))
    if N == 0 or M == 0:
        return out
    ca, cb = _bev_corners(boxes_a.float()), _bev_corners(boxes_b.float())
    area_a = (boxes_a[:, 3] * boxes_a[:, 4]).float()
    area_b = (boxes_b[:, 3] * boxes_b[:, 4]).float()
    # candidate pairs: circumscribed circles overlap
    ra = 0.5 * torch.sqrt(boxes_a[:, 3] ** 2 + boxes_a[:, 4] ** 2).float()
    rb = 0.5 * torch.sqrt(boxes_b[:, 3] ** 2 + boxes_b[:, 4] ** 2).float()
    d = torch.cdist(boxes_a[:, :2].float(), boxes_b[:, :2].float())
    ii, jj = torch.nonzero(d < (ra[:, None] + rb[None, :]), as_tuple=True)
    for k in range(0, ii.numel(), chunk):
        i, j = ii[k:k + chunk], jj[k:k + chunk]
        inter = pairwise_bev_intersection(ca[i], cb[j])
        out[i, j] = (inter / (area_a[i] + area_b[j] - inter).clamp(min=1e-8)).to(out.dtype)
    return out


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """Greedy rotated-BEV NMS with the same interface as OpenPCDet's nms_gpu."""
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    b = boxes[order].contiguous()
    if b.shape[0] == 0:
        return order[:0], None
    over = (boxes_iou_bev(b, b) > thresh)
    over = torch.triu(over, diagonal=1).cpu().numpy()
    n = over.shape[0]
    suppressed = np.zeros(n, dtype=bool)
    keep = []
    for i in range(n):
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= over[i]
    keep = torch.as_tensor(keep, dtype=torch.long, device=order.device)
    return order[keep].contiguous(), None


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """3D IoU (rotated BEV overlap x height overlap)."""
    ca, cb = _bev_corners(boxes_a.float()), _bev_corners(boxes_b.float())
    N, M = boxes_a.shape[0], boxes_b.shape[0]
    i, j = torch.meshgrid(torch.arange(N, device=boxes_a.device), torch.arange(M, device=boxes_a.device), indexing='ij')
    i, j = i.reshape(-1), j.reshape(-1)
    inter_bev = pairwise_bev_intersection(ca[i], cb[j]).view(N, M)
    a_top = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1); a_bot = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    b_top = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1); b_bot = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)
    h = (torch.min(a_top, b_top) - torch.max(a_bot, b_bot)).clamp(min=0)
    inter = inter_bev * h
    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)
    return inter / (vol_a + vol_b - inter).clamp(min=1e-6)
