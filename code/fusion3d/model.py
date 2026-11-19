"""MVX-Net-style point-level LiDAR-camera fusion detector with a PointPillars LiDAR branch.

Components (used as compression units in the paper):
  image   : ResNet (ImageNet init, stem+layer1 frozen, frozen BN) + FPN, 5 levels, 256 channels
  fusion  : PointFusion (Sindagi et al. 2019, mmdet3d form): per-level lateral 3x3 convs, bilinear
            sampling at each point's pixel, linear projection, added to projected point features
  lidar   : dynamic pillar feature net (2 PFN layers) + pillar scatter + PointPillars 2D backbone
  head    : anchor head (1x1 convs: class, box, direction), OpenPCDet AnchorHeadSingle
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.ops.misc import FrozenBatchNorm2d

from .pcdet_lite.base_bev_backbone import BaseBEVBackbone
from .pcdet_lite.anchor_head_single import AnchorHeadSingle
from .pcdet_lite import model_nms_utils
from .data import PC_RANGE, CLASS_NAMES

VOXEL = np.array([0.16, 0.16, 4.0], dtype=np.float32)
GRID = np.round((PC_RANGE[3:] - PC_RANGE[:3]) / VOXEL).astype(np.int64)   # [432, 496, 1]


class Cfg(dict):
    """Minimal attribute dict (stands in for EasyDict used by OpenPCDet code)."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    __setattr__ = dict.__setitem__

    @staticmethod
    def wrap(d):
        if isinstance(d, dict):
            return Cfg({k: Cfg.wrap(v) for k, v in d.items()})
        if isinstance(d, list):
            return [Cfg.wrap(v) for v in d]
        return d


BACKBONE_CFG = Cfg.wrap(dict(LAYER_NUMS=[3, 5, 5], LAYER_STRIDES=[2, 2, 2], NUM_FILTERS=[64, 128, 256],
                             UPSAMPLE_STRIDES=[1, 2, 4], NUM_UPSAMPLE_FILTERS=[128, 128, 128]))


def _anchor(name, size, h, mt, ut):
    return {'class_name': name, 'anchor_sizes': [size], 'anchor_rotations': [0, 1.57], 'anchor_bottom_heights': [h],
            'align_center': False, 'feature_map_stride': 2, 'matched_threshold': mt, 'unmatched_threshold': ut}


HEAD_CFG = Cfg.wrap(dict(
    CLASS_AGNOSTIC=False, USE_DIRECTION_CLASSIFIER=True, DIR_OFFSET=0.78539, DIR_LIMIT_OFFSET=0.0, NUM_DIR_BINS=2,
    ANCHOR_GENERATOR_CONFIG=[_anchor('Car', [3.9, 1.6, 1.56], -1.78, 0.6, 0.45),
                             _anchor('Pedestrian', [0.8, 0.6, 1.73], -0.6, 0.5, 0.35),
                             _anchor('Cyclist', [1.76, 0.6, 1.73], -0.6, 0.5, 0.35)],
    TARGET_ASSIGNER_CONFIG=dict(NAME='AxisAlignedTargetAssigner', POS_FRACTION=-1.0, SAMPLE_SIZE=512,
                                NORM_BY_NUM_EXAMPLES=False, MATCH_HEIGHT=False, BOX_CODER='ResidualCoder'),
    LOSS_CONFIG=dict(LOSS_WEIGHTS={'cls_weight': 1.0, 'loc_weight': 2.0, 'dir_weight': 0.2,
                                   'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]})))
NMS_CFG = Cfg.wrap(dict(MULTI_CLASSES_NMS=False, NMS_TYPE='nms_gpu', NMS_THRESH=0.01, NMS_PRE_MAXSIZE=4096,
                        NMS_POST_MAXSIZE=500))
SCORE_THRESH = 0.1


# ---------------------------------------------------------------------------------------- image
class ImageBranch(nn.Module):
    def __init__(self, depth=50, pretrained=True, fpn_channels=256):
        super().__init__()
        ctor = {18: torchvision.models.resnet18, 34: torchvision.models.resnet34, 50: torchvision.models.resnet50}[depth]
        weights = {18: 'IMAGENET1K_V1', 34: 'IMAGENET1K_V1', 50: 'IMAGENET1K_V1'}[depth] if pretrained else None
        r = ctor(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        for p in list(self.stem.parameters()) + list(self.layer1.parameters()):
            p.requires_grad_(False)                                   # frozen_stages=1 as in MVX-Net
        chans = [64, 128, 256, 512] if depth < 50 else [256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(chans, fpn_channels, extra_blocks=LastLevelMaxPool())
        self.out_channels = fpn_channels

    def forward(self, x):
        c1 = self.layer1(self.stem(x))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        feats = self.fpn({'0': c1, '1': c2, '2': c3, '3': c4})
        return list(feats.values())                                   # strides 4, 8, 16, 32, 64


# ---------------------------------------------------------------------------------------- fusion
class PointFusion(nn.Module):
    def __init__(self, img_channels=256, pts_channels=64, mid_channels=128, out_channels=128, levels=5):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(img_channels, mid_channels, 3, padding=1) for _ in range(levels)])
        self.img_transform = nn.Sequential(nn.Linear(mid_channels * levels, out_channels),
                                           nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01))
        self.pts_transform = nn.Sequential(nn.Linear(pts_channels, out_channels),
                                           nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def lateral(self, img_feats):
        return [conv(f) for conv, f in zip(self.lateral_convs, img_feats)]

    @staticmethod
    def sample(lat_feats, uv, batch_idx, batch_size, img_hw):
        H, W = img_hw
        grid = torch.stack([uv[:, 0] / W * 2 - 1, uv[:, 1] / H * 2 - 1], -1)   # align_corners=False convention
        out = uv.new_zeros((uv.shape[0], sum(f.shape[1] for f in lat_feats)))
        for b in range(batch_size):
            m = batch_idx == b
            g = grid[m].view(1, 1, -1, 2)
            per = [F.grid_sample(f[b:b + 1], g.to(f.dtype), mode='bilinear', padding_mode='zeros', align_corners=False)
                   .view(f.shape[1], -1).t() for f in lat_feats]
            out[m] = torch.cat(per, 1).to(out.dtype)
        return out

    def fuse(self, img_pts, pts_feats):
        return F.relu(self.img_transform(img_pts) + self.pts_transform(pts_feats))


# ---------------------------------------------------------------------------------------- lidar
class PFNLayer(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.linear = nn.Linear(cin, cout, bias=False)
        self.norm = nn.BatchNorm1d(cout, eps=1e-3, momentum=0.01)

    def forward(self, x):
        return F.relu(self.norm(self.linear(x)))


def scatter_max(feats, inv, n):
    out = feats.new_zeros((n, feats.shape[1]))
    return out.scatter_reduce(0, inv[:, None].expand(-1, feats.shape[1]), feats, reduce='amax', include_self=True)


def scatter_mean(feats, inv, n):
    s = feats.new_zeros((n, feats.shape[1])).index_add_(0, inv, feats)
    c = feats.new_zeros((n,)).index_add_(0, inv, torch.ones_like(inv, dtype=feats.dtype))
    return s / c.clamp(min=1)[:, None]


class DynamicPillarNet(nn.Module):
    """Dynamic pillar feature net: point features [x,y,z,r, cluster offset(3), pillar-center offset(3)]."""
    def __init__(self, feat_channels=(64, 64)):
        super().__init__()
        self.pfn1 = PFNLayer(10, feat_channels[0])
        self.pfn2 = PFNLayer(feat_channels[0] * 2, feat_channels[1])
        self.register_buffer('vsize', torch.tensor(VOXEL), persistent=False)
        self.register_buffer('pcr', torch.tensor(PC_RANGE), persistent=False)

    def voxelize(self, points):
        # points: (N, 5) [b, x, y, z, r]
        xyz = points[:, 1:4]
        coords = torch.floor((xyz - self.pcr[:3]) / self.vsize).long()
        m = (coords[:, 0] >= 0) & (coords[:, 0] < GRID[0]) & (coords[:, 1] >= 0) & (coords[:, 1] < GRID[1]) & \
            (coords[:, 2] >= 0) & (coords[:, 2] < GRID[2])
        key = points[:, 0].long() * (GRID[0] * GRID[1]) + coords[:, 1] * GRID[0] + coords[:, 0]
        return m, key, coords

    def point_features(self, points, key_inv, n_vox, coords):
        xyz = points[:, 1:4]
        f_cluster = xyz - scatter_mean(xyz, key_inv, n_vox)[key_inv]
        centers = (coords[:, :3].to(xyz.dtype) * self.vsize + self.vsize / 2 + self.pcr[:3])
        f_center = xyz - centers
        f = torch.cat([points[:, 1:5], f_cluster, f_center], 1)
        x1 = self.pfn1(f)
        x1 = torch.cat([x1, scatter_max(x1, key_inv, n_vox)[key_inv]], 1)
        return self.pfn2(x1)                                         # (N, 64) per-point features


class Detector(nn.Module):
    def __init__(self, use_image=True, img_depth=50, pretrained=True):
        super().__init__()
        self.use_image = use_image
        self.pillar = DynamicPillarNet()
        if use_image:
            self.image = ImageBranch(img_depth, pretrained)
            self.fusion = PointFusion(img_channels=self.image.out_channels)
            bev_c = 128
        else:
            bev_c = 64
        self.bev_channels = bev_c
        self.backbone = BaseBEVBackbone(BACKBONE_CFG, bev_c)
        self.head = AnchorHeadSingle(model_cfg=HEAD_CFG, input_channels=self.backbone.num_bev_features,
                                     num_class=3, class_names=CLASS_NAMES, grid_size=GRID,
                                     point_cloud_range=PC_RANGE, predict_boxes_when_training=False)

    # -- stages exposed separately so that each branch can be compressed / exported on its own
    def lidar_points(self, points, uv=None):
        m, key, coords = self.pillar.voxelize(points)
        points, key, coords = points[m], key[m], coords[m]
        if uv is not None:
            uv = uv[m]
        uniq, inv = torch.unique(key, return_inverse=True)
        return points, uv, uniq, inv, coords

    def scatter_bev(self, vox_feats, uniq, batch_size):
        C = vox_feats.shape[1]
        canvas = vox_feats.new_zeros((batch_size, C, int(GRID[1]) * int(GRID[0])))
        b = uniq // (GRID[0] * GRID[1])
        pos = uniq % (GRID[0] * GRID[1])
        canvas[b, :, pos] = vox_feats
        return canvas.view(batch_size, C, int(GRID[1]), int(GRID[0]))

    def forward_features(self, batch):
        B = batch['batch_size']
        points, uv, uniq, inv, coords = self.lidar_points(batch['points'], batch.get('uv'))
        pf = self.pillar.point_features(points, inv, uniq.numel(), coords)
        if self.use_image:
            img_feats = self.image(batch['images'])
            lat = self.fusion.lateral(img_feats)
            img_pts = self.fusion.sample(lat, uv, points[:, 0].long(), B, batch['images'].shape[-2:])
            p = float(getattr(self, 'img_drop_p', 0.0))
            if self.training and p > 0:             # modality dropout: drop the image term for whole samples
                it = self.fusion.img_transform(img_pts.to(pf.dtype))
                keep = (torch.rand(B, device=it.device) >= p).to(it.dtype) / (1 - p)
                pf = F.relu(it * keep[points[:, 0].long()][:, None] + self.fusion.pts_transform(pf))
            else:
                pf = self.fusion.fuse(img_pts.to(pf.dtype), pf)
        vox = scatter_max(pf, inv, uniq.numel())
        bev = self.scatter_bev(vox, uniq, B)
        return self.backbone({'spatial_features': bev})['spatial_features_2d']

    def forward(self, batch):
        feats = self.forward_features(batch)
        dev = feats.device
        if self.head.anchors[0].device != dev:
            self.head.anchors = [a.to(dev) for a in self.head.anchors]
        rl = self.head.reg_loss_func                    # plain tensor attribute in OpenPCDet's loss: follow the model
        if getattr(rl, 'code_weights', None) is not None and torch.is_tensor(rl.code_weights) and rl.code_weights.device != dev:
            rl.code_weights = rl.code_weights.to(dev)
        data = {'spatial_features_2d': feats.float(), 'batch_size': batch['batch_size']}
        if self.training:
            data['gt_boxes'] = batch['gt_boxes'].to(dev)
        data = self.head(data)
        if self.training:
            loss, tb = self.head.get_loss()
            return loss, tb
        return self.post_process(data)

    @torch.no_grad()
    def post_process(self, data):
        preds = []
        for b in range(data['batch_size']):
            box_preds = data['batch_box_preds'][b]
            cls_preds = torch.sigmoid(data['batch_cls_preds'][b])
            scores, labels = torch.max(cls_preds, dim=-1)
            sel, sel_scores = model_nms_utils.class_agnostic_nms(box_scores=scores, box_preds=box_preds,
                                                                nms_config=NMS_CFG, score_thresh=SCORE_THRESH)
            preds.append({'pred_boxes': box_preds[sel], 'pred_scores': sel_scores, 'pred_labels': labels[sel] + 1})
        return preds


def component_of(name):
    """Map a parameter/module name to its compression unit."""
    if name.startswith('image.'):
        return 'image'
    if name.startswith('fusion.'):
        return 'fusion'
    if name.startswith('pillar.') or name.startswith('backbone.'):
        return 'lidar'
    if name.startswith('head.'):
        return 'head'
    return 'other'
