"""Fast-BEV (M0, R18, 4 frames) and Fast-BEV++ (R50, single frame) in plain PyTorch.

Rebuilt from the upstream code without mmcv/mmdet/mmdet3d; `load_m0()` / `load_pp()` load the
official checkpoints with an explicit name map (see README for the sources and sha256s).

Both models split into the same three kinds of piece, each an nn.Module with plain tensor inputs
so each exports to ONNX on its own:

  image encoder   6 camera images (normalized NCHW, or NHWC pixels with `uint8_input`) -> per-pixel
                  features, channels-last so the view transform gathers whole rows
  view transform  a Gather with host-computed indices (geometry.py) -- Fast-Ray's lookup table.
                  `ViewM0` / `ViewPP` are the gather pieces; `geometry.backproject_*` keep the
                  upstream loop form to check them against
  BEV network     BEV encoder + detection head -> raw head maps; decode + NMS on the host
                  (decode.py)

Fast-BEV M0 (Sense-GVT/Fast-BEV configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py)
  EncoderM0: ResNet-18 -> FPN(64, BN, 4 levels) -> neck_fuse_0 (the 4 levels bilinearly resized
             to stride 4, concatenated, 3x3 conv 256->64) -> (6, 64, 176, 64) NHWC
  ViewM0:    4 time steps x (200 x 200 x 4 voxels) gathered from the 4 frames' features; the
             channel order is Fast-BEV's space-to-channel (z, t, c) -> (1, 200, 200, 1024) NHWC
             (x rows, y columns: M2BevNeck keeps upstream's (X, Y) layout)
  BevM0:     M2BevNeck (1x1 fuse 1024->256, ResModule 256, stride-2 conv -> 192, 2 x (ResModule
             + conv)) -> FreeAnchor3DHead's three 1x1 convs on the transposed (Y, X) map ->
             cls (1, 100, 100, 80) logits, reg (1, 100, 100, 72), dir (1, 100, 100, 16), NHWC

Fast-BEV++ R50 (ymlab/advanced-fastbev configs/fastbev/paper/fastbev-r50-cbgs.py)
  EncoderPP: ResNet-50 (C4, C5) -> CustomFPN (256, stride 16) -> depth_net 1x1 256 -> 59 depth
             bins + 64 features -> feats (6, 16, 44, 64) NHWC, depth = sigmoid (6, 16, 44, 59)
  ViewPP:    Index-Gather-Reshape: per voxel (128 x 128 x 7) one gathered feature row times one
             gathered depth probability, summed over the 7 height bins (fuse='sum')
             -> (1, 128, 128, 64) NHWC in the (Y, X) order the BEV encoder uses
  BevPP:     CustomResNet (3 x 2 BasicBlocks, 128/256/512) -> FPN_LSS -> CenterHead (shared 3x3 +
             6 SeparateHeads) -> 6 NHWC maps (1, 128, 128, k): heatmap 10 (logits), reg 2,
             height 1, dim 3, rot 2, vel 2
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

M0_FEAT_HW = (64, 176)
M0_GRID = (200, 200, 4)
PP_FEAT_HW = (16, 44)
PP_GRID = (128, 128, 7)
PP_D = 59
HEAD_PP = [("heatmap", 10), ("reg", 2), ("height", 1), ("dim", 3), ("rot", 2), ("vel", 2)]


def cbr(cin, cout, k=3, s=1, act=True, bias=False):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, bias=bias), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ ResNet (torchvision == mmdet names)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(y)) + idt)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, stride, 1, bias=False)  # style='pytorch': stride on the 3x3
        self.bn2 = nn.BatchNorm2d(c)
        self.conv3 = nn.Conv2d(c, c * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        return self.relu(self.bn3(self.conv3(y)) + idt)


class ResNet(nn.Module):
    def __init__(self, depth, out_indices):
        super().__init__()
        block, layers = {18: (BasicBlock, [2, 2, 2, 2]), 50: (Bottleneck, [3, 4, 6, 3])}[depth]
        self.out_indices = out_indices
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        cin = 64
        for i, n in enumerate(layers):
            c, stride = 64 * 2 ** i, 1 if i == 0 else 2
            ds = None
            if stride != 1 or cin != c * block.expansion:
                ds = nn.Sequential(nn.Conv2d(cin, c * block.expansion, 1, stride, bias=False),
                                   nn.BatchNorm2d(c * block.expansion))
            blocks = [block(cin, c, stride, ds)]
            cin = c * block.expansion
            blocks += [block(cin, c) for _ in range(n - 1)]
            setattr(self, f"layer{i + 1}", nn.Sequential(*blocks))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        outs = []
        for i in range(4):
            x = getattr(self, f"layer{i + 1}")(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


MEAN = torch.tensor([123.675, 116.28, 103.53])
STD = torch.tensor([58.395, 57.12, 57.375])


def pixels_minus_mean(x, bgr=False):
    """(N, H, W, 3) RGB pixel values 0..255 (float here; the quantized graph's uint8 input with
    scale 1, zero point 0 *is* the camera's pixels) -> (N, 3, H, W) pixels minus the mean each raw
    channel is normalized with. The 1/std and BEVDet's BGR channel swap are folded into conv1's
    weights (`fold_input_norm`): a Slice(step -1) flip and a Div on the full-resolution images took
    83% of Fast-BEV++'s encoder on the HTP. Exact, zero padding included (padding is 0 after the
    mean subtraction either way)."""
    m = MEAN.flip(0) if bgr else MEAN
    return (x.float() - m).permute(0, 3, 1, 2)


def fold_input_norm(conv, bgr=False):
    """conv1 on (x - mean) / std [channel-swapped if bgr] == conv1' on pixels_minus_mean(x)."""
    w = conv.weight.data / STD.view(1, 3, 1, 1)
    conv.weight.data = w.flip(1) if bgr else w


# ------------------------------------------------------------------ Fast-BEV M0
class EncoderM0(nn.Module):
    def __init__(self, uint8_input=False):
        super().__init__()
        self.uint8_input = uint8_input
        self.backbone = ResNet(18, (0, 1, 2, 3))
        self.lateral = nn.ModuleList([cbr(c, 64, 1, act=False) for c in (64, 128, 256, 512)])
        self.fpn = nn.ModuleList([cbr(64, 64, 3, act=False) for _ in range(4)])
        self.fuse = nn.Conv2d(256, 64, 3, 1, 1)

    def forward(self, img):
        if self.uint8_input:
            img = pixels_minus_mean(img)
        c = self.backbone(img)
        lat = [m(x) for m, x in zip(self.lateral, c)]
        for i in range(3, 0, -1):  # mmdet FPN: nearest upsample to the previous level's size
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[2:], mode="nearest")
        p = [m(x) for m, x in zip(self.fpn, lat)]
        size = p[0].shape[2:]
        p = [p[0]] + [F.interpolate(x, size=size, mode="bilinear", align_corners=False) for x in p[1:]]
        return self.fuse(torch.cat(p, 1)).permute(0, 2, 3, 1)  # (6, 64, 176, 64) NHWC


class ViewM0(nn.Module):
    """feats_t (6*64*176, 64) per time step + idx_t (200*200*4,) int32 into [feats_t; zero row]
    -> (1, 200, 200, 1024) with channel = z*256 + t*64 + c (Fast-BEV's S2C order)."""

    def forward(self, f0, f1, f2, f3, i0, i1, i2, i3):
        X, Y, Z = M0_GRID
        zero = torch.zeros(1, 64, dtype=f0.dtype)
        outs = []
        for f, i in zip((f0, f1, f2, f3), (i0, i1, i2, i3)):
            table = torch.cat([f.reshape(-1, 64), zero], 0)
            outs.append(torch.index_select(table, 0, i).reshape(X * Y, Z, 64))  # int32 Gather
        return torch.cat(outs, 2).reshape(1, X, Y, Z * 4 * 64)


class ResModule2D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv0 = cbr(c, c)
        self.conv1 = cbr(c, c, act=False)

    def forward(self, x):
        return F.relu(x + self.conv1(self.conv0(x)))


class BevM0(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse = nn.Conv2d(1024, 256, 1)
        self.model = nn.Sequential(ResModule2D(256), cbr(256, 192, s=2),
                                   ResModule2D(192), cbr(192, 192), ResModule2D(192), cbr(192, 192))
        self.conv_cls = nn.Conv2d(192, 80, 1)
        self.conv_reg = nn.Conv2d(192, 72, 1)
        self.conv_dir = nn.Conv2d(192, 16, 1)

    def forward(self, vol):  # (1, X=200, Y=200, 1024) NHWC
        x = self.model(self.fuse(vol.permute(0, 3, 1, 2)))
        x = x.transpose(-1, -2)  # FreeAnchor3DHead is_transpose: (y, x)
        return tuple(m(x).permute(0, 2, 3, 1) for m in (self.conv_cls, self.conv_reg, self.conv_dir))


# ------------------------------------------------------------------ Fast-BEV++
class EncoderPP(nn.Module):
    def __init__(self, uint8_input=False):
        super().__init__()
        self.uint8_input = uint8_input
        self.backbone = ResNet(50, (2, 3))
        self.lateral = nn.ModuleList([nn.Conv2d(1024, 256, 1), nn.Conv2d(2048, 256, 1)])
        self.fpn = nn.Conv2d(256, 256, 3, 1, 1)
        self.depth_net = nn.Conv2d(256, PP_D + 64, 1)

    def raw(self, img):
        """depth_net output (6, 123, 16, 44): 59 depth logits then 64 features."""
        if self.uint8_input:
            img = pixels_minus_mean(img, bgr=True)
        c4, c5 = self.backbone(img)
        l4, l5 = self.lateral[0](c4), self.lateral[1](c5)
        return self.depth_net(self.fpn(l4 + F.interpolate(l5, size=l4.shape[2:], mode="nearest")))

    def forward(self, img):
        x = self.raw(img).permute(0, 2, 3, 1)  # (6, 16, 44, 123)
        return x[..., PP_D:], torch.sigmoid(x[..., :PP_D])


class ViewPP(nn.Module):
    """feats (6*16*44, 64), depth (6*16*44*59,) + idx (128*128*7,) int32 into [feats; zero row],
    didx (same) into depth -> sum over the 7 height bins -> (1, 128(y), 128(x), 64)."""

    def forward(self, feats, depth, idx, didx):
        X, Y, Z = PP_GRID
        table = torch.cat([feats.reshape(-1, 64), torch.zeros(1, 64, dtype=feats.dtype)], 0)
        f = torch.index_select(table, 0, idx) * torch.index_select(depth.reshape(-1), 0, didx).unsqueeze(1)
        return f.reshape(Y * X, Z, 64).sum(1).reshape(1, Y, X, 64)  # rank <= 4 for the HTP


class BasicBlockDS(BasicBlock):
    """mmdet BasicBlock with CustomResNet's biased 3x3 conv downsample."""

    def __init__(self, cin, c, stride):
        super().__init__(cin, c, stride, nn.Conv2d(cin, c, 3, stride, 1))


class BevPP(nn.Module):
    def __init__(self):
        super().__init__()
        chans = [128, 256, 512]
        cin, layers = 64, []
        for c in chans:
            layers.append(nn.Sequential(BasicBlockDS(cin, c, 2), BasicBlock(c, c)))
            cin = c
        self.layers = nn.Sequential(*layers)
        self.neck_conv = nn.Sequential(nn.Conv2d(640, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                       nn.ReLU(inplace=True), nn.Conv2d(512, 512, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.neck_up2 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True), nn.Conv2d(256, 256, 1))
        self.shared = cbr(256, 64)
        self.heads = nn.ModuleDict({n: nn.Sequential(cbr(64, 64), nn.Conv2d(64, c, 3, 1, 1)) for n, c in HEAD_PP})

    def forward(self, bev):  # (1, 128, 128, 64) NHWC, rows = y
        x = bev.permute(0, 3, 1, 2)
        f0 = self.layers[0](x)
        f2 = self.layers[2](self.layers[1](f0))
        x = torch.cat([f0, F.interpolate(f2, scale_factor=4, mode="bilinear", align_corners=True)], 1)
        x = self.neck_conv(x)
        x = self.neck_up2(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))
        x = self.shared(x)
        # separate outputs (not one concat): each gets its own quantization scale
        return tuple(self.heads[n](x).permute(0, 2, 3, 1) for n, _ in HEAD_PP)


# ------------------------------------------------------------------ checkpoint loading
def _load(mod, sd, rename, prefix_skip=()):
    out = {}
    for k, v in sd.items():
        if k.endswith("num_batches_tracked") or any(k.startswith(p) for p in prefix_skip):
            continue
        nk = rename(k)
        if nk is not None:
            out[nk] = v
    missing, unexpected = mod.load_state_dict(out, strict=False)
    missing = [m for m in missing if not m.endswith("num_batches_tracked")]
    assert not missing and not unexpected, (type(mod).__name__, missing[:8], unexpected[:8])
    return mod.eval()


def _cm(k):
    """mmcv ConvModule (conv, bn) -> Sequential (0, 1)."""
    return k.replace(".conv.", ".0.").replace(".bn.", ".1.")


def load_m0(path, uint8_input=False):
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]

    def enc(k):
        if k.startswith("backbone."):
            return k
        if k.startswith("neck.lateral_convs."):
            return _cm(k.replace("neck.lateral_convs.", "lateral."))
        if k.startswith("neck.fpn_convs."):
            return _cm(k.replace("neck.fpn_convs.", "fpn."))
        if k.startswith("neck_fuse_0."):
            return k.replace("neck_fuse_0.", "fuse.")
        return None

    def bev(k):
        if k.startswith("neck_3d.fuse."):
            return k.replace("neck_3d.", "")
        if k.startswith("neck_3d.model."):
            return _cm(k.replace("neck_3d.", ""))
        if k.startswith("bbox_head."):
            return k.replace("bbox_head.", "").replace("conv_dir_cls", "conv_dir")
        return None

    e = _load(EncoderM0(uint8_input), sd, enc)
    if uint8_input:
        fold_input_norm(e.backbone.conv1)
    return e, _load(BevM0(), sd, bev)


def load_pp(path, uint8_input=False):
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]

    def enc(k):
        if k.startswith("img_backbone."):
            return k.replace("img_backbone.", "backbone.")
        if k.startswith("img_neck.lateral_convs."):
            return k.replace("img_neck.lateral_convs.", "lateral.").replace(".conv.", ".")
        if k.startswith("img_neck.fpn_convs.0.conv."):
            return k.replace("img_neck.fpn_convs.0.conv.", "fpn.")
        if k.startswith("img_view_transformer.depth_net."):
            return k.replace("img_view_transformer.", "")
        return None

    def bev(k):
        if k.startswith("img_bev_encoder_backbone.layers."):
            return k.replace("img_bev_encoder_backbone.", "")
        if k.startswith("img_bev_encoder_neck.conv."):
            return k.replace("img_bev_encoder_neck.conv.", "neck_conv.")
        if k.startswith("img_bev_encoder_neck.up2."):
            i, rest = k[len("img_bev_encoder_neck.up2."):].split(".", 1)
            return f"neck_up2.{int(i) - 1}.{rest}"  # up2.0 is the Upsample
        if k.startswith("pts_bbox_head.shared_conv."):
            return _cm(k.replace("pts_bbox_head.shared_conv.", "shared."))
        if k.startswith("pts_bbox_head.task_heads.0."):
            return _cm(k.replace("pts_bbox_head.task_heads.0.", "heads."))
        return None

    e = _load(EncoderPP(uint8_input), sd, enc)
    if uint8_input:
        fold_input_norm(e.backbone.conv1, bgr=True)
    return e, _load(BevPP(), sd, bev, ("img_view_transformer.voxel_coords",))
