"""Fast-Ray lookup tables (host side) for Fast-BEV M0 and Fast-BEV++, plus upstream-literal copies.

The view transform of both models is a pure lookup: every voxel center is projected into the
cameras once, and the voxel takes one camera pixel's feature row. The tables only depend on
calibration (and, for M0's previous frames, on the ego motion since then), so they are computed on
the host and the graph just gathers (model.ViewM0 / ViewPP).

M0 (Fast-BEV mmdet3d/models/detectors/fastbev.py: get_points + backproject_inplace):
  voxel centers: 200 x 200 x 4 at (0.5, 0.5, 1.5) m from (-50, -50, -4); projection =
  diag(1/4, 1/4, 1) @ lidar2img[:3] (stride 4); x/y = round-half-even(p/z); valid inside the
  176 x 64 map with z > 0; cameras overwrite in order, so the *last* valid camera wins.
  -> m0_lut(lidar2img (6, 4, 4)) = int32 (200*200*4,) index into the (6*64*176 + 1)-row feature
     table of one time step (last row = zeros for "no camera"), in (x, y, z) order.

Fast-BEV++ (advanced-fastbev mmdet3d/models/necks/fastray_transformer.py: get_fastray_input):
  voxel centers: the checkpoint's `voxel_coords` (128 x 128 x 7, x-major); camera = post @
  (K | 0) @ inv(sensor2keyego); pixel = trunc((v, u) / 16) (Tensor.long(), so (-1, 0) -> 0 is kept,
  as upstream); valid on the 16 x 44 map with 1 <= depth < 60; the *first* valid camera wins;
  depth bin = trunc(depth) - 1 of 59. Upstream zeroes camera 0's pixel (0, 0) feature row before
  the gather and maps every unassigned voxel there, so those gather the zero row here.
  -> pp_lut(...) = (idx, didx) int32 (128*128*7,) in (y, x, z) order (the NHWC BEV layout the BEV
     encoder reads), idx into the (6*16*44 + 1)-row feature table, didx into the flat depth
     probabilities (6*16*44*59,).

`ref_backproject_inplace` / `ref_fastray` are the upstream functions copied as-is (modulo the
mmcv plumbing), used by validate.py to check the LUT + gather path bit for bit.
"""
from __future__ import annotations

import numpy as np
import torch

M0_N = (200, 200, 4)
M0_SIZE = (0.5, 0.5, 1.5)
M0_ORIGIN = (0.0, 0.0, -1.0)  # KittiSetOrigin: center of point_cloud_range [-50, -50, -5, 50, 50, 3]
M0_STRIDE = 4
M0_HW = (64, 176)
PP_STRIDE = 16
PP_HW = (16, 44)
PP_D = 59
PP_DEPTH = (1.0, 60.0, 1.0)


def m0_points():
    """get_points(): (3, 200, 200, 4) float32 voxel centers (upstream: torch float32)."""
    n = torch.tensor(M0_N)
    size = torch.tensor(M0_SIZE)
    origin = torch.tensor(np.array(M0_ORIGIN, np.float32))
    pts = torch.stack(torch.meshgrid([torch.arange(n[0]), torch.arange(n[1]), torch.arange(n[2])], indexing="ij"))
    new_origin = origin - n / 2.0 * size
    return pts * size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)


def m0_projection(lidar2img):
    """_compute_projection(): (6, 3, 4) float32 = diag(1/4, 1/4, 1) @ lidar2img[:3]."""
    intr = torch.eye(4)[:3, :3].clone()
    intr[:2] /= M0_STRIDE
    return torch.stack([intr @ torch.tensor(np.asarray(e, np.float32))[:3] for e in lidar2img])


def m0_lut(lidar2img, points=None):
    points = m0_points() if points is None else points
    proj = m0_projection(lidar2img)
    n_img = proj.shape[0]
    p = points.view(1, 3, -1).expand(n_img, 3, -1)
    p = torch.cat((p, torch.ones_like(p[:, :1])), dim=1)
    p2 = torch.bmm(proj, p)
    x = (p2[:, 0] / p2[:, 2]).round().long()
    y = (p2[:, 1] / p2[:, 2]).round().long()
    z = p2[:, 2]
    h, w = M0_HW
    valid = (x >= 0) & (y >= 0) & (x < w) & (y < h) & (z > 0)
    lut = torch.full((p.shape[-1],), n_img * h * w, dtype=torch.long)  # zero row
    for i in range(n_img):  # later cameras overwrite: last valid wins
        lut[valid[i]] = i * h * w + y[i, valid[i]] * w + x[i, valid[i]]
    return lut.to(torch.int32)


def pp_lut(voxel_coords, sensor2keyego, intrin, post):
    """voxel_coords (128*128*7, 3) from the checkpoint; sensor2keyego (6, 4, 4); intrin (6, 3, 3);
    post (3, 3) resize/crop -> idx, didx int32 (128*128*7,) in (y, x, z) order."""
    s2k = torch.tensor(np.asarray(sensor2keyego), dtype=torch.float32)
    k4 = torch.eye(4).repeat(6, 1, 1)
    k4[:, :3, :3] = torch.tensor(np.asarray(intrin), dtype=torch.float32)
    camego2img = k4.matmul(torch.inverse(s2k))
    post_rot = torch.eye(3).repeat(6, 1, 1)
    post_rot[:, :2, :2] = torch.tensor(post[:2, :2], dtype=torch.float32)
    post_tran = torch.zeros(6, 3)
    post_tran[:, :2] = torch.tensor(post[:2, 2], dtype=torch.float32)
    c = camego2img[:, :3, :3].matmul(voxel_coords.float().t()) + camego2img[:, :3, 3].reshape(-1, 3, 1)
    dist = c[:, 2, :].clone()
    c[:, 2, :][c[:, 2, :] <= 0.0] = torch.inf
    c[:, :2, :] /= c[:, 2:3, :]
    c = post_rot.matmul(c) + post_tran.reshape(-1, 3, 1)
    c = (c[:, :2, :].transpose(1, 2)[..., [1, 0]] / PP_STRIDE).long()  # (6, N, (v, u))
    fh, fw = PP_HW
    on = ((c[..., 0] < fh) & (c[..., 0] >= 0) & (c[..., 1] < fw) & (c[..., 1] >= 0)
          & (dist >= PP_DEPTH[0]) & (dist < PP_DEPTH[1]))
    first = torch.cumsum(on.int(), 0) == 1
    on = on & first  # the first valid camera wins
    zero_row = 6 * fh * fw
    n = voxel_coords.shape[0]
    idx = torch.full((n,), zero_row, dtype=torch.long)
    didx = torch.zeros(n, dtype=torch.long)
    for i in range(6):
        m = on[i]
        pix = i * fh * fw + c[i, m, 0] * fw + c[i, m, 1]
        idx[m] = pix
        d = ((dist[i, m] - (PP_DEPTH[0] - PP_DEPTH[2])) / PP_DEPTH[2]).long() - 1
        didx[m] = pix * PP_D + d
    idx[idx == 0] = zero_row  # camera 0's pixel (0, 0) is zeroed upstream
    X, Y, Z = 128, 128, 7

    def reorder(t):  # (x, y, z) -> (y, x, z)
        return t.view(X, Y, Z).permute(1, 0, 2).reshape(-1).to(torch.int32)

    return reorder(idx), reorder(didx)


# ---------------------------------------------------------------- upstream-literal copies
def ref_backproject_inplace(features, points, projection):
    """Fast-BEV fastbev.py backproject_inplace, verbatim. features (6, C, H, W) -> (C, X, Y, Z)."""
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    points_2d_3 = torch.bmm(projection, points)
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()
    z = points_2d_3[:, 2]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)
    volume = torch.zeros((n_channels, points.shape[-1]), device=features.device).type_as(features)
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    return volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)


def ref_fastray(x, depth_logits_and_feat, voxel_coords, sensor2keyego, intrin, post):
    """advanced-fastbev FastrayTransformer.forward (use_depth, sigmoid, fuse='sum', batch 1,
    identity bda), the non-accelerated loop path, verbatim in substance.
    depth_logits_and_feat: depth_net output (6, 123, 16, 44) -> BEV (1, 64, Y, X)."""
    B, N = 1, 6
    D, C = PP_D, 64
    xx = depth_logits_and_feat.view(B, N, D + C, *PP_HW).permute(0, 1, 3, 4, 2).clone()
    xx[:, 0, 0, 0] = 0.0
    depth = xx[..., :D].sigmoid()
    xx = xx[..., D:(D + C)]
    s2k = torch.tensor(np.asarray(sensor2keyego), dtype=torch.float32)
    new_cam2imgs = torch.eye(4).repeat(6, 1, 1)
    new_cam2imgs[:, :3, :3] = torch.tensor(np.asarray(intrin), dtype=torch.float32)
    camego2imgs = new_cam2imgs.matmul(torch.inverse(s2k))
    post_rots = torch.eye(3).repeat(6, 1, 1)
    post_rots[:, :2, :2] = torch.tensor(post[:2, :2], dtype=torch.float32)
    post_trans = torch.zeros(6, 3)
    post_trans[:, :2] = torch.tensor(post[:2, 2], dtype=torch.float32)
    cur_coords = voxel_coords.float().transpose(1, 0)
    cur_coords = camego2imgs[:, :3, :3].matmul(cur_coords)
    cur_coords += camego2imgs[:, :3, 3].reshape(-1, 3, 1)
    dist = cur_coords[:, 2, :]
    cur_coords[:, 2, :][cur_coords[:, 2, :] <= 0.0] = torch.inf
    cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]
    cur_coords = post_rots.matmul(cur_coords)
    cur_coords += post_trans.reshape(-1, 3, 1)
    cur_coords = cur_coords[:, :2, :].transpose(1, 2)
    cur_coords = cur_coords[..., [1, 0]] / PP_STRIDE
    cur_coords = cur_coords.long()
    on_img = ((cur_coords[..., 0] < (256 / PP_STRIDE)) & (cur_coords[..., 0] >= 0)
              & (cur_coords[..., 1] < (704 / PP_STRIDE)) & (cur_coords[..., 1] >= 0)
              & (dist >= PP_DEPTH[0]) & (dist < PP_DEPTH[1]))
    for valid_i in range(1, len(on_img)):
        for valid_j in range(0, valid_i):
            on_img[valid_i][on_img[valid_j] == True] = False  # noqa: E712 (verbatim)
    img_l, dep_l, vox_l = [], [], []
    for c in range(on_img.shape[0]):
        m = cur_coords[c, on_img[c]]
        img_l.append(torch.cat([m.new(m[:, 0:1].shape).zero_() + c, m[:, 0:1], m[:, 1:2]], dim=1))
        vox_l.append(torch.nonzero(on_img[c])[:, 0])
        dep_l.append(((dist[c, on_img[c]] - (PP_DEPTH[0] - PP_DEPTH[2])) / PP_DEPTH[2]).long() - 1)
    rest = torch.nonzero(~on_img.sum(0).bool())[:, 0]
    vox_l.append(rest)
    img_l.append(torch.zeros(rest.shape[0], 3).long())
    dep_l.append(torch.zeros(rest.shape[0]).long())
    vox, img, dep = torch.cat(vox_l), torch.cat(img_l), torch.cat(dep_l)
    vf = torch.zeros((B, 128 * 128 * 7, C)).type_as(xx)
    vf[0][vox] = xx[0][img[:, 0], img[:, 1], img[:, 2]] * depth[0][img[:, 0], img[:, 1], img[:, 2], dep].unsqueeze(-1)
    out = vf.view(B, 128, 128, 7, C)
    return out.sum(dim=-2).permute(0, 3, 2, 1)
