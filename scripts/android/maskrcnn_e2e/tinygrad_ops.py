"""ONNX operators tinygrad 0.14's frontend lacks (or mishandles) for Mask R-CNN.

`NonMaxSuppression` (85 nodes) and `RoiAlign` (8 nodes) are the only operator types in
MaskRCNN-12-qdq that `tinygrad.nn.onnx` does not implement (TVM's Relax frontend lacks 11).
Both have data-dependent shapes, so they run on NumPy; a guard for zero-size `ScatterElements`
is also included. `register` plugs them into an `OnnxRunner`. Semantics follow the ONNX spec / ONNX Runtime for opset 12.
"""

from __future__ import annotations

import numpy as np
from tinygrad import Tensor
from tinygrad.nn.onnx import OnnxRunner, onnx_ops


def _scalar(value, default):
    if value is None:
        return default
    if isinstance(value, Tensor):
        value = value.numpy()
    return np.asarray(value).reshape(-1)[0].item()


def _iou(box, others, center):
    if center:  # [x_center, y_center, width, height]
        def corners(b):
            return np.stack(
                [b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2,
                 b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], axis=-1)  # fmt: skip

        box, others = corners(box), corners(others)
    else:  # [y1, x1, y2, x2] in any corner order
        def normalize(b):
            y = np.sort(b[..., [0, 2]], axis=-1)
            x = np.sort(b[..., [1, 3]], axis=-1)
            return np.stack([x[..., 0], y[..., 0], x[..., 1], y[..., 1]], axis=-1)

        box, others = normalize(box), normalize(others)
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = area + areas - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-30), 0.0)


def NonMaxSuppression(  # noqa: N802 - ONNX operator name
    boxes, scores, max_output_boxes_per_class=None, iou_threshold=None, score_threshold=None,
    center_point_box: int = 0,
):  # fmt: skip
    b, s = boxes.numpy(), scores.numpy()
    max_out = int(_scalar(max_output_boxes_per_class, 0))
    iou_thr = float(_scalar(iou_threshold, 0.0))
    score_thr = float(_scalar(score_threshold, -np.inf))
    selected = []
    if max_out > 0:
        for batch in range(b.shape[0]):
            for cls in range(s.shape[1]):
                sc = s[batch, cls]
                order = np.flatnonzero(sc > score_thr)
                order = order[np.argsort(-sc[order], kind="stable")]
                kept: list[int] = []
                while len(order) and len(kept) < max_out:
                    top = order[0]
                    kept.append(int(top))
                    if len(order) == 1:
                        break
                    overlaps = _iou(b[batch, top], b[batch, order[1:]], bool(center_point_box))
                    order = order[1:][overlaps <= iou_thr]
                selected += [(batch, cls, k) for k in kept]
    return Tensor(np.array(selected, dtype=np.int64).reshape(-1, 3))


def roi_align(x, rois, batch_indices, mode, output_height, output_width, sampling_ratio, spatial_scale):
    _, channels, height, width = x.shape
    out = np.zeros((rois.shape[0], channels, output_height, output_width), dtype=np.float32)
    ph = np.arange(output_height)[:, None, None, None]
    pw = np.arange(output_width)[None, :, None, None]
    for r in range(rois.shape[0]):
        x1, y1, x2, y2 = (rois[r] * spatial_scale).tolist()
        roi_w, roi_h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        bin_h, bin_w = roi_h / output_height, roi_w / output_width
        grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(roi_h / output_height))
        grid_w = sampling_ratio if sampling_ratio > 0 else int(np.ceil(roi_w / output_width))
        iy = np.arange(grid_h)[None, None, :, None]
        ix = np.arange(grid_w)[None, None, None, :]
        ys = np.broadcast_to(y1 + ph * bin_h + (iy + 0.5) * bin_h / grid_h, (output_height, output_width, grid_h, grid_w))
        xs = np.broadcast_to(x1 + pw * bin_w + (ix + 0.5) * bin_w / grid_w, (output_height, output_width, grid_h, grid_w))
        valid = (ys >= -1) & (ys <= height) & (xs >= -1) & (xs <= width)
        ys, xs = np.maximum(ys, 0.0), np.maximum(xs, 0.0)
        y_low, x_low = ys.astype(np.int64), xs.astype(np.int64)
        clamp_y, clamp_x = y_low >= height - 1, x_low >= width - 1
        y_low, x_low = np.where(clamp_y, height - 1, y_low), np.where(clamp_x, width - 1, x_low)
        y_high, x_high = np.where(clamp_y, height - 1, y_low + 1), np.where(clamp_x, width - 1, x_low + 1)
        ys, xs = np.where(clamp_y, y_low.astype(np.float32), ys), np.where(clamp_x, x_low.astype(np.float32), xs)
        ly, lx = ys - y_low, xs - x_low
        hy, hx = 1 - ly, 1 - lx
        feat = x[int(batch_indices[r])]
        samples = (
            feat[:, y_low, x_low] * (hy * hx) + feat[:, y_low, x_high] * (hy * lx)
            + feat[:, y_high, x_low] * (ly * hx) + feat[:, y_high, x_high] * (ly * lx)
        ) * valid  # fmt: skip
        out[r] = samples.mean(axis=(-1, -2)) if mode == "avg" else samples.max(axis=(-1, -2))
    return out


def RoiAlign(  # noqa: N802
    x, rois, batch_indices, mode: str = "avg", output_height: int = 1, output_width: int = 1,
    sampling_ratio: int = 0, spatial_scale: float = 1.0, **_ignored,
):  # fmt: skip
    return Tensor(
        roi_align(
            x.numpy(), rois.numpy(), batch_indices.numpy(), mode, int(output_height), int(output_width),
            int(sampling_ratio), float(spatial_scale),
        )
    )


def ScatterElements(x, indices, updates, axis: int = 0, reduction: str = "none"):  # noqa: N802
    """tinygrad's scatter fails on zero-size updates (an FPN level that received no RoIs)."""
    if 0 in indices.shape or 0 in updates.shape:
        return x
    return onnx_ops["ScatterElements"](x, indices, updates, axis=axis, reduction=reduction)


def register(runner: OnnxRunner) -> OnnxRunner:
    """Add the missing operators to a runner (its op table is otherwise the module global)."""
    runner.onnx_ops = {
        **onnx_ops,
        "NonMaxSuppression": NonMaxSuppression,
        "RoiAlign": RoiAlign,
        "ScatterElements": ScatterElements,
    }
    return runner
