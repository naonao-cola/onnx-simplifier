"""Image preprocessing and detection comparison shared by the TVM and tinygrad runners."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

MEAN_BGR = np.array([102.9801, 115.9465, 122.7717], dtype="float32")[:, None, None]


def canvas(path: Path, height: int, width: int) -> np.ndarray:
    """Resize to fit, convert to BGR, subtract the mean and pad onto a fixed canvas."""
    image = Image.open(path).convert("RGB")
    ratio = min(width / image.size[0], height / image.size[1])
    image = image.resize((int(image.size[0] * ratio), int(image.size[1] * ratio)), Image.BILINEAR)
    array = np.asarray(image, dtype="float32")[:, :, ::-1].transpose(2, 0, 1) - MEAN_BGR
    out = np.zeros((3, height, width), dtype="float32")
    out[:, : array.shape[1], : array.shape[2]] = array
    return out


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare(reference, candidate, threshold=0.5):
    """Match detections above `threshold` by label and box IoU; report agreement statistics."""
    ref_boxes, ref_labels, ref_scores, ref_masks = reference
    boxes, labels, scores, masks = candidate
    used, box_ious, score_diffs, mask_ious = set(), [], [], []
    for i in np.flatnonzero(ref_scores > threshold):
        best, best_j = 0.0, None
        for j in np.flatnonzero(scores > threshold):
            if j in used or labels[j] != ref_labels[i]:
                continue
            value = iou(ref_boxes[i], boxes[j])
            if value > best:
                best, best_j = value, j
        if best_j is not None and best > 0.5:
            used.add(best_j)
            box_ious.append(best)
            score_diffs.append(abs(float(ref_scores[i]) - float(scores[best_j])))
            a, b = ref_masks[i, 0] > 0.5, masks[best_j, 0] > 0.5
            mask_ious.append(float((a & b).sum() / max((a | b).sum(), 1)))
    mean = lambda values: float(np.mean(values)) if values else None  # noqa: E731
    return {
        "ref_detections": int((ref_scores > threshold).sum()),
        "tvm_detections": int((scores > threshold).sum()),
        "matched": len(box_ious),
        "mean_box_iou": mean(box_ious),
        "mean_score_absdiff": mean(score_diffs),
        "mean_mask_iou": mean(mask_ious),
    }
