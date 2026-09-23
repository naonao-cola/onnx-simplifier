"""scripts/android/deploy's `postprocess: {kind: yolo_end2end}` (YOLO26's NMS-free top-k on the
CPU) against a numpy transcription of Ultralytics' Detect.get_topk_index / postprocess."""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from onnx import parser

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts" / "android" / "deploy")
)
from stages import post  # noqa: E402

NC, N, K = 5, 64, 10


def _head_model(tmp_path, dims):
    shape = ",".join(map(str, dims))
    m = parser.parse_model(
        f"""<ir_version: 8, opset_import: ["": 17]>
        g (float[{shape}] x) => (float[{shape}] output0) {{
            output0 = Identity(x)
        }}"""
    )
    p = tmp_path / "head.onnx"
    p.write_bytes(m.SerializeToString())
    return p


def _run(tmp_path, dims, x):
    info = post.build(
        {"kind": "yolo_end2end", "max_det": K}, _head_model(tmp_path, dims), tmp_path
    )
    s = ort.InferenceSession(
        str(tmp_path / "post.onnx"), providers=["CPUExecutionProvider"]
    )
    return info, s.run(None, {"output0": x})


def _reference(head):
    """Ultralytics: preds (1, N, 4+nc); top-k anchors by best class score, then top-k of their
    (anchor, class) scores; boxes gathered, class = flat index % nc."""
    p = head[0].T
    boxes, scores = p[:, :4], p[:, 4:]
    k = min(K, p.shape[0])
    anchors = np.argsort(-scores.max(-1), kind="stable")[:k]
    cand = scores[anchors].ravel()
    top = np.argsort(-cand, kind="stable")[:k]
    return boxes[anchors[top // NC]], cand[top], top % NC


def test_head_output_topk_matches_ultralytics(tmp_path):
    rng = np.random.default_rng(0)
    # distinct scores: argsort tie order vs TopK's is not part of the contract
    head = np.concatenate(
        [
            rng.uniform(0, 640, (1, 4, N)),
            rng.permutation(N * NC).reshape(1, NC, N) / (N * NC),
        ],
        axis=1,
    ).astype(np.float32)
    info, (b, s, c) = _run(tmp_path, [1, 4 + NC, N], head)
    assert info["num_classes"] == NC and not info["topk_on_htp"]
    rb, rs, rc = _reference(head)
    np.testing.assert_array_equal(b, rb)
    np.testing.assert_array_equal(s, rs)
    np.testing.assert_array_equal(c, rc)
    assert c.dtype == np.int64


def test_topk_already_in_model_is_split(tmp_path):
    rng = np.random.default_rng(1)
    dets = rng.uniform(0, 1, (1, K, 6)).astype(np.float32)
    dets[..., 5] = rng.integers(0, 80, K)
    info, (b, s, c) = _run(tmp_path, [1, K, 6], dets)
    assert info["topk_on_htp"]
    np.testing.assert_array_equal(b, dets[0, :, :4])
    np.testing.assert_array_equal(s, dets[0, :, 4])
    np.testing.assert_array_equal(c, dets[0, :, 5].astype(np.int64))


if __name__ == "__main__":
    pytest.main([__file__])
