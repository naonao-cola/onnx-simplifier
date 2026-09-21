"""Build the camera demo's Sobel edge-detector model: ONNX, then a K210 kmodel.

    python3.10 make_edge_demo_model.py OUT_DIR

Writes OUT_DIR/edge.onnx and OUT_DIR/edge.kmodel (flash the latter at
0x00C00000; see ../firmware/camera_demo/README.md).

The network is fixed-weight, no training: a 3x3 conv turns RGB into
+/- horizontal and +/- vertical luma gradients (Sobel kernels, one output
channel each), ReLU keeps the positive parts, and a 1x1 conv sums the four --
an L1 gradient magnitude, i.e. an edge map. Everything in it (3x3 conv, ReLU,
1x1 conv) runs on the KPU, and the result is easy to check against
onnxruntime. Input is 1x3x135x240 float32 in [0,1] -- the M5StickV LCD's
size, so camera pixels map 1:1.

PTQ calibration uses synthetic image-like data (smooth noise plus rectangles),
since there is no dataset here; edge maps are sparse so the quantisation is
coarse, which is fine for a visual demo.
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, parser

from onnx_to_kmodel import convert_to_kmodel

H, W = 135, 240

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T.copy()
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def build_model() -> onnx.ModelProto:
    model = parser.parse_model(f"""
    <ir_version: 8, opset_import: ["" : 13]>
    edge (float[1,3,{H},{W}] image) => (float[1,1,{H},{W}] edges) {{
      g = Conv<kernel_shape=[3,3], pads=[1,1,1,1]>(image, W1, B1)
      r = Relu(g)
      edges = Conv<kernel_shape=[1,1]>(r, W2, B2)
    }}
    """)
    # out channels: +gx, -gx, +gy, -gy of the luma image
    w1 = np.zeros((4, 3, 3, 3), dtype=np.float32)
    for c in range(3):
        w1[0, c] = LUMA[c] * SOBEL_X
        w1[1, c] = -LUMA[c] * SOBEL_X
        w1[2, c] = LUMA[c] * SOBEL_Y
        w1[3, c] = -LUMA[c] * SOBEL_Y
    w2 = np.ones((1, 4, 1, 1), dtype=np.float32)
    for name, arr in (("W1", w1), ("B1", np.zeros(4, np.float32)), ("W2", w2), ("B2", np.zeros(1, np.float32))):
        model.graph.initializer.append(numpy_helper.from_array(arr, name))
    onnx.checker.check_model(model)
    return model


def calibration_images(n: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    out = np.zeros((n, 1, 3, H, W), dtype=np.float32)
    for i in range(n):
        # smooth colour field: coarse noise, upsampled
        coarse = rng.rand(3, H // 15 + 1, W // 15 + 1).astype(np.float32)
        img = np.kron(coarse, np.ones((15, 15), dtype=np.float32))[:, :H, :W]
        for _ in range(rng.randint(3, 8)):  # hard-edged rectangles
            y0, x0 = rng.randint(0, H - 10), rng.randint(0, W - 10)
            y1, x1 = y0 + rng.randint(8, 60), x0 + rng.randint(8, 100)
            img[:, y0:y1, x0:x1] = rng.rand(3, 1, 1)
        out[i, 0] = np.clip(img + 0.02 * rng.randn(3, H, W), 0, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = a.out_dir / "edge.onnx"
    onnx.save(build_model(), onnx_path)
    calib = calibration_images()
    kmodel = convert_to_kmodel(onnx_path, samples_count=len(calib), calibration_data=calib)
    (a.out_dir / "edge.kmodel").write_bytes(kmodel)
    print(f"wrote {onnx_path} and {a.out_dir / 'edge.kmodel'} ({len(kmodel)} bytes)")


if __name__ == "__main__":
    main()
