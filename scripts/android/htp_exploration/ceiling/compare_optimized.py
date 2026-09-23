#!/usr/bin/env python3
"""Compare the optimized backbone's raw uint8 HTP outputs against the *original* backbone.onnx on
host ONNX Runtime CPU (the true reference), converting back to the original output form:
dequantize, and for the RPN heads redo the per-anchor layout (+ sigmoid for scores).

usage: compare_optimized.py <orig backbone.onnx> <optimized.onnx> <fp32 input.bin> <out_prefix>
       (out_prefix_o<i>.bin pulled from the phone, as written by qnn_run)
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


def main():
    orig, opt, inp, prefix = sys.argv[1:5]
    x = np.fromfile(inp, np.float32).reshape(3, 800, 1088)
    so = ort.InferenceSession(orig, providers=["CPUExecutionProvider"])
    ref = dict(zip([o.name for o in so.get_outputs()], so.run(None, {"image": x})))
    g = onnx.load(opt).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    # map each optimized output to the original output with the same element count/role
    converted = {}
    ref_by_size = {}
    for k, v in ref.items():
        ref_by_size.setdefault(v.size, []).append(k)
    for i, o in enumerate(g.output):
        shp = [d.dim_value for d in o.type.tensor_type.shape.dim]
        q = prod[o.name]
        s, zp = float(init[q.input[1]]), float(init[q.input[2]])
        v = (np.fromfile(f"{prefix}_o{i}.bin", np.uint8).reshape(shp).astype(np.float32) - zp) * s
        n, c, h, w = shp
        if c == 256:
            cand = [k for k in ref_by_size[v.size] if ref[k].shape == v.shape]
            got, kind = v, "fpn"
        elif c == 12:
            got = v.reshape(n, 3, 4, h, w).transpose(0, 3, 4, 1, 2).reshape(n, -1, 4)
            cand, kind = [k for k in ref if ref[k].shape == got.shape], "deltas"
        else:
            got = 1 / (1 + np.exp(-v.transpose(0, 2, 3, 1).reshape(n, -1)))
            cand, kind = [k for k in ref if ref[k].shape == got.shape], "scores"
        (k,) = cand
        d = np.abs(got - ref[k])
        r = ref[k]
        print(f"{kind:6} {k[:22]:22} {str(r.shape):18} maxabs={d.max():.4g} meanabs={d.mean():.3g} "
              f"range=[{r.min():.3g},{r.max():.3g}]")
        converted[k] = got.astype(np.float32)
    # also write them in the original model's output order/form, for ../qnn_shell/detect_compare.py
    for i, o in enumerate(so.get_outputs()):
        converted[o.name].tofile(f"{prefix}_asorig_o{i}.bin")


if __name__ == "__main__":
    main()
