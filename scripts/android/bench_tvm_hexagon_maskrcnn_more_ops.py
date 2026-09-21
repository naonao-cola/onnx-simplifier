#!/usr/bin/env python3
"""Benchmark the remaining Mask R-CNN operators (backbone eltwise, box head, RPN/box
post-processing) on a Hexagon DSP.

Complements `bench_tvm_hexagon_maskrcnn_ops.py` (Conv/MaxPool/Resize/RoiAlign/ConvTranspose/QDQ).
Static shapes (residual Add/Relu feature maps, box-head MatMul weights, the NMS IoU threshold)
come from the model; the data-dependent box counts are set with --proposals/--roi-batch.
Every kernel is checked against a NumPy reference on the phone before it is timed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
import setuptools  # noqa: F401  # TVM 0.17 imports distutils during module initialization.
import tvm
from tvm import te, topi
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.contrib.hexagon.tools import register_linker
from tvm.rpc.tracker import Tracker

ALL_OPS = (
    "add,relu,add_relu,matmul,matmul_hvx,softmax,sigmoid,sigmoid_poly,box_decode,level_mapper,filter,topk,nms,"
    "nonzero,gather,scatter,concat,split,transpose,flatten,reducemin"
)
# Operators that are pure shape/metadata glue on tiny tensors (no DSP kernel worth timing).
METADATA_ONLY = "Shape, Unsqueeze, Squeeze, Reshape, ConstantOfShape, Expand, Cast(int<->int)"


def _configure_linker():
    toolchain = os.environ.get("HEXAGON_TOOLCHAIN")
    if not toolchain:
        return
    clang_link = Path(toolchain) / "bin" / "hexagon-clang++"
    wrapper = Path("/tmp/tvm-hexagon-link-wrapper")
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import subprocess, sys\n"
        f"clang = {str(clang_link)!r}\n"
        "args = ['-Wl,--export-dynamic' if x == '-export-dynamic' else x for x in sys.argv[1:]]\n"
        # Kernels only use C symbols; a dynamic libc++ dependency crashes on the DSP (see
        # relink_hexagon_skel_static_libcxx.sh).
        "raise SystemExit(subprocess.call([clang, '-nostdlib++', *args]))\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    register_linker(lambda: str(wrapper))


def _hexagon_target():
    target = tvm.target.hexagon("v73")
    return tvm.target.Target(target, host=target)


def _model_shapes(model_path: Path):
    """Static residual Add/Relu feature maps, MatMul weight shapes and the NMS IoU threshold."""
    model = onnx.load(str(model_path))
    for value in model.graph.input:
        dims = value.type.tensor_type.shape.dim
        if len(dims) == 3:
            for dim, size in zip(dims, (3, 224, 224)):
                dim.dim_value = size
                dim.ClearField("dim_param")
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    shapes = {}
    for value in list(model.graph.value_info) + list(model.graph.input):
        shapes[value.name] = tuple(
            dim.dim_value if dim.HasField("dim_value") else None
            for dim in value.type.tensor_type.shape.dim
        )
    initializers = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    feature_maps, weights, iou = [], [], None
    for node in model.graph.node:
        if node.op_type in ("Add", "Relu"):
            shape = shapes.get(node.input[0])
            if shape and len(shape) == 4 and None not in shape and shape not in feature_maps:
                feature_maps.append(shape)
        elif node.op_type == "MatMul":
            # QDQ models feed MatMul from a DequantizeLinear, so use the inferred shape.
            weight = shapes.get(node.input[1])
            if weight is None and node.input[1] in initializers:
                weight = initializers[node.input[1]].shape
            if weight and len(weight) == 2 and None not in weight:
                weights.append(tuple(weight))
        elif node.op_type == "NonMaxSuppression" and len(node.input) > 3 and iou is None:
            threshold = initializers.get(node.input[3])  # first NMS is the RPN's (IoU 0.7)
            if threshold is not None:
                iou = float(np.asarray(threshold).reshape(-1)[0])
    if not feature_maps or not weights:
        raise RuntimeError("Could not find residual feature maps / box-head MatMul weights")
    return feature_maps, weights, 0.7 if iou is None else iou


def _random_boxes(rng, count, extent=224.0):
    xy = rng.uniform(0, extent - 40, (count, 2))
    wh = rng.uniform(8, 140, (count, 2))
    return np.concatenate([xy, np.minimum(xy + wh, extent)], axis=1).astype("float32")


def _schedule(outputs):
    return topi.hexagon.schedule_injective(outputs)


def _placeholder(shape, name, dtype="float32"):
    return te.placeholder(shape, name=name, dtype=dtype)


def _build(outputs, args, target, name="main"):
    schedule = _schedule(outputs)
    return tvm.build(schedule, list(args) + list(outputs), target=target, name=name)


# ---- operator workloads: each returns (label, module, arrays, output_specs, compare) ----


def _eltwise(kind, shape, rng, target):
    a = _placeholder(shape, "a")
    b = _placeholder(shape, "b")
    if kind == "add":
        out, args = topi.add(a, b), [a, b]
    elif kind == "add_relu":
        out, args = topi.nn.relu(topi.add(a, b)), [a, b]
    else:
        out, args = topi.nn.relu(a), [a]
    arrays = [rng.normal(size=shape).astype("float32") for _ in args]
    expected = arrays[0] + arrays[1] if kind != "relu" else arrays[0]
    if kind != "add":
        expected = np.maximum(expected, 0)
    label = f"{kind}_{'x'.join(map(str, shape))}"
    return (
        label,
        _build([out], args, target),
        arrays,
        [(shape, "float32")],
        lambda got: np.testing.assert_allclose(got[0], expected, rtol=1e-6, atol=1e-6),
    )


def _matmul(k, m, roi_batch, relu, rng, target):
    """Box-head fully connected layer: [N, K] @ [K, M] + bias (+ Relu)."""
    x = _placeholder((roi_batch, k), "x")
    w = _placeholder((k, m), "w")
    bias = _placeholder((m,), "bias")
    rk = te.reduce_axis((0, k), name="rk")
    acc = te.compute(
        (roi_batch, m), lambda n, j: te.sum(x[n, rk] * w[rk, j], axis=rk), name="acc"
    )

    def epilogue(n, j):
        value = acc[n, j] + bias[j]
        return tvm.tir.max(value, tvm.tir.const(0, "float32")) if relu else value

    out = te.compute((roi_batch, m), epilogue, name="fc")
    schedule = te.create_schedule(out.op)
    n_axis, j_axis = schedule[acc].op.axis
    j_outer, j_inner = schedule[acc].split(j_axis, factor=32)
    schedule[acc].reorder(j_outer, n_axis, rk, j_inner)
    schedule[acc].vectorize(j_inner)
    schedule[acc].parallel(j_outer)
    schedule[out].compute_root()
    module = tvm.build(schedule, [x, w, bias, out], target=target, name="main")
    arrays = [
        rng.normal(0, 0.1, (roi_batch, k)).astype("float32"),
        rng.normal(0, 0.02, (k, m)).astype("float32"),
        rng.normal(0, 0.1, (m,)).astype("float32"),
    ]
    expected = arrays[0] @ arrays[1] + arrays[2]
    if relu:
        expected = np.maximum(expected, 0)
    return (
        f"matmul_{roi_batch}x{k}x{m}{'_relu' if relu else ''}",
        module,
        arrays,
        [((roi_batch, m), "float32")],
        lambda got: np.testing.assert_allclose(got[0], expected, rtol=2e-4, atol=2e-4),
    )


def _matmul_hvx(k, m, roi_batch, relu, rng, target, lanes=32, vectors=2, unroll=4):
    """Box-head FC with hand-written HVX qf32 accumulation (see bench_tvm_hexagon_conv_transpose).

    Weights [K, M] are zero-padded on the host to a multiple of `lanes * vectors` output columns
    (done once at model load) and each parallel task owns one column tile: every weight vector is
    loaded once and reused by all `rows` activation rows kept in accumulator registers.
    """
    from bench_tvm_hexagon_conv_transpose import _qf32_intrin

    rows = min(roi_batch, 8)
    assert roi_batch % rows == 0 and k % unroll == 0
    span = lanes * vectors
    m_pad = -(-m // span) * span
    x = _placeholder((roi_batch, k), "x")
    w = _placeholder((k, m_pad), "w")
    bias = _placeholder((m_pad,), "bias")

    def body(ins, outs):
        ib = tvm.tir.ir_builder.create()

        def flat(buf):
            size = int(np.prod([int(d) for d in buf.shape]))
            return ib.buffer_ptr(tvm.tir.decl_buffer((size,), buf.dtype, data=buf.data))

        x_buf, w_buf, b_buf = (flat(b) for b in ins)
        out_buf = flat(outs[0])
        acc = ib.allocate("int32", (rows * vectors * lanes,), name="acc", scope="local")
        zero = tvm.tir.Broadcast(tvm.tir.const(0, "int32"), lanes)

        def slot(r, v):
            return tvm.tir.Ramp((r * vectors + v) * lanes, 1, lanes)

        with ib.for_range(0, m_pad // span, name="tile", kind="parallel") as tile:
            m0 = tile * span
            with ib.for_range(0, roi_batch // rows, name="row_block") as row_block:
                for r in range(rows):
                    for v in range(vectors):
                        acc[slot(r, v)] = zero
                with ib.for_range(0, k // unroll, name="k_block") as k_block:
                    for u in range(unroll):
                        kk = k_block * unroll + u
                        weights = [
                            tvm.tir.reinterpret(
                                "int32x32",
                                w_buf[tvm.tir.Ramp(kk * m_pad + m0 + v * lanes, 1, lanes)],
                            )
                            for v in range(vectors)
                        ]
                        for r in range(rows):
                            splat = tvm.tir.reinterpret(
                                "int32x32",
                                tvm.tir.Broadcast(x_buf[(row_block * rows + r) * k + kk], lanes),
                            )
                            for v in range(vectors):
                                prod = _qf32_intrin("int32x32", "vmpy.qf32.sf", splat, weights[v])
                                acc[slot(r, v)] = _qf32_intrin(
                                    "int32x32", "vadd.qf32", acc[slot(r, v)], prod
                                )
                for r in range(rows):
                    for v in range(vectors):
                        value = tvm.tir.reinterpret(
                            "float32x32", _qf32_intrin("int32x32", "vconv.sf.qf32", acc[slot(r, v)])
                        ) + b_buf[tvm.tir.Ramp(m0 + v * lanes, 1, lanes)]
                        if relu:
                            value = tvm.te.max(
                                value, tvm.tir.Broadcast(tvm.tir.const(0.0, "float32"), lanes)
                            )
                        out_buf[
                            tvm.tir.Ramp((row_block * rows + r) * m_pad + m0 + v * lanes, 1, lanes)
                        ] = value
        return ib.get()

    out = te.extern((roi_batch, m_pad), [x, w, bias], body, name="fc_hvx", dtype="float32")
    schedule = te.create_schedule(out.op)
    binds = {
        t: tvm.tir.decl_buffer(t.shape, t.dtype, name=t.op.name, data_alignment=128, offset_factor=1)
        for t in (x, w, bias, out)
    }
    module = tvm.build(schedule, [x, w, bias, out], target=target, name="main", binds=binds)
    weight = np.zeros((k, m_pad), dtype="float32")
    weight[:, :m] = rng.normal(0, 0.02, (k, m))
    bias_array = np.zeros(m_pad, dtype="float32")
    bias_array[:m] = rng.normal(0, 0.1, m)
    x_array = rng.normal(0, 0.1, (roi_batch, k)).astype("float32")
    expected = x_array @ weight + bias_array
    if relu:
        expected = np.maximum(expected, 0)
    return (
        f"matmul_hvx_{roi_batch}x{k}x{m}{'_relu' if relu else ''}",
        module,
        [x_array, weight, bias_array],
        [((roi_batch, m_pad), "float32")],
        lambda got: np.testing.assert_allclose(
            got[0][:, :m], expected[:, :m], rtol=2e-4, atol=2e-4
        ),
    )


def _softmax(roi_batch, classes, rng, target):
    x = _placeholder((roi_batch, classes), "x")
    out = topi.nn.softmax(x, axis=1)
    schedule = te.create_schedule(out.op)
    module = tvm.build(schedule, [x, out], target=target, name="main")
    array = rng.normal(0, 2, (roi_batch, classes)).astype("float32")
    shifted = np.exp(array - array.max(axis=1, keepdims=True))
    expected = shifted / shifted.sum(axis=1, keepdims=True)
    return (
        f"softmax_{roi_batch}x{classes}",
        module,
        [array],
        [((roi_batch, classes), "float32")],
        lambda got: np.testing.assert_allclose(got[0], expected, rtol=1e-4, atol=1e-6),
    )


def _sigmoid(roi_batch, classes, rng, target):
    shape = (roi_batch, classes, 28, 28)
    x = _placeholder(shape, "x")
    out = topi.sigmoid(x)
    array = rng.normal(0, 3, shape).astype("float32")
    expected = 1 / (1 + np.exp(-array))
    return (
        f"sigmoid_mask_logits_{'x'.join(map(str, shape))}",
        _build([out], [x], target),
        [array],
        [(shape, "float32")],
        lambda got: np.testing.assert_allclose(got[0], expected, rtol=1e-4, atol=1e-5),
    )


def _fast_exp(x):
    """exp(x) for x <= 0 without libm: 2^(x*log2e) = 2^n * poly(f), all vectorizable."""
    t = tvm.te.max(x * 1.4426950408889634, -126.0)
    # floor(t) from an int cast: te.floor lowers to one scalar floorf call per lane.
    trunc = tvm.tir.Cast("int32", t)
    trunc = trunc - tvm.tir.Select(tvm.tir.Cast("float32", trunc) > t, 1, 0)
    n = tvm.tir.Cast("float32", trunc)
    f = t - n
    # 2^f on [0, 1): degree-5 minimax-style polynomial (rel. error ~2e-7).
    poly = 1.0 + f * (0.6931472 + f * (0.2402265 + f * (0.0555041 + f * (0.0096181 + f * 0.0013333))))
    scale = tvm.tir.reinterpret("float32", (trunc + 127) << 23)
    return poly * scale


def _fast_reciprocal(d):
    """1/d for d in [1, 2]: magic-number seed plus three Newton steps (no divide on HVX)."""
    seed = tvm.tir.reinterpret("float32", tvm.tir.const(0x7EF311C7, "int32") - tvm.tir.reinterpret("int32", d))
    for _ in range(3):
        seed = seed * (2.0 - d * seed)
    return seed


def _sigmoid_poly(roi_batch, classes, rng, target):
    """Mask-logit Sigmoid with a polynomial exp + Newton reciprocal instead of libm expf."""
    shape = (roi_batch, classes, 28, 28)
    x = _placeholder(shape, "x")

    def sigmoid(*idx):
        v = x[idx]
        magnitude = tvm.te.max(v, -v)  # te.abs lowers to a scalar fabs call on Hexagon
        positive = _fast_reciprocal(1.0 + _fast_exp(-magnitude))
        # Select (not if_then_else): lazy branches would keep the loop scalar.
        return tvm.tir.Select(v >= 0.0, positive, 1.0 - positive)

    out = te.compute(shape, sigmoid, name="sigmoid_poly")
    array = rng.normal(0, 3, shape).astype("float32")
    expected = 1 / (1 + np.exp(-array))
    return (
        f"sigmoid_poly_mask_logits_{'x'.join(map(str, shape))}",
        _build([out], [x], target),
        [array],
        [(shape, "float32")],
        lambda got: np.testing.assert_allclose(got[0], expected, rtol=1e-4, atol=1e-5),
    )


def _box_decode(count, rng, target):
    """torchvision BoxCoder.decode: weights (10, 10, 5, 5), dw/dh clipped to log(1000/16)."""
    clip = float(np.log(1000.0 / 16))
    deltas = _placeholder((count, 4), "deltas")
    anchors = _placeholder((count, 4), "anchors")

    def decode(k, j):
        width = anchors[k, 2] - anchors[k, 0]
        height = anchors[k, 3] - anchors[k, 1]
        center_x = anchors[k, 0] + width * 0.5
        center_y = anchors[k, 1] + height * 0.5
        pred_cx = deltas[k, 0] / 10.0 * width + center_x
        pred_cy = deltas[k, 1] / 10.0 * height + center_y
        pred_w = te.exp(tvm.te.min(deltas[k, 2] / 5.0, clip)) * width
        pred_h = te.exp(tvm.te.min(deltas[k, 3] / 5.0, clip)) * height
        return tvm.tir.if_then_else(
            j == 0,
            pred_cx - pred_w * 0.5,
            tvm.tir.if_then_else(
                j == 1,
                pred_cy - pred_h * 0.5,
                tvm.tir.if_then_else(j == 2, pred_cx + pred_w * 0.5, pred_cy + pred_h * 0.5),
            ),
        )

    out = te.compute((count, 4), decode, name="decoded")
    anchor_array = _random_boxes(rng, count)
    delta_array = rng.normal(0, 1.0, (count, 4)).astype("float32")
    width = anchor_array[:, 2] - anchor_array[:, 0]
    height = anchor_array[:, 3] - anchor_array[:, 1]
    cx = anchor_array[:, 0] + 0.5 * width
    cy = anchor_array[:, 1] + 0.5 * height
    pcx = delta_array[:, 0] / 10 * width + cx
    pcy = delta_array[:, 1] / 10 * height + cy
    pw = np.exp(np.minimum(delta_array[:, 2] / 5, clip)) * width
    ph = np.exp(np.minimum(delta_array[:, 3] / 5, clip)) * height
    expected = np.stack([pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2], axis=1)
    return (
        f"box_decode_{count}",
        _build([out], [deltas, anchors], target),
        [delta_array, anchor_array],
        [((count, 4), "float32")],
        lambda got: np.testing.assert_allclose(got[0], expected, rtol=1e-4, atol=1e-3),
    )


def _level_mapper(count, rng, target):
    """FPN level assignment: floor(4 + log2(sqrt(area) / 224)), clamped to [2, 5]."""
    boxes = _placeholder((count, 4), "boxes")

    def level(k):
        area = (boxes[k, 2] - boxes[k, 0]) * (boxes[k, 3] - boxes[k, 1])
        value = te.floor(4.0 + te.log(te.sqrt(area) / 224.0 + 1e-6) / float(np.log(2.0)))
        return tvm.tir.Cast("int32", tvm.te.max(tvm.te.min(value, 5.0), 2.0))

    out = te.compute((count,), level, name="level")
    array = _random_boxes(rng, count)
    area = (array[:, 2] - array[:, 0]) * (array[:, 3] - array[:, 1])
    expected = np.clip(np.floor(4 + np.log2(np.sqrt(area) / 224 + 1e-6)), 2, 5).astype("int32")

    def compare(got):
        # Values sitting exactly on a log2 boundary may round differently in float32.
        assert np.mean(got[0] != expected) < 2e-3, np.mean(got[0] != expected)

    return f"level_mapper_{count}", _build([out], [boxes], target), [array], [((count,), "int32")], compare


def _filter(count, rng, target):
    """Score/size filter: Greater & Not(Less) & Not(Less) (+ Cast to a uint8 mask)."""
    scores = _placeholder((count,), "scores")
    boxes = _placeholder((count, 4), "boxes")

    def keep(k):
        wide = (boxes[k, 2] - boxes[k, 0]) >= 1e-3
        tall = (boxes[k, 3] - boxes[k, 1]) >= 1e-3
        return tvm.tir.Cast("uint8", tvm.tir.all(scores[k] > 0.05, wide, tall))

    out = te.compute((count,), keep, name="keep")
    score_array = rng.uniform(0, 1, count).astype("float32")
    box_array = _random_boxes(rng, count)
    box_array[::17, 2] = box_array[::17, 0]  # degenerate boxes must be dropped
    expected = (
        (score_array > 0.05)
        & (box_array[:, 2] - box_array[:, 0] >= 1e-3)
        & (box_array[:, 3] - box_array[:, 1] >= 1e-3)
    ).astype("uint8")
    return (
        f"score_size_filter_{count}",
        _build([out], [scores, boxes], target),
        [score_array, box_array],
        [((count,), "uint8")],
        lambda got: np.testing.assert_array_equal(got[0], expected),
    )


def _topk(candidates, k, rng, target):
    """RPN pre-NMS top-k over the objectness scores of one FPN level."""
    scores = _placeholder((1, candidates), "scores")
    values, indices = topi.topk(scores, k=k, axis=1, ret_type="both", is_ascend=False, dtype="int32")
    schedule = te.create_schedule([values.op, indices.op])
    module = tvm.build(schedule, [scores, values, indices], target=target, name="main")
    array = rng.permutation(candidates).astype("float32").reshape(1, candidates)
    order = np.argsort(-array[0], kind="stable")[:k]

    def compare(got):
        np.testing.assert_array_equal(got[0][0], array[0][order])
        np.testing.assert_array_equal(got[1][0], order)

    return (
        f"topk_{candidates}_to_{k}",
        module,
        [array],
        [((1, k), "float32"), ((1, k), "int32")],
        compare,
    )


def _nms(count, iou_threshold, rng, target):
    """Greedy NMS over score-sorted boxes; emits a 0/1 keep mask (O(count^2) IoU tests)."""
    boxes = _placeholder((count, 4), "boxes")

    def body(ins, outs):
        ib = tvm.tir.ir_builder.create()
        box = ib.buffer_ptr(
            tvm.tir.decl_buffer((count * 4,), "float32", data=ins[0].data)
        )
        keep = ib.buffer_ptr(tvm.tir.decl_buffer((count,), "int32", data=outs[0].data))
        with ib.for_range(0, count, name="init") as i:
            keep[i] = tvm.tir.const(1, "int32")
        with ib.for_range(0, count, name="i") as i:
            with ib.if_scope(keep[i] == 1):
                area_i = (box[i * 4 + 2] - box[i * 4]) * (box[i * 4 + 3] - box[i * 4 + 1])
                with ib.for_range(0, count - i - 1, name="jj") as jj:
                    j = i + 1 + jj
                    with ib.if_scope(keep[j] == 1):
                        inter_w = tvm.te.max(
                            tvm.te.min(box[i * 4 + 2], box[j * 4 + 2])
                            - tvm.te.max(box[i * 4], box[j * 4]),
                            0.0,
                        )
                        inter_h = tvm.te.max(
                            tvm.te.min(box[i * 4 + 3], box[j * 4 + 3])
                            - tvm.te.max(box[i * 4 + 1], box[j * 4 + 1]),
                            0.0,
                        )
                        inter = inter_w * inter_h
                        area_j = (box[j * 4 + 2] - box[j * 4]) * (
                            box[j * 4 + 3] - box[j * 4 + 1]
                        )
                        union = area_i + area_j - inter
                        with ib.if_scope(inter > iou_threshold * union):
                            keep[j] = tvm.tir.const(0, "int32")
        return ib.get()

    out = te.extern((count,), [boxes], body, name="nms_keep", dtype="int32")
    schedule = te.create_schedule(out.op)
    module = tvm.build(schedule, [boxes, out], target=target, name="main")
    # Cluster boxes so a fair share are suppressed.
    centers = rng.uniform(20, 200, (max(count // 8, 1), 2))
    picks = rng.integers(0, len(centers), count)
    xy = centers[picks] + rng.normal(0, 6, (count, 2))
    wh = rng.uniform(30, 60, (count, 2))
    array = np.concatenate([xy, xy + wh], axis=1).astype("float32")
    keep = np.ones(count, dtype=bool)
    areas = (array[:, 2] - array[:, 0]) * (array[:, 3] - array[:, 1])
    for i in range(count):
        if not keep[i]:
            continue
        iw = np.maximum(np.minimum(array[i, 2], array[i + 1 :, 2]) - np.maximum(array[i, 0], array[i + 1 :, 0]), 0)
        ih = np.maximum(np.minimum(array[i, 3], array[i + 1 :, 3]) - np.maximum(array[i, 1], array[i + 1 :, 1]), 0)
        inter = iw * ih
        keep[i + 1 :] &= ~(inter > iou_threshold * (areas[i] + areas[i + 1 :] - inter))
    expected = keep.astype("int32")

    def compare(got):
        # Float32 rounding at IoU == threshold can flip a rare pair; allow a tiny mismatch.
        assert np.mean(got[0] != expected) < 5e-3, np.mean(got[0] != expected)

    return f"nms_{count}_iou{iou_threshold:g}", module, [array], [((count,), "int32")], compare


def _nonzero(count, rng, target):
    """NonZero on a candidate mask: compacted indices plus the element count."""
    mask = _placeholder((count,), "mask", "uint8")

    def body(ins, outs):
        ib = tvm.tir.ir_builder.create()
        src = ib.buffer_ptr(tvm.tir.decl_buffer((count,), "uint8", data=ins[0].data))
        indices = ib.buffer_ptr(tvm.tir.decl_buffer((count,), "int64", data=outs[0].data))
        total = ib.buffer_ptr(tvm.tir.decl_buffer((1,), "int64", data=outs[1].data))
        cursor = ib.allocate("int64", (1,), name="cursor", scope="local")
        cursor[0] = tvm.tir.const(0, "int64")
        with ib.for_range(0, count, name="i") as i:
            with ib.if_scope(src[i] != 0):
                indices[cursor[0]] = tvm.tir.Cast("int64", i)
                cursor[0] = cursor[0] + 1
        total[0] = cursor[0]
        return ib.get()

    indices, total = te.extern(
        [(count,), (1,)], [mask], body, name="nonzero", dtype=["int64", "int64"]
    )
    schedule = te.create_schedule([indices.op])
    module = tvm.build(schedule, [mask, indices, total], target=target, name="main")
    array = (rng.uniform(0, 1, count) > 0.7).astype("uint8")
    expected = np.flatnonzero(array)

    def compare(got):
        n = int(got[1][0])
        assert n == len(expected), (n, len(expected))
        np.testing.assert_array_equal(got[0][:n], expected)

    return (
        f"nonzero_{count}",
        module,
        [array],
        [((count,), "int64"), ((1,), "int64")],
        compare,
    )


def _gather(total, count, rng, target):
    """Gather box rows by candidate indices (ONNX int64 indices)."""
    data = _placeholder((total, 4), "boxes")
    indices = _placeholder((count,), "indices", "int64")
    out = topi.take(data, indices, axis=0)
    array = _random_boxes(rng, total)
    index = rng.integers(0, total, count).astype("int64")
    return (
        f"gather_rows_{total}_by_{count}",
        _build([out], [data, indices], target),
        [array, index],
        [((count, 4), "float32")],
        lambda got: np.testing.assert_array_equal(got[0], array[index]),
    )


def _scatter(roi_batch, rng, target):
    """ScatterElements used to merge per-FPN-level RoI features: [N,256,7,7] along axis 0."""
    shape = (roi_batch, 256, 7, 7)
    rows = roi_batch // 2
    data = _placeholder(shape, "data")
    indices = _placeholder((rows, 256, 7, 7), "indices", "int64")
    updates = _placeholder((rows, 256, 7, 7), "updates")
    out = topi.scatter_elements(data, indices, updates, axis=0)
    schedule = te.create_schedule(out.op)
    module = tvm.build(schedule, [data, indices, updates, out], target=target, name="main")
    row_ids = rng.permutation(roi_batch)[:rows].astype("int64")
    index_array = np.broadcast_to(row_ids[:, None, None, None], (rows, 256, 7, 7)).copy()
    data_array = np.zeros(shape, dtype="float32")
    update_array = rng.normal(size=(rows, 256, 7, 7)).astype("float32")
    expected = data_array.copy()
    np.put_along_axis(expected, index_array, update_array, axis=0)
    return (
        f"scatter_elements_{'x'.join(map(str, shape))}",
        module,
        [data_array, index_array, update_array],
        [(shape, "float32")],
        lambda got: np.testing.assert_array_equal(got[0], expected),
    )


def _concat(rng, target):
    """RPN candidate concat across the five FPN levels (3 anchors per position, 224px input)."""
    sizes = [3 * side * side for side in (56, 28, 14, 7, 4)]
    parts = [_placeholder((size, 4), f"level{i}") for i, size in enumerate(sizes)]
    out = topi.concatenate(parts, axis=0)
    arrays = [rng.normal(size=(size, 4)).astype("float32") for size in sizes]
    expected = np.concatenate(arrays, axis=0)
    return (
        f"concat_rpn_levels_{sum(sizes)}x4",
        _build([out], parts, target),
        arrays,
        [((sum(sizes), 4), "float32")],
        lambda got: np.testing.assert_array_equal(got[0], expected),
    )


def _split(count, rng, target):
    """Split [K,4] boxes into four [K,1] columns (Split / Slice on box coordinates)."""
    boxes = _placeholder((count, 4), "boxes")
    outs = [topi.strided_slice(boxes, [0, i], [count, i + 1], [1, 1]) for i in range(4)]
    array = _random_boxes(rng, count)
    return (
        f"split_boxes_{count}",
        _build(outs, [boxes], target),
        [array],
        [((count, 1), "float32")] * 4,
        lambda got: [np.testing.assert_array_equal(got[i], array[:, i : i + 1]) for i in range(4)],
    )


def _transpose(count, rng, target):
    x = _placeholder((1, count), "x")
    out = topi.transpose(x, (1, 0))
    array = rng.normal(size=(1, count)).astype("float32")
    return (
        f"transpose_1x{count}",
        _build([out], [x], target),
        [array],
        [((count, 1), "float32")],
        lambda got: np.testing.assert_array_equal(got[0], array.T),
    )


def _flatten(roi_batch, rng, target):
    """RoI features [N,256,7,7] -> [N,12544] (box-head input)."""
    shape = (roi_batch, 256, 7, 7)
    x = _placeholder(shape, "x")
    out = topi.reshape(x, (roi_batch, 256 * 7 * 7))
    array = rng.normal(size=shape).astype("float32")
    return (
        f"flatten_{'x'.join(map(str, shape))}",
        _build([out], [x], target),
        [array],
        [((roi_batch, 256 * 7 * 7), "float32")],
        lambda got: np.testing.assert_array_equal(got[0], array.reshape(roi_batch, -1)),
    )


def _reducemin(count, rng, target):
    x = _placeholder((count,), "x")
    out = topi.min(x, axis=0, keepdims=True)
    schedule = te.create_schedule(out.op)
    module = tvm.build(schedule, [x, out], target=target, name="main")
    array = rng.normal(size=count).astype("float32")
    return (
        f"reducemin_{count}",
        module,
        [array],
        [((1,), "float32")],
        lambda got: np.testing.assert_array_equal(got[0], array.min(keepdims=True)),
    )


def _workloads(op, model_info, args, rng, target):
    feature_maps, weights, iou = model_info
    n, k = args.roi_batch, args.proposals
    if op in ("add", "relu", "add_relu"):
        return [_eltwise(op, shape, rng, target) for shape in feature_maps]
    if op == "matmul":
        cases = []
        for index, (in_dim, out_dim) in enumerate(weights):
            cases.append(_matmul(in_dim, out_dim, n, relu=index < 2, rng=rng, target=target))
        return cases
    if op == "matmul_hvx":
        return [
            _matmul_hvx(in_dim, out_dim, n, relu=index < 2, rng=rng, target=target)
            for index, (in_dim, out_dim) in enumerate(weights)
        ]
    builders = {
        "sigmoid_poly": lambda: [_sigmoid_poly(n, 81, rng, target)],
        "softmax": lambda: [_softmax(n, 81, rng, target)],
        "sigmoid": lambda: [_sigmoid(n, 81, rng, target)],
        "box_decode": lambda: [_box_decode(k, rng, target)],
        "level_mapper": lambda: [_level_mapper(k, rng, target)],
        "filter": lambda: [_filter(3 * 56 * 56, rng, target)],
        "topk": lambda: [_topk(3 * 56 * 56, k, rng, target)],
        "nms": lambda: [_nms(k, iou, rng, target)],
        "nonzero": lambda: [_nonzero(3 * 56 * 56, rng, target)],
        "gather": lambda: [_gather(3 * 56 * 56, k, rng, target)],
        "scatter": lambda: [_scatter(n, rng, target)],
        "concat": lambda: [_concat(rng, target)],
        "split": lambda: [_split(k, rng, target)],
        "transpose": lambda: [_transpose(k, rng, target)],
        "flatten": lambda: [_flatten(n, rng, target)],
        "reducemin": lambda: [_reducemin(k, rng, target)],
    }
    if op not in builders:
        raise ValueError(f"Unknown operator: {op}")
    return builders[op]()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="239dbd8f")
    parser.add_argument("--roi-batch", type=int, default=8)
    parser.add_argument("--proposals", type=int, default=1000)
    parser.add_argument("--ops", default=ALL_OPS)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    _configure_linker()
    model_info = _model_shapes(args.model)
    target = _hexagon_target()
    tracker = Tracker(host="127.0.0.1", port=9198)
    launcher = HexagonLauncher(
        args.device,
        rpc_info={
            "rpc_tracker_host": "127.0.0.1",
            "rpc_tracker_port": 9198,
            "rpc_server_port": 7078,
            "workspace_base": "/data/local/tmp/tvm_hexagon_more_ops",
            "adb_server_socket": None,
        },
    )
    rng = np.random.default_rng(53)
    failures = []
    try:
        launcher.start_server()
        for op in args.ops.split(","):
            try:
                workloads = _workloads(op, model_info, args, rng, target)
            except Exception as error:  # noqa: BLE001 - report unsupported ops, keep going
                failures.append((op, f"build: {type(error).__name__}: {str(error).splitlines()[-1][:160]}"))
                print(f"{op}: BUILD FAILED ({failures[-1][1]})", flush=True)
                continue
            for name, module, arrays, output_specs, compare in workloads:
                path = Path("/tmp") / f"tvm_hexagon_maskrcnn_more_{name}.so"
                try:
                    module.save(str(path))
                    with launcher.create_session() as session:
                        dsp = session.device
                        remote = session.load_module(session.upload(str(path), path.name))
                        inputs = [tvm.nd.array(array, dsp) for array in arrays]
                        outputs = [tvm.nd.empty(shape, dtype, dsp) for shape, dtype in output_specs]
                        remote["main"](*inputs, *outputs)
                        got = [output.numpy() for output in outputs]
                        compare(got)
                        samples = remote.time_evaluator(
                            "main", dsp, number=1, repeat=args.repeat
                        )(*inputs, *outputs).results
                    print(
                        f"{name}: median={np.median(samples) * 1e3:.3f} ms, correctness=PASS",
                        flush=True,
                    )
                except Exception as error:  # noqa: BLE001
                    failures.append((name, f"{type(error).__name__}: {str(error).splitlines()[-1][:160]}"))
                    print(f"{name}: FAILED ({failures[-1][1]})", flush=True)
    finally:
        launcher.stop_server()
        tracker.terminate()
    if failures:
        print("\nFailures:")
        for name, reason in failures:
            print(f"  {name}: {reason}")
    print(f"\nMetadata-only operators not benchmarked: {METADATA_ONLY}")


if __name__ == "__main__":
    main()
