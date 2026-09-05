"""Tests for ``onnxsim.apply_brecq`` (BRECQ, see ``onnxsim/brecq.py``) --
jointly optimizes every quantized MatMul/Gemm layer inside a caller-
delimited block against the *block's own final output* reconstruction
error (Fisher-diagonal weighted), instead of each layer's own output
independently as :mod:`onnxsim.adaround` does.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=21, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _residual_block_model(D=32, seed=0):
    # A ResNet "BasicBlock"-shaped toy: two stacked MatMuls (standing in for
    # two stacked convolutions) plus a residual Add back to the block's own
    # input -- exactly the topology onnxsim.brecq's own docstring documents
    # discovering. D must be a multiple of 32:
    # onnxsim.quantize_weight_only_int4's own fixed block size.
    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal((D, D)) * 0.3).astype(np.float32)
    w2 = (rng.standard_normal((D, D)) * 0.3).astype(np.float32)
    return _model(
        f"""
        g (float[batch,{D}] X) => (float[batch,{D}] Yout)
        {{
          Y1 = MatMul(X, W1)
          Y2 = MatMul(Y1, W2)
          Yout = Add(Y2, X)
        }}
        """,
        [_f32(w1, "W1"), _f32(w2, "W2")],
    )


def _quantize_chain_int4(model, weight_names):
    # onnxsim.quantize_weight_only_int4 only ever quantizes the *first*
    # MatMul in a chain whose activation input is a graph input -- a MatMul
    # fed by another node's own output (i.e. every non-first layer of a
    # chain) is left untouched (verified directly: a plain 2- or 3-MatMul
    # chain only ever gets its first layer quantized). That is a limitation
    # of the existing pass this module does not attempt to fix (out of
    # scope for a new reconstruction module). To build a *fully* quantized
    # multi-layer chain for these tests, quantize each named weight's own
    # MatMul in isolation (a tiny standalone one-node model, where it *is*
    # the first/only layer) via the real production pass, then splice each
    # resulting DequantizeLinear + Wq/Ws back into the full chain model --
    # exactly the RTN codes the real pass would have produced for that
    # layer on its own, just assembled by hand instead of relying on the
    # pass's own multi-layer traversal.
    orig_inits = {t.name: t for t in model.graph.initializer}
    quant = onnx.ModelProto()
    quant.CopyFrom(model)

    new_nodes = []
    new_initializers = [
        t for t in quant.graph.initializer if t.name not in weight_names
    ]
    for idx, node in enumerate(quant.graph.node):
        if node.op_type not in ("MatMul", "Gemm") or node.input[1] not in weight_names:
            new_nodes.append(node)
            continue
        w_name = node.input[1]
        w_init = orig_inits[w_name]
        k, n = w_init.dims[0], w_init.dims[1]
        iso = _model(
            f"""
            h (float[batch,{k}] Ain) => (float[batch,{n}] Aout)
            {{
              Aout = MatMul(Ain, {w_name})
            }}
            """,
            [w_init],
        )
        iso_quant = onnxsim.quantize_weight_only_int4(iso)
        dq_node = next(
            n for n in iso_quant.graph.node if n.op_type == "DequantizeLinear"
        )
        wq_init = next(
            t for t in iso_quant.graph.initializer if t.name == dq_node.input[0]
        )
        ws_init = next(
            t for t in iso_quant.graph.initializer if t.name == dq_node.input[1]
        )

        suffix = f"_{idx}"
        wq_renamed = onnx.TensorProto()
        wq_renamed.CopyFrom(wq_init)
        wq_renamed.name += suffix
        ws_renamed = onnx.TensorProto()
        ws_renamed.CopyFrom(ws_init)
        ws_renamed.name += suffix
        dq_out_name = f"{w_name}_dq{suffix}"

        new_dq = onnx.NodeProto()
        new_dq.CopyFrom(dq_node)
        new_dq.input[0] = wq_renamed.name
        new_dq.input[1] = ws_renamed.name
        new_dq.output[0] = dq_out_name
        new_initializers.extend([wq_renamed, ws_renamed])
        new_nodes.append(new_dq)

        new_node = onnx.NodeProto()
        new_node.CopyFrom(node)
        new_node.input[1] = dq_out_name
        new_nodes.append(new_node)

    del quant.graph.node[:]
    quant.graph.node.extend(new_nodes)
    del quant.graph.initializer[:]
    quant.graph.initializer.extend(new_initializers)
    return quant


def _correlated_calibration(D=32, num_samples=64, rank=6, seed=1):
    # Same motivating scenario onnxsim.gptq/onnxsim.foem's own tests use:
    # input features correlated via a handful of latent factors, so the
    # chain's two layers' own errors actually interact instead of being
    # independent.
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, D)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, D)).astype(np.float32) * 0.05
    return x


def _dequantize_int4_for(model, matmul_output_name):
    # Same decode as onnxsim.gptq/onnxsim.foem's own test helpers, but
    # locating the DequantizeLinear feeding a *specific* MatMul's own weight
    # input by output name, since a block has more than one quantized layer.
    matmul_node = next(n for n in model.graph.node if n.output[0] == matmul_output_name)
    dq_node = next(n for n in model.graph.node if n.output[0] == matmul_node.input[1])
    wq = next(t for t in model.graph.initializer if t.name == dq_node.input[0])
    ws = next(t for t in model.graph.initializer if t.name == dq_node.input[1])
    block_size = next(a.i for a in dq_node.attribute if a.name == "block_size")
    axis = next((a.i for a in dq_node.attribute if a.name == "axis"), 1)

    dims = list(wq.dims)
    numel = int(np.prod(dims))
    raw = np.frombuffer(wq.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int8)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    codes = codes.reshape(dims).astype(np.float64)

    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    scale_full = np.repeat(scale, block_size, axis=axis)
    slicer = [slice(None)] * codes.ndim
    slicer[axis] = slice(0, codes.shape[axis])
    return codes * scale_full[tuple(slicer)]


def _block_output_via_numpy(model, x):
    w1 = _dequantize_int4_for(model, "Y1")
    w2 = _dequantize_int4_for(model, "Y2")
    y1 = x.astype(np.float64) @ w1
    y2 = y1 @ w2
    return y2 + x.astype(np.float64)


def test_brecq_reduces_block_reconstruction_error_vs_independent_adaround():
    # BRECQ's own claim, verified directly: jointly optimizing a block's
    # layers against the *block's own final output* beats optimizing each
    # layer independently against its own output (onnxsim.apply_adaround),
    # on a scenario engineered so the two layers' errors interact (a
    # correlated input feeding a chain with a residual add). This is not a
    # claim that joint block optimization always wins -- the BRECQ paper's
    # own claim is a modest, incremental gain over AdaRound, not a dramatic
    # one -- only the honest, measured result on this scenario.
    model = _residual_block_model(D=32, seed=0)
    x = _correlated_calibration(D=32, num_samples=64, rank=6, seed=1)
    calibration_data = [{"X": x}]

    quant = _quantize_chain_int4(model, {"W1", "W2"})
    w1_float = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == "W1")
    ).astype(np.float64)
    w2_float = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == "W2")
    ).astype(np.float64)
    y1_float = x.astype(np.float64) @ w1_float
    final_float = y1_float @ w2_float + x.astype(np.float64)

    adaround_model = onnxsim.apply_adaround(
        model, quant, calibration_data=calibration_data
    )
    onnx.checker.check_model(adaround_model)
    final_adaround = _block_output_via_numpy(adaround_model, x)
    adaround_err = np.linalg.norm(final_float - final_adaround)

    brecq_model = onnxsim.apply_brecq(
        model, quant, blocks=[("X", "Yout")], calibration_data=calibration_data
    )
    onnx.checker.check_model(brecq_model)
    final_brecq = _block_output_via_numpy(brecq_model, x)
    brecq_err = np.linalg.norm(final_float - final_brecq)

    final_rtn = _block_output_via_numpy(quant, x)
    rtn_err = np.linalg.norm(final_float - final_rtn)

    assert brecq_err < adaround_err
    assert brecq_err < rtn_err


def test_brecq_output_stays_close_to_float_via_onnxruntime():
    model = _residual_block_model(D=32, seed=2)
    x = _correlated_calibration(D=32, num_samples=32, rank=4, seed=3)
    calibration_data = [{"X": x}]

    brecq_model = onnxsim.apply_brecq(
        model,
        _quantize_chain_int4(model, {"W1", "W2"}),
        blocks=[("X", "Yout")],
        calibration_data=calibration_data,
    )
    onnx.checker.check_model(brecq_model)

    (float_y,) = _run(model, {"X": x})
    (brecq_y,) = _run(brecq_model, {"X": x})
    assert np.all(np.isfinite(brecq_y))
    assert _rel_l2(float_y, brecq_y) < 0.25


def test_brecq_preserves_scale_and_shape():
    model = _residual_block_model(D=32, seed=4)
    x = _correlated_calibration(D=32, num_samples=16, rank=3, seed=5)
    calibration_data = [{"X": x}]

    quant = _quantize_chain_int4(model, {"W1", "W2"})
    before_scales = {}
    for name in ("Y1", "Y2"):
        matmul_node = next(n for n in quant.graph.node if n.output[0] == name)
        dq_node = next(
            n for n in quant.graph.node if n.output[0] == matmul_node.input[1]
        )
        before_scales[name] = onnx.numpy_helper.to_array(
            next(t for t in quant.graph.initializer if t.name == dq_node.input[1])
        )

    brecq_model = onnxsim.apply_brecq(
        model, quant, blocks=[("X", "Yout")], calibration_data=calibration_data
    )
    for name in ("Y1", "Y2"):
        matmul_node = next(n for n in brecq_model.graph.node if n.output[0] == name)
        dq_node = next(
            n for n in brecq_model.graph.node if n.output[0] == matmul_node.input[1]
        )
        after_scale = onnx.numpy_helper.to_array(
            next(t for t in brecq_model.graph.initializer if t.name == dq_node.input[1])
        )
        np.testing.assert_array_equal(before_scales[name], after_scale)

        wq = next(
            t for t in brecq_model.graph.initializer if t.name == dq_node.input[0]
        )
        assert list(wq.dims) == [32, 32]


def test_brecq_codes_stay_in_range():
    model = _residual_block_model(D=32, seed=6)
    x = _correlated_calibration(D=32, num_samples=16, rank=2, seed=7) * 3
    calibration_data = [{"X": x}]

    quant = _quantize_chain_int4(model, {"W1", "W2"})
    brecq_model = onnxsim.apply_brecq(
        model, quant, blocks=[("X", "Yout")], calibration_data=calibration_data
    )
    checked_any = False
    for t in brecq_model.graph.initializer:
        if t.data_type != onnx.TensorProto.INT4:
            continue
        checked_any = True
        numel = int(np.prod(list(t.dims)))
        raw = np.frombuffer(t.raw_data, dtype=np.uint8)
        lo = (raw & 0x0F).astype(np.int8)
        hi = ((raw >> 4) & 0x0F).astype(np.int8)
        lo = np.where(lo >= 8, lo - 16, lo)
        hi = np.where(hi >= 8, hi - 16, hi)
        codes = np.empty(numel, dtype=np.int8)
        codes[0::2] = lo[: (numel + 1) // 2]
        codes[1::2] = hi[: numel // 2]
        assert np.all(codes >= -7) and np.all(codes <= 7)
    assert checked_any


def test_brecq_single_layer_chain_without_residual():
    # A "block" that is really just one layer, with block_output_name equal
    # to that layer's own output (no residual Add) -- the degenerate case
    # discovery must still handle.
    rng = np.random.default_rng(9)
    D = 32
    w = (rng.standard_normal((D, D)) * 0.3).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{D}] X) => (float[batch,{D}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(w, "W")],
    )
    x = _correlated_calibration(D=D, num_samples=32, rank=3, seed=10)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    brecq_model = onnxsim.apply_brecq(
        model, quant, blocks=[("X", "Y")], calibration_data=calibration_data
    )
    onnx.checker.check_model(brecq_model)

    w_float = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == "W")
    ).astype(np.float64)
    w_rtn = _dequantize_int4_for(quant, "Y")
    w_brecq = _dequantize_int4_for(brecq_model, "Y")
    y_float = x.astype(np.float64) @ w_float
    rtn_err = np.linalg.norm(y_float - x.astype(np.float64) @ w_rtn)
    brecq_err = np.linalg.norm(y_float - x.astype(np.float64) @ w_brecq)
    assert brecq_err < rtn_err


def test_brecq_noop_when_block_topology_not_discovered():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_brecq(
        model,
        model,
        blocks=[("X", "Y")],
        calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}],
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_brecq_noop_when_no_blocks_given():
    model = _residual_block_model(D=32, seed=11)
    quant = _quantize_chain_int4(model, {"W1", "W2"})
    result = onnxsim.apply_brecq(
        model,
        quant,
        blocks=[],
        calibration_data=[{"X": np.zeros((1, 32), dtype=np.float32)}],
    )
    assert result.SerializeToString() == quant.SerializeToString()
