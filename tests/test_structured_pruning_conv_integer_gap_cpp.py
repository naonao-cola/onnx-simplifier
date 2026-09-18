"""Focused tests for the ``GlobalAveragePool -> {Flatten(axis=1), Reshape to
[batch, -1], Squeeze-of-trailing-axes} -> {MatMul, vanilla Gemm}``
classifier-head hop of the ``ConvInteger``-based dynamic-quantization Conv
structured-pruning port (``onnxsim/structured_pruning_entry.cpp``'s own
"DynamicQuantizeConv (ConvInteger, ...)" section --
``MatchGapFlattenMatmulConsumer``/``WalkToConvIntegerConsumer``/
``ConvIntegerChain::matmul_consumer``/``ApplyConvIntegerChains``).

This hop mirrors ``pruning.py``'s own ``_match_gap_flatten_matmul_consumer``
exactly (see that function's own docstring for the full
axis-correspondence proof every matched shape here relies on), with one
deliberate, documented narrowing: the matched MatMul/Gemm consumer's own
weight must be FLOAT32 (not pruning.py's own wider FLOAT/FLOAT16/BFLOAT16
``_is_supported_float_dtype`` gate) -- see
``MatchGapFlattenMatmulConsumer``'s own comment for why (``SliceConsumerWeight``,
used to slice it, assumes a raw FLOAT32 buffer; mirrors ``MatchProducer``'s own
identical FLOAT32-only narrowing elsewhere in this file).

``tests/test_structured_pruning_cpp.py``'s own "DynamicQuantizeConv
(ConvInteger, ...)" section covers the ordinary same-family
producer/consumer chain (plus the Clip/depthwise-mid-chain hops); this file
is scoped to ONLY the classifier-head exception above.
"""

import os
import tempfile

import numpy as np
import onnx
import onnx.checker
import onnx.helper
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=21):
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _quantize_dynamic_conv_weight(W, spatial=8):
    """Runs the REAL ``onnxruntime.quantization.quantize_dynamic`` tool on a
    minimal ``Y = Conv(X, W)`` wrapper model and returns the genuine
    quantized ``(W_int8[M,C,kH,kW], W_scale, W_zero_point)`` triple it emits
    for `W` -- mirrors ``tests/test_structured_pruning_cpp.py``'s own
    identical helper.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    m, c, kh, kw = W.shape
    pad_h, pad_w = (kh - 1) // 2, (kw - 1) // 2
    src = _model(
        f"""
        g (float[1,{c},{spatial},{spatial}] X) => (float[1,{m},{spatial},{spatial}] Y)
        {{
          Y = Conv<kernel_shape=[{kh},{kw}], pads=[{pad_h},{pad_w},{pad_h},{pad_w}]>(X, W)
        }}
        """,
        initializer=[_f32(W, "W")],
        opset=21,
    )
    with tempfile.TemporaryDirectory() as d:
        src_path = os.path.join(d, "src.onnx")
        dst_path = os.path.join(d, "dst.onnx")
        onnx.save(src, src_path)
        quantize_dynamic(
            src_path,
            dst_path,
            per_channel=False,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["Conv"],
        )
        q = onnx.load(dst_path)
    inits = {t.name: t for t in q.graph.initializer}
    Wq = onnx.numpy_helper.to_array(inits["W_quantized"]).copy()
    Wscale = onnx.numpy_helper.to_array(inits["W_scale"]).copy()
    Wzp = onnx.numpy_helper.to_array(inits["W_zero_point"]).copy()
    return Wq, Wscale, Wzp


def _gap_classifier_model(
    c,
    m1,
    out,
    w1f,
    w3f,
    body_after_gap,
    extra_initializer=(),
    spatial=8,
    out_shape=None,
):
    """Builds ``DynamicQuantizeLinear -> ConvInteger -> Cast -> Mul ->
    GlobalAveragePool -> {body_after_gap}`` -- a single ``ConvInteger``
    producer feeding a classifier-head consumer shape DIRECTLY (no
    downstream ``ConvInteger`` at all), for `body_after_gap` (an ONNX-text
    fragment reading `p`, the GlobalAveragePool output, and producing the
    graph's own output `y`) to determine the exact shape under test.
    """
    w1q, w1s, w1zp = _quantize_dynamic_conv_weight(w1f, spatial=spatial)
    kh1, kw1 = w1f.shape[2], w1f.shape[3]
    ph1, pw1 = (kh1 - 1) // 2, (kw1 - 1) // 2
    if out_shape is None:
        out_shape = f"1,{out}"

    body = f"""
    g (float[1,{c},{spatial},{spatial}] x) => (float[{out_shape}] y)
    {{
      xq, x_scale, x_zero_point = DynamicQuantizeLinear(x)
      combined_scale1 = Mul(x_scale, w1_scale)
      c1i = ConvInteger<kernel_shape=[{kh1},{kw1}], pads=[{ph1},{pw1},{ph1},{pw1}]>(xq, w1_quantized, x_zero_point, w1_zero_point)
      c1f = Cast<to=1>(c1i)
      h1 = Mul(c1f, combined_scale1)
      p = GlobalAveragePool(h1)
      {body_after_gap}
    }}
    """
    inits = [
        onnx.numpy_helper.from_array(w1q, "w1_quantized"),
        onnx.numpy_helper.from_array(w1s, "w1_scale"),
        onnx.numpy_helper.from_array(w1zp, "w1_zero_point"),
        w3f,
        *extra_initializer,
    ]
    return _model(body, initializer=inits, opset=21)


def _assert_matches_python_reference_and_prunes(model, m1):
    pruned_py = onnxsim.apply_structured_pruning_dynamic_quantize_conv(
        model, sparsity=0.5
    )
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)
    # A real channel was actually removed (not a vacuous no-op comparison).
    assert pruned_py.SerializeToString() != model.SerializeToString()

    py_bytes = {t.name: t.SerializeToString() for t in pruned_py.graph.initializer}
    cpp_bytes = {t.name: t.SerializeToString() for t in pruned_cpp.graph.initializer}
    assert py_bytes == cpp_bytes
    inits = {t.name: t for t in pruned_cpp.graph.initializer}
    assert list(inits["w1_quantized"].dims)[0] == m1 // 2
    return pruned_cpp


# --- The three shapes MatchGapFlattenMatmulConsumer recognizes -------------


def test_gap_flatten_axis1_gemm_matches_python_reference():
    c, m1, out = 4, 8, 5
    rng = np.random.default_rng(101)
    w1f = (rng.standard_normal((m1, c, 3, 3)) * 0.3).astype(np.float32)
    w3f = _f32(rng.standard_normal((out, m1)) * 0.3, "w3")
    model = _gap_classifier_model(
        c, m1, out, w1f, w3f, "f = Flatten<axis=1>(p)\n      y = Gemm<transB=1>(f, w3)"
    )
    onnx.checker.check_model(model)
    pruned = _assert_matches_python_reference_and_prunes(model, m1)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["w3"].dims)[1] == m1 // 2


def test_gap_flatten_axis1_matmul_matches_python_reference():
    # `MatMul` (not just `Gemm`) is one of the two matched families -- the
    # matched weight here is untransposed ([K, N] storage), unlike the
    # `Gemm<transB=1>` fixtures above/below.
    c, m1, out = 4, 8, 5
    rng = np.random.default_rng(102)
    w1f = (rng.standard_normal((m1, c, 3, 3)) * 0.3).astype(np.float32)
    w3f = _f32(rng.standard_normal((m1, out)) * 0.3, "w3")
    model = _gap_classifier_model(
        c, m1, out, w1f, w3f, "f = Flatten<axis=1>(p)\n      y = MatMul(f, w3)"
    )
    onnx.checker.check_model(model)
    pruned = _assert_matches_python_reference_and_prunes(model, m1)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["w3"].dims)[0] == m1 // 2


def test_gap_reshape_batch_neg1_gemm_matches_python_reference():
    # `x.reshape(x.shape[0], -1)`'s own static-batch shape: `shape` is a
    # constant, 1-D, 2-element `[1, -1]` initializer.
    c, m1, out = 4, 8, 5
    rng = np.random.default_rng(103)
    w1f = (rng.standard_normal((m1, c, 3, 3)) * 0.3).astype(np.float32)
    w3f = _f32(rng.standard_normal((out, m1)) * 0.3, "w3")
    rshape = onnx.numpy_helper.from_array(np.array([1, -1], dtype=np.int64), "rshape")
    model = _gap_classifier_model(
        c,
        m1,
        out,
        w1f,
        w3f,
        "f = Reshape(p, rshape)\n      y = Gemm<transB=1>(f, w3)",
        extra_initializer=[rshape],
    )
    onnx.checker.check_model(model)
    _assert_matches_python_reference_and_prunes(model, m1)


def test_gap_squeeze_single_node_both_trailing_axes_gemm_matches_python_reference():
    # A single `Squeeze(axes=[2, 3])` collapsing BOTH trailing size-1 axes
    # from `GlobalAveragePool`'s own `[N, C, 1, 1]` output at once, using
    # non-negative axis indices (unambiguous, no rank inference needed) --
    # unlike the two-hop `Squeeze(-1)`-then-`Squeeze(-1)` chain the next test
    # covers, this is a SINGLE `Squeeze` node with a 2-element `axes`.
    c, m1, out = 4, 8, 5
    rng = np.random.default_rng(104)
    w1f = (rng.standard_normal((m1, c, 3, 3)) * 0.3).astype(np.float32)
    w3f = _f32(rng.standard_normal((out, m1)) * 0.3, "w3")
    axes = onnx.numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "sqaxes")
    model = _gap_classifier_model(
        c,
        m1,
        out,
        w1f,
        w3f,
        "f = Squeeze(p, sqaxes)\n      y = Gemm<transB=1>(f, w3)",
        extra_initializer=[axes],
    )
    onnx.checker.check_model(model)
    _assert_matches_python_reference_and_prunes(model, m1)


def test_gap_squeeze_negative_non_last_axis_is_left_undeclined():
    # `axis == -2` is NOT one of the unambiguously-safe trailing axes this
    # hop recognizes (only non-negative `axis >= 2` or the single
    # unambiguous negative case `axis == -1` are -- see
    # `SqueezeAxisIsTrailingSpatial`'s own comment for the full proof: any
    # other negative axis would need the tensor's own rank to resolve
    # unambiguously and is declined, never guessed at), so a single
    # `Squeeze(axes=[-1, -2])` declines the WHOLE match right there --
    # mirrors pruning.py's own `_squeeze_axis_is_trailing_spatial` exactly.
    c, m1, out = 4, 8, 5
    rng = np.random.default_rng(1040)
    w1f = (rng.standard_normal((m1, c, 3, 3)) * 0.3).astype(np.float32)
    w3f = _f32(rng.standard_normal((out, m1)) * 0.3, "w3")
    axes = onnx.numpy_helper.from_array(np.array([-1, -2], dtype=np.int64), "sqaxes")
    model = _gap_classifier_model(
        c,
        m1,
        out,
        w1f,
        w3f,
        "f = Squeeze(p, sqaxes)\n      y = Gemm<transB=1>(f, w3)",
        extra_initializer=[axes],
    )
    onnx.checker.check_model(model)
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_structured_pruning_dynamic_quantize_conv(
        model, sparsity=0.5
    )
    assert pruned_cpp.SerializeToString() == model.SerializeToString()
    assert pruned_py.SerializeToString() == model.SerializeToString()


def test_gap_squeeze_chain_of_two_matches_python_reference():
    # `x.squeeze(-1).squeeze(-1)` -- a CHAIN of two single-axis `Squeeze`
    # nodes, each squeezing axis -1, mirroring a real export's own preferred
    # shape at least as often as an explicit `nn.Flatten()`.
    c, m1, out = 4, 8, 5
    rng = np.random.default_rng(105)
    w1f = (rng.standard_normal((m1, c, 3, 3)) * 0.3).astype(np.float32)
    w3f = _f32(rng.standard_normal((out, m1)) * 0.3, "w3")
    axes = onnx.numpy_helper.from_array(np.array([-1], dtype=np.int64), "sqaxes")
    model = _gap_classifier_model(
        c,
        m1,
        out,
        w1f,
        w3f,
        "s1 = Squeeze(p, sqaxes)\n      f = Squeeze(s1, sqaxes)\n      y = Gemm<transB=1>(f, w3)",
        extra_initializer=[axes],
    )
    onnx.checker.check_model(model)
    _assert_matches_python_reference_and_prunes(model, m1)


def test_gap_reshape_dynamic_batch_two_consumer_shape_matches_python_reference():
    # `x.view(x.size(0), -1)`'s own DYNAMIC-batch shape: `gap_out` is read
    # TWICE -- once by the `Reshape` that consumes it as data, once by a
    # `Shape` node feeding the batch-size computation
    # (`Shape -> Gather(indices=[0]) -> Unsqueeze`) that's `Concat`-ed with a
    # constant `-1` tail. The `-1` tail is deliberately sourced from a
    # `Constant` node (not a genuine `graph.initializer` entry) -- the shape
    # a real `torch.onnx.export` emits -- exercising
    # `FindConvIntegerChains`'s own `BuildConstantMap`-based (not a bare
    # `graph.initializer()` loop) constant resolution.
    c, m1, out = 4, 8, 5
    rng = np.random.default_rng(106)
    w1f = (rng.standard_normal((m1, c, 3, 3)) * 0.3).astype(np.float32)
    w3f = _f32(rng.standard_normal((out, m1)) * 0.3, "w3")
    extra = [
        onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), "gather_idx"),
        onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), "unsq_axes"),
    ]
    body_after_gap = """
      bshape = Shape(p)
      b0 = Gather<axis=0>(bshape, gather_idx)
      b0u = Unsqueeze(b0, unsq_axes)
      neg1 = Constant<value = int64[1] {-1}>()
      rshape2 = Concat<axis=0>(b0u, neg1)
      f = Reshape(p, rshape2)
      y = Gemm<transB=1>(f, w3)
    """
    model = _gap_classifier_model(
        c, m1, out, w1f, w3f, body_after_gap, extra_initializer=extra
    )
    onnx.checker.check_model(model)
    _assert_matches_python_reference_and_prunes(model, m1)


# --- Negative cases: left completely undeclined, never mis-matched ---------


def test_gap_feeding_unrecognized_op_before_flatten_is_left_undeclined():
    # `GlobalAveragePool -> Relu -> Flatten -> Gemm`: `Relu` isn't one of the
    # three recognized shapes reading `gap_out` directly, so the whole match
    # declines right there -- the producer is left completely untouched,
    # exactly like any other unmatched topology (never guessed at).
    c, m1, out = 4, 8, 5
    rng = np.random.default_rng(107)
    w1f = (rng.standard_normal((m1, c, 3, 3)) * 0.3).astype(np.float32)
    w3f = _f32(rng.standard_normal((out, m1)) * 0.3, "w3")
    model = _gap_classifier_model(
        c,
        m1,
        out,
        w1f,
        w3f,
        "r = Relu(p)\n      f = Flatten<axis=1>(r)\n      y = Gemm<transB=1>(f, w3)",
    )
    onnx.checker.check_model(model)
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    pruned_py = onnxsim.apply_structured_pruning_dynamic_quantize_conv(
        model, sparsity=0.5
    )
    assert pruned_cpp.SerializeToString() == model.SerializeToString()
    assert pruned_py.SerializeToString() == model.SerializeToString()


def test_gap_flatten_axis0_is_left_undeclined():
    # `Flatten(axis=0)` merges the batch axis into the channel one -- not the
    # trivial 1:1 channel relabeling `axis=1` is -- so it's declined, exactly
    # matching pruning.py's own `_match_flatten_axis1_pass_through` scope.
    c, m1 = 4, 8
    rng = np.random.default_rng(108)
    w1f = (rng.standard_normal((m1, c, 3, 3)) * 0.3).astype(np.float32)
    w3f = _f32(rng.standard_normal((1, m1)) * 0.3, "w3")
    model = _gap_classifier_model(
        c,
        m1,
        None,
        w1f,
        w3f,
        "f = Flatten<axis=0>(p)\n      y = Gemm<transB=1>(f, w3)",
        out_shape="1,1",
    )
    onnx.checker.check_model(model)
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()


def test_gap_two_reshape_consumers_not_reshape_plus_shape_is_left_undeclined():
    # `gap_out` read by TWO `Reshape` nodes (not one `Reshape` + one `Shape`)
    # -- doesn't match the one tolerated two-consumer shape, so declined.
    c, m1, out = 4, 8, 5
    rng = np.random.default_rng(109)
    w1f = (rng.standard_normal((m1, c, 3, 3)) * 0.3).astype(np.float32)
    w3f = _f32(rng.standard_normal((out, m1)) * 0.3, "w3")
    rshape = onnx.numpy_helper.from_array(np.array([1, -1], dtype=np.int64), "rshape")
    model = _gap_classifier_model(
        c,
        m1,
        out,
        w1f,
        w3f,
        "f1 = Reshape(p, rshape)\n      f2 = Reshape(p, rshape)\n      y = Gemm<transB=1>(f1, w3)",
        extra_initializer=[rshape],
    )
    onnx.checker.check_model(model)
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()


def test_gap_reshape_non_neg1_target_is_left_undeclined():
    # A `Reshape` to a LITERAL (non-`-1`) target shape needs a real
    # memory-layout re-derivation, not a trivial pass-through -- declined,
    # matching pruning.py's own `_match_reshape_batch_neg1_pass_through`
    # scope exactly.
    c, m1, out = 4, 8, 5
    rng = np.random.default_rng(110)
    w1f = (rng.standard_normal((m1, c, 3, 3)) * 0.3).astype(np.float32)
    w3f = _f32(rng.standard_normal((out, m1)) * 0.3, "w3")
    rshape = onnx.numpy_helper.from_array(np.array([1, m1], dtype=np.int64), "rshape")
    model = _gap_classifier_model(
        c,
        m1,
        out,
        w1f,
        w3f,
        "f = Reshape(p, rshape)\n      y = Gemm<transB=1>(f, w3)",
        extra_initializer=[rshape],
    )
    onnx.checker.check_model(model)
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()


def test_gap_flatten_matmul_float16_weight_is_left_undeclined_but_python_prunes():
    # Deliberate, documented scope narrowing (see this file's own top
    # comment and `MatchGapFlattenMatmulConsumer`'s own comment): the
    # matched MatMul/Gemm consumer's own weight must be FLOAT32 here, unlike
    # pruning.py's own wider FLOAT/FLOAT16/BFLOAT16
    # `_is_supported_float_dtype` gate -- a FLOAT16 consumer weight is left
    # completely undeclined by the C++ port (never mis-sliced), while the
    # Python reference DOES prune it, confirming this is a genuine, narrower
    # (but still correct) scope rather than an oversight.
    c, m1, out = 4, 8, 5
    rng = np.random.default_rng(111)
    w1f = (rng.standard_normal((m1, c, 3, 3)) * 0.3).astype(np.float32)
    w3f = onnx.numpy_helper.from_array(
        (rng.standard_normal((out, m1)) * 0.3).astype(np.float16), "w3"
    )
    model = _gap_classifier_model(
        c, m1, out, w1f, w3f, "f = Flatten<axis=1>(p)\n      y = Gemm<transB=1>(f, w3)"
    )
    onnx.checker.check_model(model)
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned_cpp.SerializeToString() == model.SerializeToString()

    pruned_py = onnxsim.apply_structured_pruning_dynamic_quantize_conv(
        model, sparsity=0.5
    )
    assert pruned_py.SerializeToString() != model.SerializeToString()
    py_inits = {t.name: t for t in pruned_py.graph.initializer}
    assert list(py_inits["w1_quantized"].dims)[0] == m1 // 2


def test_gap_flatten_matmul_zero_sparsity_is_a_no_op():
    c, m1, out = 4, 8, 5
    rng = np.random.default_rng(112)
    w1f = (rng.standard_normal((m1, c, 3, 3)) * 0.3).astype(np.float32)
    w3f = _f32(rng.standard_normal((out, m1)) * 0.3, "w3")
    model = _gap_classifier_model(
        c, m1, out, w1f, w3f, "f = Flatten<axis=1>(p)\n      y = Gemm<transB=1>(f, w3)"
    )
    onnx.checker.check_model(model)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.0)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_gap_flatten_matmul_consumer_weight_aliased_elsewhere_matches_python_reference():
    # `w3` (the matched Gemm consumer's own weight) is ALSO read by a
    # second, unrelated `Identity` node elsewhere in the graph.
    # `MatchGapFlattenMatmulConsumer` -- like pruning.py's own
    # `_match_gap_flatten_matmul_consumer` -- does NOT check whether this
    # matched weight is exclusively read by the matched consumer (unlike the
    # ordinary same-family `ConvInteger` producer/consumer role, where
    # `MatchConvInteger` DOES require a sole reader): both the Python
    # reference and this C++ port prune `w3` here regardless, corrupting
    # `w3_alias`'s own now-stale shape -- a pre-existing behavior of the
    # Python reference this port faithfully reproduces byte-for-byte, not a
    # new decline to add. (Fixing that pre-existing gap is out of scope for
    # this port; ported behavior must match the reference, not improve on
    # it.)
    c, m1, out = 4, 8, 5
    rng = np.random.default_rng(113)
    w1f = (rng.standard_normal((m1, c, 3, 3)) * 0.3).astype(np.float32)
    w3f = _f32(rng.standard_normal((out, m1)) * 0.3, "w3")
    model = _gap_classifier_model(
        c, m1, out, w1f, w3f, "f = Flatten<axis=1>(p)\n      y = Gemm<transB=1>(f, w3)"
    )
    extra = onnx.helper.make_node("Identity", ["w3"], ["w3_alias"], name="alias")
    model.graph.node.append(extra)
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(
            "w3_alias", onnx.TensorProto.FLOAT, [out, m1]
        )
    )
    onnx.checker.check_model(model)
    pruned_py = onnxsim.apply_structured_pruning_dynamic_quantize_conv(
        model, sparsity=0.5
    )
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)
    assert pruned_py.SerializeToString() != model.SerializeToString()
    py_bytes = {t.name: t.SerializeToString() for t in pruned_py.graph.initializer}
    cpp_bytes = {t.name: t.SerializeToString() for t in pruned_cpp.graph.initializer}
    assert py_bytes == cpp_bytes
