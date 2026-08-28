"""Tests for MACs / FLOPs counting in ``onnxsim.model_info``.

Every model is built via ``onnx.parser.parse_model`` (the ONNX text format; no
torch dependency). MAC counts are asserted against hand-computed values. The
symbolic (sympy) path is exercised when sympy is installed and skipped
otherwise, mirroring the optional ``onnxsim[symbolic]`` extra.
"""

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper, parser

from onnxsim import model_info
from onnxsim.model_info import (
    METADATA_PREFIX,
    ModelInfo,
    annotate_metadata,
    human_readable_density,
    human_readable_num,
    human_readable_size,
)


def _model(body, initializer=(), opset=23, ir_version=10):
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


def _weight(shape, name):
    # Zero-filled but potentially large (e.g. 4*3*3*3, 16*32 elements) --
    # spelling these out as text-literal tensors would require one literal
    # per element, so they're built as numpy arrays and attached as
    # initializers after parsing, per the repo's established exception.
    return numpy_helper.from_array(np.zeros(shape, np.float32), name)


def _macs(body, initializer=(), opset=23):
    return ModelInfo(_model(body, initializer, opset)).macs


# --------------------------------------------------------------------------- #
# Core compute operators
# --------------------------------------------------------------------------- #
def test_conv_macs():
    # output 1*4*8*8, cin/group 3, kernel 3*3 -> 256 * 3 * 9
    body = """
    g (float[1,3,8,8] x) => (float[1,4,8,8] y)
    {
      y = Conv<kernel_shape=[3,3], pads=[1,1,1,1]>(x, w)
    }
    """
    macs = _macs(body, [_weight([4, 3, 3, 3], "w")])
    assert macs == 1 * 4 * 8 * 8 * 3 * (3 * 3)


def test_conv_grouped_macs():
    # depthwise: groups=4, weight [4, 1, 3, 3]; cin/group = 1
    body = """
    g (float[1,4,8,8] x) => (float[1,4,8,8] y)
    {
      y = Conv<kernel_shape=[3,3], pads=[1,1,1,1], group=4>(x, w)
    }
    """
    macs = _macs(body, [_weight([4, 1, 3, 3], "w")])
    assert macs == 1 * 4 * 8 * 8 * 1 * (3 * 3)


def test_conv_transpose_macs():
    # input 1*3*8*8, out_channels/group 4, kernel 3*3
    body = """
    g (float[1,3,8,8] x) => (float[1,4,10,10] y)
    {
      y = ConvTranspose<kernel_shape=[3,3]>(x, w)
    }
    """
    # w is [in, out/group, kH, kW]
    macs = _macs(body, [_weight([3, 4, 3, 3], "w")])
    assert macs == 1 * 3 * 8 * 8 * 4 * (3 * 3)


def test_gemm_macs():
    body = """
    g (float[5,7] a, float[7,3] b) => (float[5,3] y)
    {
      y = Gemm(a, b)
    }
    """
    assert _macs(body) == 5 * 3 * 7


def test_gemm_transposed_macs():
    # transB=1: b is [N, K] = [3, 7]; M=5, N=3, K=7
    body = """
    g (float[5,7] a, float[3,7] b) => (float[5,3] y)
    {
      y = Gemm<transB=1>(a, b)
    }
    """
    assert _macs(body) == 5 * 3 * 7


def test_matmul_batched_macs():
    # A [2, 5, 7], B [7, 3] -> Y [2, 5, 3]; K=7
    body = """
    g (float[2,5,7] a) => (float[2,5,3] y)
    {
      y = MatMul(a, b)
    }
    """
    macs = _macs(body, [_weight([7, 3], "b")])
    assert macs == 2 * 5 * 3 * 7


# --------------------------------------------------------------------------- #
# Attention (ai.onnx opset 23+)
# --------------------------------------------------------------------------- #
def _attention_4d(hq, hkv):
    b, sq, skv, d, dv = 2, 16, 16, 64, 64
    body = f"""
    g (float[{b},{hq},{sq},{d}] q, float[{b},{hkv},{skv},{d}] k,
       float[{b},{hkv},{skv},{dv}] v) => (float[{b},{hq},{sq},{dv}] y)
    {{
      y = Attention(q, k, v)
    }}
    """
    expected = b * hq * sq * skv * d + b * hq * sq * skv * dv
    return _macs(body), expected


def test_attention_4d_mha():
    got, expected = _attention_4d(hq=8, hkv=8)
    assert got == expected


def test_attention_4d_gqa_uses_query_heads():
    # kv heads < q heads, but all q heads are evaluated.
    got, expected = _attention_4d(hq=8, hkv=2)
    assert got == expected


def test_attention_3d_uses_head_attrs():
    b, sq, skv, hidden, heads = 2, 16, 16, 512, 8
    body = f"""
    g (float[{b},{sq},{hidden}] q, float[{b},{skv},{hidden}] k,
       float[{b},{skv},{hidden}] v) => (float[{b},{sq},{hidden}] y)
    {{
      y = Attention<q_num_heads={heads}, kv_num_heads={heads}>(q, k, v)
    }}
    """
    d = hidden // heads
    expected = b * heads * sq * skv * d + b * heads * sq * skv * d
    assert _macs(body) == expected


def test_attention_3d_without_head_attrs_is_zero():
    # Head split is unknowable without q_num_heads / kv_num_heads.
    b, sq, hidden = 2, 16, 512
    body = f"""
    g (float[{b},{sq},{hidden}] q, float[{b},{sq},{hidden}] k,
       float[{b},{sq},{hidden}] v) => (float[{b},{sq},{hidden}] y)
    {{
      y = Attention(q, k, v)
    }}
    """
    assert _macs(body) == 0


# --------------------------------------------------------------------------- #
# Quantized twins reuse the float formulas at the right operand indices
# --------------------------------------------------------------------------- #
def test_matmul_integer_macs():
    body = """
    g (uint8[4,8] a, uint8[8,16] b) => (int32[4,16] y)
    {
      y = MatMulInteger(a, b)
    }
    """
    assert _macs(body, opset=10) == 4 * 16 * 8


def test_qlinearconv_weight_at_input3():
    # QLinearConv packs weight at input[3]: x, x_s, x_z, w, w_s, w_z, y_s, y_z
    body = """
    g (uint8[1,3,8,8] x, float x_s, uint8 x_z, uint8[4,3,3,3] w,
       float[4] w_s, uint8[4] w_z, float y_s, uint8 y_z)
      => (uint8[1,4,8,8] y)
    {
      y = QLinearConv<kernel_shape=[3,3], pads=[1,1,1,1]>(
        x, x_s, x_z, w, w_s, w_z, y_s, y_z)
    }
    """
    assert _macs(body, opset=10) == 1 * 4 * 8 * 8 * 3 * 9


# --------------------------------------------------------------------------- #
# Unknown / dynamic shapes
# --------------------------------------------------------------------------- #
def test_unnamed_dynamic_dim_counts_per_sample():
    # A batch dim with neither a value nor a name ("?" in the text format):
    # ONNX shape inference assigns it a generated symbol (e.g. "unk__0"), so
    # the count is linear in that axis and collapses to the per-sample MACs
    # when the axis is set to 1.
    body = """
    g (float[?,3,8,8] x) => (float[?,4,8,8] y)
    {
      y = Conv<kernel_shape=[3,3], pads=[1,1,1,1]>(x, w)
    }
    """
    macs = _macs(body, [_weight([4, 3, 3, 3], "w")])
    assert model_info._representative_number(macs) == 1 * 4 * 8 * 8 * 3 * 9


def test_uninferrable_node_contributes_zero():
    # When a required operand has no shape at all (empty shape map), the
    # counter returns 0 rather than guessing. This calls the counter
    # directly with a bare NodeProto, not a full model, so it stays on
    # onnx.helper rather than onnx.parser.
    node = helper.make_node("Gemm", ["a", "b"], ["y"])
    assert model_info._gemm_macs(node, {}) == 0


def test_flops_is_twice_macs():
    body = """
    g (float[5,7] a, float[7,3] b) => (float[5,3] y)
    {
      y = Gemm(a, b)
    }
    """
    info = ModelInfo(_model(body))
    assert info.flops == 2 * info.macs


# --------------------------------------------------------------------------- #
# Symbolic (sympy) path
# --------------------------------------------------------------------------- #
def _symbolic_batch_conv_model():
    body = """
    g (float[batch,3,8,8] x) => (float[batch,4,8,8] y)
    {
      y = Conv<kernel_shape=[3,3], pads=[1,1,1,1]>(x, w)
    }
    """
    return _model(body, [_weight([4, 3, 3, 3], "w")])


def test_dynamic_dim_symbolic():
    sympy = pytest.importorskip("sympy")
    macs = ModelInfo(_symbolic_batch_conv_model()).macs
    batch = sympy.Symbol("batch", positive=True, integer=True)
    assert sympy.simplify(macs - 1 * 4 * 8 * 8 * 3 * 9 * batch) == 0


def test_symbolic_dims_unify_across_tensors():
    # A shared dim_param name ("seq") must collapse to one symbol so the two
    # attention matmuls combine into a single seq**2 term.
    sympy = pytest.importorskip("sympy")
    b, hq, d = 2, 8, 64
    body = f"""
    g (float[{b},{hq},seq,{d}] q, float[{b},{hq},seq,{d}] k,
       float[{b},{hq},seq,{d}] v) => (float[{b},{hq},seq,{d}] y)
    {{
      y = Attention(q, k, v)
    }}
    """
    macs = ModelInfo(_model(body)).macs
    seq = sympy.Symbol("seq", positive=True, integer=True)
    assert sympy.simplify(macs - 2 * b * hq * d * seq**2) == 0


def test_symbolic_human_readable_num():
    sympy = pytest.importorskip("sympy")
    batch = sympy.Symbol("batch", positive=True, integer=True)
    assert human_readable_num(9472 * batch) == "9472*batch"


def test_print_simplifying_info_symbolic_does_not_raise():
    pytest.importorskip("sympy")
    model = _symbolic_batch_conv_model()
    model_info.print_simplifying_info(model, model)  # must not raise on symbolic "<"


def test_print_simplifying_info_factor_recursion_error_falls_back(monkeypatch):
    # On real-world models with many unresolved symbolic dims (e.g. 1000+ node
    # graphs with data-dependent output shapes), sympy.factor()'s polynomial
    # arithmetic can exceed Python's recursion limit -- reproduced against
    # VOICEVOX's predict_sing_f0.onnx / sf_decode.onnx. That must degrade to
    # the unfactored formula, not crash the whole report.
    pytest.importorskip("sympy")
    import sympy

    def _raise_recursion_error(_expr):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(sympy, "factor", _raise_recursion_error)

    model = _symbolic_batch_conv_model()
    model_info.print_simplifying_info(model, model)  # must not raise

    batch = sympy.Symbol("batch")
    assert human_readable_num(9472 * batch) == str(9472 * batch)
    assert human_readable_size(9472 * batch) == str(9472 * batch)
    assert human_readable_density(9472 * batch) == f"{9472 * batch} FLOP/Byte"


def test_factor_skipped_above_max_free_symbols(monkeypatch):
    # A model whose shape inference doesn't deduplicate dynamic dims back to a
    # handful of named input dims (reproduced exporting a real multi-submodule
    # TTS model to ONNX) can produce formulas with hundreds of distinct free
    # symbols. sympy.factor() over that many variables doesn't raise -- it just
    # never returns -- so it must be skipped by symbol count, not just guarded
    # against RecursionError. Prove it's skipped (not merely fast) by making
    # factor() itself blow up if it's ever called.
    sympy = pytest.importorskip("sympy")

    def _must_not_be_called(_expr):
        raise AssertionError("sympy.factor() must be skipped above the threshold")

    monkeypatch.setattr(sympy, "factor", _must_not_be_called)

    symbols = sympy.symbols(
        f"s0:{model_info._MAX_FACTOR_FREE_SYMBOLS + 1}", positive=True, integer=True
    )
    expr = sum(symbols)
    assert model_info._factor_or_str(expr) == str(expr)


def test_factor_still_applied_within_threshold():
    # Below the threshold, factoring must still run (it is a real formatting
    # nicety for the common, well-named-dims case) -- confirm it actually
    # changes the printed form rather than always falling back.
    sympy = pytest.importorskip("sympy")
    a, b, c = sympy.symbols("a b c", positive=True, integer=True)
    expr = 2 * a * b + 4 * a * c
    factored = model_info._factor_or_str(expr)
    assert factored != str(expr)
    # Re-parse against the *same* symbol objects: sympify() would otherwise
    # mint fresh, assumption-less a/b/c that don't cancel against the
    # positive-integer ones above.
    reparsed = sympy.sympify(factored, locals={"a": a, "b": b, "c": c})
    assert sympy.simplify(reparsed - expr) == 0


def test_representative_number_does_not_use_subs(monkeypatch):
    # _representative_number collapses every free symbol to 1 via xreplace, a
    # direct syntactic substitution. subs() additionally runs structural-
    # equality/simplification passes meant for pattern-based substitution, and
    # over hundreds of undeduplicated dynamic-dim symbols (see above) that took
    # minutes instead of milliseconds. Prove xreplace (not subs) is used by
    # making subs() itself blow up if it's ever called.
    sympy = pytest.importorskip("sympy")

    def _must_not_be_called(self, *args, **kwargs):
        raise AssertionError("Expr.subs() must not be used here; use xreplace")

    monkeypatch.setattr(sympy.Expr, "subs", _must_not_be_called)

    n = 300
    symbols = sympy.symbols(f"s0:{n}", positive=True, integer=True)
    expr = sum(symbols)
    assert model_info._representative_number(expr) == n


def test_dynamic_dim_without_sympy_assumes_one(monkeypatch):
    # With sympy unavailable, dynamic dims are assumed 1 (per-sample MACs).
    monkeypatch.setattr(model_info, "sympy", None)
    assert ModelInfo(_symbolic_batch_conv_model()).macs == 1 * 4 * 8 * 8 * 3 * 9


# --------------------------------------------------------------------------- #
# onnx-shape-inference backend (onnxsim[shape-inference])
# --------------------------------------------------------------------------- #
def test_infer_shapes_uses_onnx_shape_inference_when_installed():
    # onnx-shape-inference (https://github.com/justinchuby/onnx-shape-inference)
    # does data propagation through a Shape -> Gather -> Concat -> Reshape chain
    # that onnx.shape_inference, called without data_prop (as ModelInfo does by
    # default), cannot see through -- it should recover "y"'s shape as
    # ["batch", 2, 3] instead of leaving it unknown.
    pytest.importorskip("onnx_shape_inference")
    body = """
    g (float[batch,6] x) => (float[] y)
    <int64 idx = {0}, int64[1] axes = {0}, int64[1] two = {2}, int64[1] three = {3}>
    {
      x_shape = Shape(x)
      batch_dim = Gather<axis=0>(x_shape, idx)
      batch_dim_1 = Unsqueeze(batch_dim, axes)
      new_shape = Concat<axis=0>(batch_dim_1, two, three)
      y = Reshape(x, new_shape)
    }
    """
    inferred = ModelInfo._infer_shapes(_model(body))
    (out,) = [o for o in inferred.graph.output if o.name == "y"]
    dims = list(out.type.tensor_type.shape.dim)
    assert dims[0].dim_param == "batch"
    assert dims[1].dim_value == 2
    assert dims[2].dim_value == 3


def test_infer_shapes_falls_back_without_onnx_shape_inference(monkeypatch):
    # With the optional onnx-shape-inference backend unavailable, _infer_shapes
    # must still work via onnx's own shape_inference.
    monkeypatch.setattr(model_info, "infer_symbolic_shapes", None)
    body = """
    g (float[1,3,8,8] x) => (float[1,4,8,8] y)
    {
      y = Conv<kernel_shape=[3,3], pads=[1,1,1,1]>(x, w)
    }
    """
    model = _model(body, [_weight([4, 3, 3, 3], "w")])
    inferred = model_info.ModelInfo._infer_shapes(model)
    assert any(vi.name == "y" for vi in inferred.graph.output)


def test_infer_shapes_falls_back_when_onnx_shape_inference_raises(monkeypatch):
    # A failure in the optional backend (e.g. an unsupported op) must degrade to
    # onnx.shape_inference with a warning, not propagate.
    pytest.importorskip("onnx_shape_inference")

    def _raise(_ir_model):
        raise RuntimeError("boom")

    monkeypatch.setattr(model_info, "infer_symbolic_shapes", _raise)
    body = """
    g (float[1,3,8,8] x) => (float[1,4,8,8] y)
    {
      y = Conv<kernel_shape=[3,3], pads=[1,1,1,1]>(x, w)
    }
    """
    model = _model(body, [_weight([4, 3, 3, 3], "w")])
    with pytest.warns(UserWarning, match="onnx-shape-inference failed"):
        inferred = ModelInfo._infer_shapes(model)
    assert any(vi.name == "y" for vi in inferred.graph.output)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def test_human_readable_num_units():
    assert human_readable_num(0) == "0.0"
    assert human_readable_num(9472) == "9.5K"
    assert human_readable_num(3_000_000) == "3.0M"


def test_human_readable_size_units():
    assert human_readable_size(512) == "512.0B"
    assert human_readable_size(1024) == "1.0KiB"


def test_human_readable_density_units():
    assert human_readable_density(0) == "0.00 FLOP/Byte"
    assert human_readable_density(2.5) == "2.50 FLOP/Byte"


def test_human_readable_size_symbolic():
    sympy = pytest.importorskip("sympy")
    batch = sympy.Symbol("batch", positive=True, integer=True)
    assert human_readable_size(512 * batch) == "512*batch"


# --------------------------------------------------------------------------- #
# Memory metrics: access traffic, peak footprint, compute density
# --------------------------------------------------------------------------- #
def _info(body, initializer=(), opset=23):
    return ModelInfo(_model(body, initializer, opset))


def test_memory_access_gemm():
    # float32 tensors: reads a (5*7) + b (7*3), writes y (5*3), 4 bytes each.
    body = """
    g (float[5,7] a, float[7,3] b) => (float[5,3] y)
    {
      y = Gemm(a, b)
    }
    """
    expected = (5 * 7 + 7 * 3 + 5 * 3) * 4
    assert _info(body).mem_access == expected


def test_memory_access_counts_weights():
    # A weight fed as an initializer is read from memory, so its bytes count.
    body = """
    g (float[4,8] a) => (float[4,16] y)
    {
      y = MatMul(a, b)
    }
    """
    expected = (4 * 8 + 8 * 16 + 4 * 16) * 4
    assert _info(body, [_weight([8, 16], "b")]).mem_access == expected


def test_memory_access_respects_dtype_size():
    # float16 halves the bytes of the float32 case.
    body = """
    g (float16[5,7] a, float16[7,3] b) => (float16[5,3] y)
    {
      y = Gemm(a, b)
    }
    """
    expected = (5 * 7 + 7 * 3 + 5 * 3) * 2
    assert _info(body).mem_access == expected


def test_memory_access_unknown_shape_contributes_zero():
    # An intermediate with no inferred shape simply drops out of the total; the
    # known operands are still counted. No shapes are supplied (empty maps), so
    # the node's traffic is entirely unknown and totals zero. This calls the
    # counter directly with a bare NodeProto, not a full model, so it stays on
    # onnx.helper rather than onnx.parser.
    node = helper.make_node("Gemm", ["a", "b"], ["y"])
    assert model_info._node_memory_access(node, {}, {}) == 0


def test_compute_density_is_flops_over_bytes():
    body = """
    g (float[5,7] a, float[7,3] b) => (float[5,3] y)
    {
      y = Gemm(a, b)
    }
    """
    info = _info(body)
    assert info.compute_density == info.flops / info.mem_access


def test_compute_density_zero_when_no_traffic():
    # A shapeless model has no measurable traffic, so density is 0 (not a crash).
    body = """
    g (float[] x) => (float[] y)
    {
      y = Relu(x)
    }
    """
    info = _info(body)
    assert info.mem_access == 0
    assert info.compute_density == 0


def test_memory_footprint_single_node():
    # Gemm: inputs a, b live through the node and output y is produced -> peak is
    # every tensor resident at once.
    body = """
    g (float[5,7] a, float[7,3] b) => (float[5,3] y)
    {
      y = Gemm(a, b)
    }
    """
    info = _info(body)
    assert info.memory_footprint == (5 * 7 + 7 * 3 + 5 * 3) * 4


def test_memory_footprint_reuses_freed_activations():
    # x -> Conv -> h -> Relu -> y. x is dead after Conv, so it is not resident
    # during Relu; the peak is the larger of the two per-node working sets, not
    # the sum of every tensor.
    body = """
    g (float[1,3,8,8] x) => (float[1,4,8,8] y)
    {
      h = Conv<kernel_shape=[3,3], pads=[1,1,1,1]>(x, w)
      y = Relu(h)
    }
    """
    info = _info(body, [_weight([4, 3, 3, 3], "w")])
    b = 4  # float32
    weight_bytes = 4 * 3 * 3 * 3 * b
    x_bytes = 1 * 3 * 8 * 8 * b
    h_bytes = 1 * 4 * 8 * 8 * b
    y_bytes = 1 * 4 * 8 * 8 * b
    conv_peak = weight_bytes + x_bytes + h_bytes  # x still live, h produced
    relu_peak = weight_bytes + h_bytes + y_bytes  # x freed, y produced
    assert info.memory_footprint == max(conv_peak, relu_peak)
    # ... and strictly less than holding every tensor at once.
    assert info.memory_footprint < weight_bytes + x_bytes + h_bytes + y_bytes


def test_memory_metrics_symbolic_dynamic_batch():
    sympy = pytest.importorskip("sympy")
    body = """
    g (float[batch,7] a) => (float[batch,3] y)
    {
      y = MatMul(a, b)
    }
    """
    info = _info(body, [_weight([7, 3], "b")])
    batch = sympy.Symbol("batch", positive=True, integer=True)
    expected = (7 * batch + 3 * batch) * 4 + (
        7 * 3
    ) * 4  # a + y scale with batch; b fixed
    assert sympy.simplify(info.mem_access - expected) == 0


def test_memory_metrics_reported_in_summary(capsys):
    body = """
    g (float[5,7] a, float[7,3] b) => (float[5,3] y)
    {
      y = Gemm(a, b)
    }
    """
    model = _model(body)
    model_info.print_simplifying_info(model, model)
    out = capsys.readouterr().out
    assert "Memory Access" in out
    assert "Memory Footprint" in out
    assert "Compute Density" in out


# --------------------------------------------------------------------------- #
# Storing metrics into metadata_props
# --------------------------------------------------------------------------- #
def _meta(proto):
    return {e.key: e.value for e in proto.metadata_props}


def _gemm_model():
    body = """
    g (float[5,7] a, float[7,3] b) => (float[5,3] y)
    {
      [gemm0] y = Gemm(a, b)
    }
    """
    return _model(body)


def test_annotate_metadata_model_level():
    model = _gemm_model()
    info = ModelInfo(model)
    out = annotate_metadata(model)
    meta = _meta(out)
    p = METADATA_PREFIX
    assert meta[p + "macs"] == str(info.macs)
    assert meta[p + "flops"] == str(info.flops)
    assert meta[p + "mem_access"] == str(info.mem_access)
    assert meta[p + "memory_footprint"] == str(info.memory_footprint)
    assert meta[p + "model_size"] == str(info.model_size)
    assert p + "compute_density" in meta


def test_annotate_metadata_node_level():
    out = annotate_metadata(_gemm_model())
    (node,) = out.graph.node
    meta = _meta(node)
    p = METADATA_PREFIX
    assert meta[p + "macs"] == str(5 * 3 * 7)
    assert meta[p + "flops"] == str(2 * 5 * 3 * 7)
    assert meta[p + "mem_access"] == str((5 * 7 + 7 * 3 + 5 * 3) * 4)


def test_annotate_metadata_value_level():
    out = annotate_metadata(_gemm_model())
    by_name = {
        vi.name: _meta(vi) for vi in list(out.graph.input) + list(out.graph.output)
    }
    p = METADATA_PREFIX
    assert by_name["a"][p + "bytes"] == str(5 * 7 * 4)
    assert by_name["b"][p + "bytes"] == str(7 * 3 * 4)
    assert by_name["y"][p + "bytes"] == str(5 * 3 * 4)


def test_annotate_metadata_annotates_initializers():
    body = """
    g (float[4,8] a) => (float[4,16] y)
    {
      y = MatMul(a, b)
    }
    """
    model = _model(body, [_weight([8, 16], "b")])
    out = annotate_metadata(model)
    (init,) = out.graph.initializer
    assert _meta(init)[METADATA_PREFIX + "bytes"] == str(8 * 16 * 4)


def test_annotate_metadata_does_not_mutate_input():
    model = _gemm_model()
    annotate_metadata(model)
    assert len(model.metadata_props) == 0
    assert all(len(n.metadata_props) == 0 for n in model.graph.node)
    assert all(len(vi.metadata_props) == 0 for vi in model.graph.input)


def test_annotate_metadata_output_passes_checker():
    onnx.checker.check_model(annotate_metadata(_gemm_model()))


def test_annotate_metadata_custom_prefix():
    out = annotate_metadata(_gemm_model(), prefix="myprefix.")
    assert "myprefix.macs" in _meta(out)
    assert "myprefix.macs" in _meta(out.graph.node[0])


def test_annotate_metadata_symbolic_stores_formula():
    pytest.importorskip("sympy")
    body = """
    g (float[batch,7] a) => (float[batch,3] y)
    {
      [mm] y = MatMul(a, b)
    }
    """
    model = _model(body, [_weight([7, 3], "b")])
    out = annotate_metadata(model)
    assert _meta(out.graph.node[0])[METADATA_PREFIX + "macs"] == "21*batch"
    a_vi = next(vi for vi in out.graph.input if vi.name == "a")
    assert _meta(a_vi)[METADATA_PREFIX + "bytes"] == "28*batch"


def test_annotate_metadata_recurses_subgraphs():
    # An If whose branches each hold a Gemm: the node inside a branch must be
    # annotated, and the model total must include that branch's MACs.
    body = """
    g (bool cond) => (float[5,3] y)
    {
      y = If (cond) <
        then_branch = then_graph () => (float[5,3] y)
        {
          [gemm_t] y = Gemm(a_t, b_t)
        },
        else_branch = else_graph () => (float[5,3] y)
        {
          [gemm_e] y = Gemm(a_e, b_e)
        }
      >
    }
    """
    model = _model(body)
    if_node = model.graph.node[0]
    then_g = next(a.g for a in if_node.attribute if a.name == "then_branch")
    else_g = next(a.g for a in if_node.attribute if a.name == "else_branch")
    # The branch subgraphs resolve a_t/b_t (a_e/b_e) purely from their own
    # initializers -- an If's branches take no formal inputs.
    then_g.initializer.extend([_weight([5, 7], "a_t"), _weight([7, 3], "b_t")])
    else_g.initializer.extend([_weight([5, 7], "a_e"), _weight([7, 3], "b_e")])

    out = annotate_metadata(model)
    # Locate the Gemm inside the then-branch and check it carries metrics.
    then_out_g = next(
        a.g for a in out.graph.node[0].attribute if a.name == "then_branch"
    )
    gemm = next(n for n in then_out_g.node if n.op_type == "Gemm")
    assert _meta(gemm)[METADATA_PREFIX + "macs"] == str(5 * 3 * 7)
    # Model total counts both branches' Gemms.
    assert _meta(out)[METADATA_PREFIX + "macs"] == str(2 * 5 * 3 * 7)


# --------------------------------------------------------------------------- #
# Warnings replace silent failures
# --------------------------------------------------------------------------- #
def test_warns_when_shape_inference_fails(monkeypatch):
    def boom(_model):
        raise RuntimeError("boom")

    monkeypatch.setattr(model_info.shape_inference, "infer_shapes", boom)
    # Disable the optional onnx-shape-inference backend so this exercises the
    # plain onnx.shape_inference path (and its failure) regardless of whether
    # onnx-shape-inference happens to be installed.
    monkeypatch.setattr(model_info, "infer_symbolic_shapes", None)
    body = """
    g (float[1,4] x) => (float[1,4] y)
    {
      y = Relu(x)
    }
    """
    with pytest.warns(UserWarning, match="Shape inference failed"):
        ModelInfo(_model(body))


# There is no "warn when a MAC counter raises" test: the counting is delegated
# to the C++ implementation, whose counters are bounds-checked and return 0
# rather than throwing, so a malformed node degrades to 0 without a per-counter
# Python warning (the graceful-degradation behaviour is preserved structurally;
# see the "macs == 0 when shapes are unknown" cases above).


# --------------------------------------------------------------------------- #
# Model-local function operators are counted via their bodies
# --------------------------------------------------------------------------- #
_MY_LINEAR_FUNCTION = """
<
  domain: "custom",
  opset_import: ["": 18]
>
MyLinear (x, w) => (y)
{
  y = MatMul(x, w)
}
"""

_BLOCK_FUNCTION = """
<
  domain: "custom",
  opset_import: ["": 18, "custom": 1]
>
Block (x, w) => (y)
{
  t = custom.MyLinear(x, w)
  y = Relu(t)
}
"""


def _function_model(body, initializer, functions):
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 18, "custom": 1]
        >
        {body}
        {functions}
        """
    )
    model.graph.initializer.extend(initializer)
    onnx.checker.check_model(model)
    return model


def test_function_body_counted_and_op_kept():
    # MyLinear used twice: op counts show the function op, MACs sum the bodies.
    body = """
    g (float[4,8] X) => (float[4,32] Y)
    {
      H = custom.MyLinear(X, W1)
      Y = custom.MyLinear(H, W2)
    }
    """
    model = _function_model(
        body, [_weight([8, 16], "W1"), _weight([16, 32], "W2")], _MY_LINEAR_FUNCTION
    )
    info = ModelInfo(model)
    assert info.op_nums["MyLinear"] == 2  # op count keeps the function op
    assert info.macs == 4 * 16 * 8 + 4 * 32 * 16  # MACs come from the bodies


def test_nested_function_body_counted():
    # Block(x, w) -> MyLinear(x, w) -> Relu; nested functions are inlined too.
    body = """
    g (float[4,8] X) => (float[4,16] Y)
    {
      Y = custom.Block(X, W1)
    }
    """
    model = _function_model(
        body, [_weight([8, 16], "W1")], _MY_LINEAR_FUNCTION + _BLOCK_FUNCTION
    )
    info = ModelInfo(model)
    assert info.op_nums["Block"] == 1
    assert info.macs == 4 * 16 * 8


def test_warns_when_inlining_fails(monkeypatch):
    import onnx.inliner

    def boom(_model):
        raise RuntimeError("inliner boom")

    monkeypatch.setattr(onnx.inliner, "inline_local_functions", boom)
    body = """
    g (float[4,8] X) => (float[4,16] Y)
    {
      Y = custom.MyLinear(X, W1)
    }
    """
    model = _function_model(body, [_weight([8, 16], "W1")], _MY_LINEAR_FUNCTION)
    with pytest.warns(UserWarning, match="Failed to expand function bodies"):
        info = ModelInfo(model)
    assert info.macs == 0  # body compute uncounted, but no crash


def test_schema_function_fallback_counts_attention(monkeypatch):
    # Drop Attention's bespoke counter so the generic schema-function fallback
    # must expand its context-dependent body and count the two internal MatMuls.
    monkeypatch.delitem(model_info._MAC_COUNTERS, "Attention")
    b, h, sq, d = 2, 8, 16, 64
    body = f"""
    g (float[{b},{h},{sq},{d}] Q, float[{b},{h},{sq},{d}] K,
       float[{b},{h},{sq},{d}] V) => (float[{b},{h},{sq},{d}] Y)
    {{
      Y = Attention(Q, K, V)
    }}
    """
    info = ModelInfo(_model(body, opset=24))
    assert info.op_nums["Attention"] == 1  # op count still shows the op
    assert info.macs == b * h * sq * sq * d * 2  # exact, via the expanded body


# --------------------------------------------------------------------------- #
# Graph diff: node/value level diff between original and simplified graphs
# --------------------------------------------------------------------------- #
def test_diff_graphs_unchanged_model_is_empty():
    body = """
    g (float[1,4] x) => (float[1,4] y)
    {
      y = Relu(x)
    }
    """
    model = _model(body)
    diff = model_info.diff_graphs(model, model)
    assert diff.removed_nodes == []
    assert diff.added_nodes == []
    assert diff.changed_nodes == []
    assert diff.removed_values == []
    assert diff.added_values == []


def test_diff_graphs_detects_removed_and_added_nodes():
    # x -> Identity -> Relu -> y  becomes  x -> Relu -> y (Identity eliminated).
    ori = _model(
        """
        g (float[1,4] x) => (float[1,4] y)
        {
          [id0] mid = Identity(x)
          [relu0] y = Relu(mid)
        }
        """
    )
    opt = _model(
        """
        g (float[1,4] x) => (float[1,4] y)
        {
          [relu0] y = Relu(x)
        }
        """
    )

    diff = model_info.diff_graphs(ori, opt)
    assert [n.name for n in diff.removed_nodes] == ["id0"]
    assert diff.added_nodes == []
    # relu0 kept its output name "y" but its inputs changed (mid -> x).
    assert len(diff.changed_nodes) == 1
    before, after = diff.changed_nodes[0]
    assert before.inputs == ("mid",)
    assert after.inputs == ("x",)
    assert diff.removed_values == ["mid"]
    assert diff.added_values == []


def test_diff_graphs_added_node_from_constant_folding():
    # A Shape node folded away and replaced by a Constant under a new name.
    ori = _model(
        """
        g (float[1,4] x) => (float[1] y)
        {
          [shape0] y = Shape(x)
        }
        """
    )
    opt = _model(
        """
        g (float[1,4] x) => (float[1] y_folded)
        {
          [const0] y_folded = Constant<value = int64[1]{4}>()
        }
        """
    )
    diff = model_info.diff_graphs(ori, opt)
    assert [n.name for n in diff.removed_nodes] == ["shape0"]
    assert [n.name for n in diff.added_nodes] == ["const0"]
    assert diff.changed_nodes == []
    assert diff.removed_values == ["y"]
    assert diff.added_values == ["y_folded"]


def test_diff_graphs_detects_op_type_change():
    # Same output name "y", but the producing op_type changed.
    ori = _model(
        """
        g (float[1,4] x) => (float[1,4] y)
        {
          [n] y = Relu(x)
        }
        """
    )
    opt = _model(
        """
        g (float[1,4] x) => (float[1,4] y)
        {
          [n] y = Sigmoid(x)
        }
        """
    )
    diff = model_info.diff_graphs(ori, opt)
    assert diff.removed_nodes == []
    assert diff.added_nodes == []
    (before, after) = diff.changed_nodes[0]
    assert before.op_type == "Relu"
    assert after.op_type == "Sigmoid"


def test_diff_graphs_removed_and_added_initializers():
    ori = _model(
        """
        g (float[4,8] x) => (float[4,16] y)
        {
          y = MatMul(x, b)
        }
        """,
        [_weight([8, 16], "b")],
    )
    opt = _model(
        """
        g (float[4,8] x) => (float[4,16] y)
        {
          y = MatMul(x, b_folded)
        }
        """,
        [_weight([8, 16], "b_folded")],
    )
    diff = model_info.diff_graphs(ori, opt)
    assert diff.removed_values == ["b"]
    assert diff.added_values == ["b_folded"]


def test_print_graph_diff_does_not_raise(capsys):
    ori = _model(
        """
        g (float[1,4] x) => (float[1,4] y)
        {
          [id0] mid = Identity(x)
          [relu0] y = Relu(mid)
        }
        """
    )
    opt = _model(
        """
        g (float[1,4] x) => (float[1,4] y)
        {
          [relu0] y = Relu(x)
        }
        """
    )
    model_info.print_graph_diff(ori, opt)
    out = capsys.readouterr().out
    assert "Nodes removed" in out
    assert "id0" in out
    assert "Values removed" in out
    assert "mid" in out


def test_print_graph_diff_caps_output(capsys):
    ori = _model(
        """
        g (float[1,4] x) => (float[1,4] mid0)
        {
          [id0] mid0 = Identity(x)
          [id1] mid1 = Identity(x)
          [id2] mid2 = Identity(x)
          [id3] mid3 = Identity(x)
          [id4] mid4 = Identity(x)
        }
        """
    )
    opt = _model(
        """
        g (float[1,4] x) => (float[1,4] x)
        {
        }
        """
    )
    model_info.print_graph_diff(ori, opt, limit=2)
    out = capsys.readouterr().out
    assert "... and" in out


def test_schema_function_fallback_layernorm_is_zero():
    # A context-dependent function with no matmuls must expand cleanly to 0 MACs.
    body = """
    g (float[4,16] X) => (float[4,16] Y)
    {
      Y = LayerNormalization(X, scale, bias)
    }
    """
    model = _model(body, [_weight([16], "scale"), _weight([16], "bias")], opset=17)
    assert ModelInfo(model).macs == 0
