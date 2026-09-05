"""What Pulsar2's static AX650 coverage heuristic can and can't tell apart,
across the many ONNX shapes that all lower to "a Conv" or "a MatMul."

`scripts/axera/pulsar2_simulator.partition()` (see that module's own
docstring) classifies a node as NPU-eligible purely by
`pulsar2_ops.AX650_SUPPORTED_OPS` op-*type* membership -- it does not look at
attributes at all. This is a known, already-documented limitation (Pulsar2's
own docs mention per-op attribute limits, e.g. "Conv's auto_pad must be
NOTSET", that this harness has never had real hardware/Docker access to
verify). This suite makes the resulting blind spot concrete: standard,
grouped, depthwise, dilated, strided and `auto_pad`-using Conv, 1-D and 3-D
Conv, and `ConvTranspose` are all indistinguishable to `partition()` (all
"npu", op_type `Conv`/`ConvTranspose`), and the same is true across `MatMul`,
broadcasting/batched `MatMul`, and `Gemm` with any combination of
`alpha`/`beta`/`transA`/`transB`.

It also pins down one thing that *is* a real, checkable-without-hardware
finding: Pulsar2's confirmed `AX650_SUPPORTED_OPS` list (scraped from its own
public docs) does not include any of ONNX's own quantized conv/matmul ops
(`QLinearConv`, `ConvInteger`, `QLinearMatMul`, `MatMulInteger`) -- consistent
with `pulsar2_quantizer.py`'s docstring, which found that Pulsar2's real INT8
pipeline quantizes to its own proprietary `AxQuantizedConv`-family ops, not
standard ONNX quantized operators. A graph pre-quantized with those standard
ops (e.g. via `onnxsim.quantize_dynamic`/`onnxruntime.quantization`) reads as
an AX650 build risk under the current heuristic, op type by itself.

No real Pulsar2/Docker/device access is used or claimed anywhere in this
file -- every assertion below is reproducible from `scripts/axera/pulsar2_ops.py`
and `pulsar2_simulator.py`'s existing, checked-in heuristic alone.
"""

import os
import sys

import onnx
from onnx import parser

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import pulsar2_backend as backend  # noqa: E402
import pulsar2_ops as ops  # noqa: E402
import pulsar2_simulator as sim  # noqa: E402


def _model(body, opset=13, ir_version=10):
    return parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )


# --- Conv-family variants: all opset-13, all group into the same op_type ---

_CONV_VARIANTS = {
    "conv_standard": """
        g (float[1,8,10,10] X, float[8,8,3,3] W) => (float[1,8,8,8] Y)
        { Y = Conv(X, W) }
    """,
    "conv_grouped": """
        g (float[1,8,10,10] X, float[8,2,3,3] W) => (float[1,8,8,8] Y)
        { Y = Conv<group = 4>(X, W) }
    """,
    "conv_depthwise": """
        g (float[1,8,10,10] X, float[8,1,3,3] W) => (float[1,8,8,8] Y)
        { Y = Conv<group = 8>(X, W) }
    """,
    "conv_dilated": """
        g (float[1,8,10,10] X, float[8,8,3,3] W) => (float[1,8,6,6] Y)
        { Y = Conv<dilations = [2, 2]>(X, W) }
    """,
    "conv_strided": """
        g (float[1,8,10,10] X, float[8,8,3,3] W) => (float[1,8,4,4] Y)
        { Y = Conv<strides = [2, 2]>(X, W) }
    """,
    "conv_same_upper": """
        g (float[1,8,10,10] X, float[8,8,3,3] W) => (float[1,8,10,10] Y)
        { Y = Conv<auto_pad = "SAME_UPPER">(X, W) }
    """,
    "conv_1d": """
        g (float[1,8,10] X, float[8,8,3] W) => (float[1,8,8] Y)
        { Y = Conv(X, W) }
    """,
    "conv_3d": """
        g (float[1,8,10,10,10] X, float[8,8,3,3,3] W) => (float[1,8,8,8,8] Y)
        { Y = Conv(X, W) }
    """,
    "conv_transpose": """
        g (float[1,8,8,8] X, float[8,8,3,3] W) => (float[1,8,10,10] Y)
        { Y = ConvTranspose(X, W) }
    """,
}

# Quantized conv variants: standard ONNX quantized ops, not Pulsar2's own
# proprietary AxQuantizedConv -- see this module's docstring.
_QUANT_CONV_VARIANTS = {
    "conv_integer": """
        g (uint8[1,8,10,10] X, uint8[8,8,3,3] W) => (int32[1,8,8,8] Y)
        { Y = ConvInteger(X, W) }
    """,
    "qlinear_conv": """
        g (uint8[1,8,10,10] X) => (uint8[1,8,8,8] Y)
        <
          float x_scale = {0.5},
          uint8 x_zero_point = {0},
          float[8,8,3,3] w = {0.0},
          float w_scale = {0.5},
          uint8 w_zero_point = {0},
          float y_scale = {0.5},
          uint8 y_zero_point = {0}
        >
        { Y = QLinearConv(X, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point) }
    """,
}

_MATMUL_VARIANTS = {
    "matmul_2d": """
        g (float[4,8] A, float[8,16] B) => (float[4,16] Y)
        { Y = MatMul(A, B) }
    """,
    "matmul_batched_broadcast": """
        g (float[2,4,8] A, float[8,16] B) => (float[2,4,16] Y)
        { Y = MatMul(A, B) }
    """,
    "gemm_no_trans": """
        g (float[4,8] A, float[8,16] B) => (float[4,16] Y)
        { Y = Gemm(A, B) }
    """,
    "gemm_transb": """
        g (float[4,8] A, float[16,8] B) => (float[4,16] Y)
        { Y = Gemm<transB = 1>(A, B) }
    """,
    "gemm_alpha_beta_bias": """
        g (float[4,8] A, float[8,16] B, float[16] C) => (float[4,16] Y)
        { Y = Gemm<alpha = 0.5, beta = 0.5>(A, B, C) }
    """,
}

_QUANT_MATMUL_VARIANTS = {
    "matmul_integer": """
        g (uint8[4,8] A, uint8[8,16] B) => (int32[4,16] Y)
        { Y = MatMulInteger(A, B) }
    """,
    "qlinear_matmul": """
        g (uint8[4,8] A) => (uint8[4,16] Y)
        <
          float a_scale = {0.5},
          uint8 a_zero_point = {0},
          float[8,16] b = {0.0},
          float b_scale = {0.5},
          uint8 b_zero_point = {0},
          float y_scale = {0.5},
          uint8 y_zero_point = {0}
        >
        { Y = QLinearMatMul(A, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point) }
    """,
}


def _build(name: str) -> onnx.ModelProto:
    body = {
        **_CONV_VARIANTS,
        **_QUANT_CONV_VARIANTS,
        **_MATMUL_VARIANTS,
        **_QUANT_MATMUL_VARIANTS,
    }[name]
    return _model(body)


def test_all_variants_are_well_formed():
    for name in {
        **_CONV_VARIANTS,
        **_QUANT_CONV_VARIANTS,
        **_MATMUL_VARIANTS,
        **_QUANT_MATMUL_VARIANTS,
    }:
        onnx.checker.check_model(_build(name))


def test_conv_variants_are_indistinguishable_to_the_heuristic():
    """`partition()` reports full NPU coverage for every plain-float Conv
    shape, regardless of group/dilation/stride/auto_pad/rank -- it only looks
    at `node.op_type`. This is the exact "does not model attribute-level
    limits" gap `pulsar2_simulator.py`'s own docstring already calls out.

    `conv_transpose` is excluded: `ConvTranspose` is one of the 7 ops
    confirmed via real hardware to hard-fail despite being listed in
    `AX650_SUPPORTED_OPS` (`pulsar2_ops.AX650_CONFIRMED_BROKEN_OPS`), so the
    heuristic now correctly distinguishes it -- see the dedicated test below.
    """
    for name in _CONV_VARIANTS:
        if name == "conv_transpose":
            continue
        model = _build(name)
        assert sim.coverage(model) == "full", name
        assert backend.ax650_build_risks(model) == [], name


def test_conv_transpose_is_flagged_as_confirmed_broken():
    """Unlike the other Conv variants above, `ConvTranspose` IS distinguished
    by the heuristic now: it's listed in `AX650_SUPPORTED_OPS` but confirmed
    via real hardware to hard-fail a real build anyway (real
    `RuntimeError("Op Execution Error...")` during quantization -- see
    `pulsar2_ops.AX650_CONFIRMED_BROKEN_OPS`)."""
    model = _build("conv_transpose")
    assert sim.coverage(model) == "none"
    risks = backend.ax650_build_risks(model)
    assert any("ConvTranspose" in r and "confirmed to hard-fail" in r for r in risks)


def test_matmul_variants_are_indistinguishable_to_the_heuristic():
    """Same blind spot for the MatMul/Gemm family: broadcasting shape and
    alpha/beta/transA/transB are all invisible to op-type-only partitioning."""
    for name in _MATMUL_VARIANTS:
        model = _build(name)
        assert sim.coverage(model) == "full", name
        assert backend.ax650_build_risks(model) == [], name


def test_standard_onnx_quantized_conv_matmul_ops_are_not_on_ax650_list():
    """`QLinearConv`/`ConvInteger`/`QLinearMatMul`/`MatMulInteger` (ONNX's own
    quantized-op vocabulary) are absent from the real, docs-scraped
    `AX650_SUPPORTED_OPS` -- consistent with `pulsar2_quantizer.py`'s finding
    that Pulsar2's real PTQ output uses its own non-standard
    `AxQuantizedConv`-family ops instead. A graph already quantized with
    ONNX's standard quantized ops (e.g. via `onnxsim.quantize_dynamic`) reads
    as an AX650 build risk under this heuristic today."""
    for name in {**_QUANT_CONV_VARIANTS, **_QUANT_MATMUL_VARIANTS}:
        model = _build(name)
        op_type = model.graph.node[0].op_type
        assert op_type not in ops.AX650_SUPPORTED_OPS, name
        assert sim.coverage(model) == "none", name
        risks = backend.ax650_build_risks(model)
        assert any(op_type in risk for risk in risks), (name, risks)


def test_plain_conv_and_matmul_op_types_are_on_ax650_list():
    # Sanity check on the two op types this whole file is about: the
    # unquantized forms are on the list (this is what makes the "full
    # coverage regardless of attributes" finding above interesting rather
    # than trivial -- if Conv/MatMul weren't supported at all, every variant
    # would already read as "none").
    assert "Conv" in ops.AX650_SUPPORTED_OPS
    assert "ConvTranspose" in ops.AX650_SUPPORTED_OPS
    assert "MatMul" in ops.AX650_SUPPORTED_OPS
    assert "Gemm" in ops.AX650_SUPPORTED_OPS
