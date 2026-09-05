"""Tests for ``onnxsim.apply_qmoe_expert_channel_pruning_cpp`` -- the C++-
backed port of ``onnxsim.apply_qmoe_expert_channel_pruning`` (see
``onnxsim/structured_pruning_entry.cpp``'s own "QMoE (com.microsoft,
quantized-weight Mixture-of-Experts) expert-channel structured pruning"
section). Scope note: only the EXPERT-CHANNEL half is ported -- the
complementary WHOLE-EXPERT half (``apply_qmoe_whole_expert_pruning``) needs
runtime calibration data (a real ONNX Runtime inference session observing
router activations), which this C++ port has no ONNX Runtime linked into at
all (see the repo's own CLAUDE.md: the wheel build never builds ORT), so it
stays out of scope here too.

Covers ``quant_type='int'`` (whole-row and blockwise) and
``quant_type='nvfp4'`` (schema-derived only -- see below). Tests here mirror
``tests/test_pruning.py``'s own QMoE expert-channel coverage (the Python
reference this C++ pass is a port of), reusing the exact same
"onnxsim-code-free" reference quantizers/model builders that file already
established (each one re-derived independently, never importing anything
from ``onnxsim.pruning`` itself, so a test comparing against it is a genuine
cross-check).

QMoE has no ``onnx.reference`` fallback (a custom-domain op with no
decomposition attached), so every ``quant_type='int'`` test that actually
prunes something runs the result through a real ``onnxruntime`` CPU
session -- the only real oracle available for this op. QMoE also can't be
built via ``onnx.parser``'s text format at all (a custom-domain op whose
``fc1``/``fc2`` weights are packed ``uint8`` tensors -- the parser's own
tensor-literal syntax encodes ``float_data``, not ``raw_data``, and has no
uint8 literal syntax either), so every QMoE model below is built via
``onnx.helper.make_node``/``make_tensor`` directly, per CLAUDE.md's own
documented fallback for exactly this kind of case.

This environment's onnxruntime (1.29.0) has NO CPU kernel at all for
``quant_type='nvfp4'`` (confirmed directly: a hand-built, otherwise
schema-valid ``QMoE('nvfp4')`` node fails at
``onnxruntime.InferenceSession(...)`` construction with "NOT_IMPLEMENTED").
So, unlike the ``quant_type='int'`` tests (each run through a real ORT CPU
session), every nvfp4 test instead checks (a) that this pass's own output is
byte-exact-identical to slicing the already-quantized reference tensors
directly (never re-derived), the same "slice, don't recompute" bar every
other test in this file is held to, and (b) that dequantizing the pruned
tensors reproduces exactly what slicing the full dequantized reference would
give -- the "slice commutes with dequant" property this quant_type's own
packed/block-scaled format needs to hold for pruning to be safe at all.
"""

import ml_dtypes
import numpy as np
import onnx
import onnx.checker
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _f16(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float16), name)


def _bf16(array, name):
    return onnx.numpy_helper.from_array(array.astype(ml_dtypes.bfloat16), name)


# Maps a numpy dtype (as passed to _qmoe_model's own `float_dtype`) to the
# matching (ONNX enum, tensor-builder) pair -- lets _qmoe_model build an
# activation-dtype-FLOAT16/BFLOAT16 QMoE model for the FP16/BFloat16 weight
# support tests below, while every existing FLOAT32 call site (the default)
# is completely unaffected.
_FLOAT_DTYPE_INFO = {
    np.dtype(np.float32): (onnx.TensorProto.FLOAT, _f32),
    np.dtype(np.float16): (onnx.TensorProto.FLOAT16, _f16),
    np.dtype(ml_dtypes.bfloat16): (onnx.TensorProto.BFLOAT16, _bf16),
}


def _u8(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.uint8), name)


def _f8e4m3(array, name):
    return onnx.numpy_helper.from_array(array.astype(ml_dtypes.float8_e4m3fn), name)


# --- quant_type='int' reference quantizer/packer (onnxsim-code-free) -------


def _qmoe_quantize_channel(w, bits, zero_point=None):
    """Reference per-channel symmetric QMoE quantizer: `w` is one expert's
    own ``[N, K]`` float weight. Returns ``(packed [N, K/pack] uint8, scale
    [N] float32, dequant [N, K] float32)`` -- packed low-index-in-low-bits,
    ``8 // bits`` values per byte, this op's own confirmed-live raw storage
    convention. `zero_point`, when given, is a per-channel ``[N]`` int
    array; otherwise every channel uses the schema's own documented
    default, ``2 ** (bits - 1)``.
    """
    n, k = w.shape
    pack = 8 // bits
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    default_zp = 1 << (bits - 1)
    zp = (
        np.full(n, default_zp, dtype=np.int64)
        if zero_point is None
        else np.asarray(zero_point, dtype=np.int64)
    )
    scale = np.abs(w).max(axis=1) / float(-qmin)
    scale = np.maximum(scale, np.finfo(np.float32).eps).astype(np.float32)
    q = np.clip(np.round(w / scale[:, None]), qmin, qmax).astype(np.int64) + zp[:, None]
    q = np.clip(q, 0, (1 << bits) - 1).astype(np.uint8)
    parts = [(q[:, i::pack] & ((1 << bits) - 1)) for i in range(pack)]
    packed = np.zeros_like(parts[0])
    for i, p in enumerate(parts):
        packed = packed | (p << (bits * i))
    packed = packed.astype(np.uint8)
    dequant = (q.astype(np.float32) - zp[:, None].astype(np.float32)) * scale[:, None]
    return packed, scale, dequant


def _qmoe_quantize(w, bits, zero_point=None):
    """Batched (per-expert) :func:`_qmoe_quantize_channel`: `w` is
    ``[E, N, K]``. Returns ``(packed [E, N, K/pack], scale [E, N], dequant
    [E, N, K])``.
    """
    e, n, k = w.shape
    pack = 8 // bits
    packed = np.zeros((e, n, k // pack), dtype=np.uint8)
    scale = np.zeros((e, n), dtype=np.float32)
    dequant = np.zeros_like(w)
    for ei in range(e):
        packed[ei], scale[ei], dequant[ei] = _qmoe_quantize_channel(
            w[ei], bits, zero_point=None if zero_point is None else zero_point[ei]
        )
    return packed, scale, dequant


def _qmoe_quantize_channel_blockwise(w, bits, block_size, zero_point=None):
    """Reference per-``block_size``-group symmetric QMoE quantizer for
    `block_size` set: `w` is one expert's own ``[N, K]`` float weight,
    quantized independently per ``block_size``-sized group along `K`.
    Returns ``(packed [N, K/pack] uint8, scale [N, K // block_size]
    float32, dequant [N, K] float32)`` -- weight packing is still the flat,
    block-boundary-oblivious low-index-in-low-bits convention, not
    restarted at each block.
    """
    n, k = w.shape
    pack = 8 // bits
    kb = k // block_size
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    default_zp = 1 << (bits - 1)
    zp = (
        np.full((n, kb), default_zp, dtype=np.int64)
        if zero_point is None
        else np.asarray(zero_point, dtype=np.int64)
    )
    w_blocks = w.reshape(n, kb, block_size)
    scale = np.abs(w_blocks).max(axis=-1) / float(-qmin)
    scale = np.maximum(scale, np.finfo(np.float32).eps).astype(np.float32)
    q = (
        np.clip(np.round(w_blocks / scale[..., None]), qmin, qmax).astype(np.int64)
        + zp[..., None]
    )
    q = np.clip(q, 0, (1 << bits) - 1).astype(np.uint8)
    q_flat = q.reshape(n, k)
    parts = [(q_flat[:, i::pack] & ((1 << bits) - 1)) for i in range(pack)]
    packed = np.zeros_like(parts[0])
    for i, p in enumerate(parts):
        packed = packed | (p << (bits * i))
    packed = packed.astype(np.uint8)
    dequant = (q.astype(np.float32) - zp[..., None].astype(np.float32)) * scale[
        ..., None
    ]
    dequant = dequant.reshape(n, k)
    return packed, scale, dequant


def _qmoe_quantize_blockwise(w, bits, block_size, zero_point=None):
    """Batched (per-expert) :func:`_qmoe_quantize_channel_blockwise`."""
    e, n, k = w.shape
    pack = 8 // bits
    kb = k // block_size
    packed = np.zeros((e, n, k // pack), dtype=np.uint8)
    scale = np.zeros((e, n, kb), dtype=np.float32)
    dequant = np.zeros_like(w)
    for ei in range(e):
        packed[ei], scale[ei], dequant[ei] = _qmoe_quantize_channel_blockwise(
            w[ei],
            bits,
            block_size,
            zero_point=None if zero_point is None else zero_point[ei],
        )
    return packed, scale, dequant


def _pack_vals(vals, bits):
    """Independent sub-byte packer for a generic array of already-
    quantized ``uint8`` values in ``[0, 2**bits)`` on its own last axis --
    used to build `fc1_zero_points`/`fc2_zero_points` test tensors directly.
    """
    pack = 8 // bits
    vals = np.asarray(vals, dtype=np.uint8)
    n = vals.shape[-1]
    pad = (-n) % pack
    if pad:
        vals = np.pad(vals, [(0, 0)] * (vals.ndim - 1) + [(0, pad)])
    reshaped = vals.reshape(*vals.shape[:-1], -1, pack)
    out = np.zeros(reshaped.shape[:-1], dtype=np.uint8)
    for i in range(pack):
        out = out | (reshaped[..., i] << (bits * i))
    return out.astype(np.uint8)


def _unpack_vals(packed, bits, logical_len):
    """Independent unpacker, inverse of :func:`_pack_vals`."""
    pack = 8 // bits
    mask = (1 << bits) - 1
    parts = [(packed >> (bits * i)) & mask for i in range(pack)]
    unpacked = np.stack(parts, axis=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * pack
    )
    return unpacked[..., :logical_len]


def _qmoe_model(
    fc1_q,
    fc1_scale,
    fc2_q,
    fc2_scale,
    bits,
    fc1_bias=None,
    fc2_bias=None,
    fc1_zp=None,
    fc2_zp=None,
    fc3_q=None,
    fc3_scale=None,
    activation="relu",
    swiglu_fusion=0,
    k=2,
    tokens=6,
    quant_type="int",
    block_size=0,
    weights_prepacked=None,
    router_weights=False,
    tied_fc1_weight_elsewhere=False,
    extra_nodes=(),
    extra_outputs=(),
    float_dtype=np.float32,
):
    # A router-logit input R feeds a real com.microsoft::QMoE node -- see
    # this file's own top comment for why the onnx.parser text format
    # can't express QMoE's packed uint8 operands. `float_dtype` (FLOAT32 by
    # default, every existing call site's own behavior unchanged) controls
    # X/R/Y's own activation dtype and every FLOAT-family operand's own
    # storage dtype (fc1/fc2 scale and bias) -- see the "FP16/BFloat16
    # weight support" section below, which passes FLOAT16/BFLOAT16 here to
    # build the widened-matcher test models. `fc1_q`/`fc2_q`/zero_points
    # always stay UINT8 regardless -- quantized weight codes have no
    # separate float-family storage to widen.
    onnx_dtype, float_tensor = _FLOAT_DTYPE_INFO[np.dtype(float_dtype)]
    num_experts, inter, hidden_packed = fc1_q.shape
    hidden = hidden_packed * (8 // bits)
    inputs = [
        onnx.helper.make_tensor_value_info("X", onnx_dtype, [tokens, hidden]),
        onnx.helper.make_tensor_value_info("R", onnx_dtype, [tokens, num_experts]),
    ]
    outputs = [onnx.helper.make_tensor_value_info("Y", onnx_dtype, [tokens, hidden])]
    inits = [
        _u8(fc1_q, "FC1Q"),
        float_tensor(fc1_scale, "FC1S"),
        _u8(fc2_q, "FC2Q"),
        float_tensor(fc2_scale, "FC2S"),
    ]
    node_inputs = ["X", "R", "FC1Q", "FC1S", "", "FC2Q", "FC2S", ""]
    if fc1_bias is not None:
        node_inputs[4] = "FC1B"
        inits.append(float_tensor(fc1_bias, "FC1B"))
    if fc2_bias is not None:
        node_inputs[7] = "FC2B"
        inits.append(float_tensor(fc2_bias, "FC2B"))
    while len(node_inputs) < 13:
        node_inputs.append("")
    if fc3_q is not None:
        node_inputs[8] = "FC3Q"
        node_inputs[9] = "FC3S"
        inits += [_u8(fc3_q, "FC3Q"), float_tensor(fc3_scale, "FC3S")]
    if fc1_zp is not None:
        node_inputs[11] = "FC1ZP"
        inits.append(_u8(fc1_zp, "FC1ZP"))
    if fc2_zp is not None:
        node_inputs[12] = "FC2ZP"
        inits.append(_u8(fc2_zp, "FC2ZP"))
    if router_weights:
        while len(node_inputs) < 15:
            node_inputs.append("")
        node_inputs[14] = "RWEIGHTS"
        inits.append(_f32(np.ones((tokens, num_experts), np.float32), "RWEIGHTS"))

    extra_nodes = list(extra_nodes)
    if tied_fc1_weight_elsewhere:
        extra_nodes.append(onnx.helper.make_node("Identity", ["FC1Q"], ["FC1Q2"]))
        extra_outputs = list(extra_outputs) + [
            onnx.helper.make_tensor_value_info(
                "FC1Q2", onnx.TensorProto.UINT8, list(fc1_q.shape)
            )
        ]

    kwargs = dict(
        k=k,
        activation_type=activation,
        swiglu_fusion=swiglu_fusion,
        expert_weight_bits=bits,
        quant_type=quant_type,
    )
    if block_size:
        kwargs["block_size"] = block_size
    if weights_prepacked is not None:
        kwargs["weights_prepacked"] = weights_prepacked
    node = onnx.helper.make_node(
        "QMoE", node_inputs, ["Y"], domain="com.microsoft", name="qmoe", **kwargs
    )
    graph = onnx.helper.make_graph(
        [node, *extra_nodes],
        "g",
        inputs,
        [*outputs, *extra_outputs],
        initializer=inits,
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 18),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    return model


def _qmoe_inits(model):
    return {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}


def _qmoe_block_keep_reference(importance, n, block_size, sparsity):
    # Independent re-derivation of QMoEBlockAlignedKeep's own algorithm.
    num_blocks = n // block_size
    block_importance = np.sqrt(
        np.sum(importance.reshape(num_blocks, block_size) ** 2, axis=1)
    )
    keep_blocks_count = max(1, num_blocks - round(num_blocks * sparsity))
    keep_block_idx = np.sort(np.argsort(-block_importance)[:keep_blocks_count])
    keep = np.arange(n).reshape(num_blocks, block_size)[keep_block_idx].reshape(-1)
    return keep, keep_block_idx


# --- quant_type='nvfp4' reference quantizer/packer (onnxsim-code-free) -----

_E2M1_LUT = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float64,
)


def _e2m1_encode_nearest(values):
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(-1)
    codes = np.argmin(np.abs(_E2M1_LUT[None, :] - flat[:, None]), axis=1)
    return codes.reshape(values.shape).astype(np.uint8)


def _e2m1_decode(codes):
    return _E2M1_LUT[np.asarray(codes).astype(np.int64)]


def _qmoe_nvfp4_quantize_channel_blockwise(w, global_scale, block_size=16):
    n, k = w.shape
    kb = k // block_size
    w_blocks = w.reshape(n, kb, block_size).astype(np.float64)
    amax = np.maximum(np.abs(w_blocks).max(axis=-1), 1e-12)
    block_scale_f64 = amax / (6.0 * global_scale)  # 6.0 == E2M1's own max magnitude
    block_scale_f8 = block_scale_f64.astype(ml_dtypes.float8_e4m3fn)
    block_scale_rt = block_scale_f8.astype(np.float64)
    codes = _e2m1_encode_nearest(w_blocks / (block_scale_rt[..., None] * global_scale))
    packed = _pack_vals(codes.reshape(n, k), 4)
    dequant = (_e2m1_decode(codes) * block_scale_rt[..., None] * global_scale).reshape(
        n, k
    )
    return packed, block_scale_f8, dequant


def _qmoe_nvfp4_quantize_blockwise(w, global_scale, block_size=16):
    e, n, k = w.shape
    kb = k // block_size
    packed = np.zeros((e, n, k // 2), dtype=np.uint8)
    block_scale = np.zeros((e, n, kb), dtype=ml_dtypes.float8_e4m3fn)
    dequant = np.zeros((e, n, k), dtype=np.float64)
    for ei in range(e):
        packed[ei], block_scale[ei], dequant[ei] = (
            _qmoe_nvfp4_quantize_channel_blockwise(
                w[ei], float(global_scale[ei]), block_size
            )
        )
    return packed, block_scale, dequant


def _qmoe_nvfp4_dequantize(packed, block_scale, global_scale, block_size=16):
    e, n, k_packed = packed.shape
    k = k_packed * 2
    codes = _unpack_vals(packed, 4, k).astype(np.int64)
    mag = _e2m1_decode(codes)
    kb = k // block_size
    bs = block_scale.astype(np.float64)
    gs = np.asarray(global_scale, dtype=np.float64)
    mag_blocks = mag.reshape(e, n, kb, block_size)
    dequant = mag_blocks * bs[:, :, :, None] * gs[:, None, None, None]
    return dequant.reshape(e, n, k)


def _qmoe_nvfp4_model(
    fc1_q,
    fc1_scale,
    fc2_q,
    fc2_scale,
    fc1_global_scale,
    fc2_global_scale,
    fc1_bias=None,
    fc2_bias=None,
    activation="relu",
    k=2,
    tokens=6,
    block_size=16,
    quant_type="nvfp4",
    fc1_zp=None,
):
    num_experts, inter, hidden_packed = fc1_q.shape
    hidden = hidden_packed * 2
    inputs = [
        onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [tokens, hidden]
        ),
        onnx.helper.make_tensor_value_info(
            "R", onnx.TensorProto.FLOAT, [tokens, num_experts]
        ),
    ]
    outputs = [
        onnx.helper.make_tensor_value_info(
            "Y", onnx.TensorProto.FLOAT, [tokens, hidden]
        )
    ]
    inits = [
        _u8(fc1_q, "FC1Q"),
        _f8e4m3(fc1_scale, "FC1S"),
        _u8(fc2_q, "FC2Q"),
        _f8e4m3(fc2_scale, "FC2S"),
    ]
    node_inputs = ["X", "R", "FC1Q", "FC1S", "", "FC2Q", "FC2S", ""]
    if fc1_bias is not None:
        node_inputs[4] = "FC1B"
        inits.append(_f32(fc1_bias, "FC1B"))
    if fc2_bias is not None:
        node_inputs[7] = "FC2B"
        inits.append(_f32(fc2_bias, "FC2B"))
    while len(node_inputs) < 12:
        node_inputs.append("")
    if fc1_zp is not None:
        node_inputs[11] = "FC1ZP"
        inits.append(_u8(fc1_zp, "FC1ZP"))
    while len(node_inputs) < 16:
        node_inputs.append("")
    if fc1_global_scale is not None:
        node_inputs[15] = "FC1GS"
        inits.append(_f32(fc1_global_scale, "FC1GS"))
    node_inputs.append("FC2GS" if fc2_global_scale is not None else "")
    if fc2_global_scale is not None:
        inits.append(_f32(fc2_global_scale, "FC2GS"))
    node = onnx.helper.make_node(
        "QMoE",
        node_inputs,
        ["Y"],
        domain="com.microsoft",
        name="qmoe",
        k=k,
        activation_type=activation,
        expert_weight_bits=4,
        quant_type=quant_type,
        block_size=block_size,
    )
    graph = onnx.helper.make_graph([node], "g", inputs, outputs, initializer=inits)
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 18),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    return model


def _if_wrap(inner_model):
    """Wraps `inner_model`'s own graph as BOTH branches of a new top-level
    `If` node -- verifies the C++ port's own subgraph-recursion (IterSubgraphs)
    reaches a QMoE node nested inside an If branch, mirroring
    tests/test_pruning.py's own identical helper.
    """
    g = inner_model.graph

    def _branch(name):
        return onnx.helper.make_graph(
            list(g.node),
            name,
            [],
            list(g.output),
            initializer=list(g.initializer),
            value_info=list(g.value_info),
        )

    then_graph = _branch("then_graph")
    else_graph = _branch("else_graph")
    top_out_names = [f"top__{o.name}" for o in g.output]
    if_node = onnx.helper.make_node(
        "If", ["cond"], top_out_names, then_branch=then_graph, else_branch=else_graph
    )
    cond = onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])
    top_outputs = []
    for name, o in zip(top_out_names, g.output):
        vi = onnx.ValueInfoProto()
        vi.CopyFrom(o)
        vi.name = name
        top_outputs.append(vi)

    wrap_graph = onnx.helper.make_graph([if_node], "g", [*g.input, cond], top_outputs)
    wrap_model = onnx.helper.make_model(
        wrap_graph, opset_imports=list(inner_model.opset_import)
    )
    wrap_model.ir_version = max(inner_model.ir_version, 10)
    onnx.checker.check_model(wrap_model)
    return wrap_model


def _if_branches(wrapped_model):
    if_node = next(n for n in wrapped_model.graph.node if n.op_type == "If")
    then_g = else_g = None
    for attr in if_node.attribute:
        if attr.name == "then_branch":
            then_g = attr.g
        elif attr.name == "else_branch":
            else_g = attr.g
    return then_g, else_g


# --- quant_type='int', whole-row (no block_size) ----------------------------


def test_qmoe_expert_channel_pruning_cpp_matches_hand_built_presliced_reference():
    # A dedicated, independently hand-built "already pruned" reference QMoE
    # node -- fc1/fc2 re-quantized from scratch from the *sliced* float
    # weights (never touching the C++ port's own pruned bytes), including a
    # non-default per-channel fc1_zero_points operand, so this exercises the
    # packed zero_point slicing path too, not just the default-omitted case.
    E, hidden, inter, bits, tokens = 3, 8, 12, 4, 6
    rng = np.random.default_rng(211)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    fc1_zp_vals = rng.integers(4, 12, size=(E, inter))

    fc1_q, fc1_s, fc1_dq = _qmoe_quantize(fc1_w, bits, zero_point=fc1_zp_vals)
    fc2_q, fc2_s, fc2_dq = _qmoe_quantize(fc2_w, bits)
    fc1_zp = _pack_vals(fc1_zp_vals, bits)

    model = _qmoe_model(
        fc1_q,
        fc1_s,
        fc2_q,
        fc2_s,
        bits,
        fc1_bias=fc1_b,
        fc1_zp=fc1_zp,
        k=2,
        tokens=tokens,
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    sq = (
        np.sum(fc1_dq**2, axis=(0, 2))
        + np.sum(fc2_dq**2, axis=(0, 1))
        + np.sum(fc1_b**2, axis=0)
    )
    keep = np.sort(np.argsort(-np.sqrt(sq))[:6])  # 12 - round(12*0.5) = 6

    inits = _qmoe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q[:, keep, :])
    np.testing.assert_array_equal(inits["FC1S"], fc1_s[:, keep])
    np.testing.assert_array_equal(inits["FC1B"], fc1_b[:, keep])
    np.testing.assert_array_equal(
        inits["FC1ZP"], _pack_vals(fc1_zp_vals[:, keep], bits)
    )
    expected_fc2_q = _pack_vals(_unpack_vals(fc2_q, bits, inter)[:, :, keep], bits)
    np.testing.assert_array_equal(inits["FC2Q"], expected_fc2_q)

    fc1_q_ref, fc1_s_ref, _ = _qmoe_quantize(
        fc1_w[:, keep, :], bits, zero_point=fc1_zp_vals[:, keep]
    )
    reference = _qmoe_model(
        fc1_q_ref,
        fc1_s_ref,
        expected_fc2_q,
        fc2_s,
        bits,
        fc1_bias=fc1_b[:, keep],
        fc1_zp=_pack_vals(fc1_zp_vals[:, keep], bits),
        k=2,
        tokens=tokens,
    )
    onnx.checker.check_model(reference)

    feed_rng = np.random.default_rng(213)
    feeds = {
        "X": (feed_rng.standard_normal((tokens, hidden)) * 0.2).astype(np.float32),
        "R": feed_rng.standard_normal((tokens, E)).astype(np.float32),
    }
    (out_pruned,) = _run(pruned, feeds)
    (out_ref,) = _run(reference, feeds)
    np.testing.assert_allclose(out_pruned, out_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bits", [2, 8])
def test_qmoe_expert_channel_pruning_cpp_matches_hand_built_reference_other_bit_widths(
    bits,
):
    # bits=8 has no packing at all (pack=1, a plain index-select on both
    # fc1/fc2), bits=2 packs four values per byte -- both go through the
    # exact same production code path (QMoEUnpackSubbyte/QMoEPackSubbyte,
    # parametrized on `bits`), so this confirms it generalizes correctly.
    E, hidden, inter, tokens = 2, 8, 8, 5
    rng = np.random.default_rng(220 + bits)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_q, fc1_s, fc1_dq = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s, fc2_dq = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_model(fc1_q, fc1_s, fc2_q, fc2_s, bits, k=1, tokens=tokens)
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    sq = np.sum(fc1_dq**2, axis=(0, 2)) + np.sum(fc2_dq**2, axis=(0, 1))
    keep = np.sort(np.argsort(-np.sqrt(sq))[:4])  # 8 - round(8*0.5) = 4

    inits = _qmoe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q[:, keep, :])
    expected_fc2_q = _pack_vals(_unpack_vals(fc2_q, bits, inter)[:, :, keep], bits)
    np.testing.assert_array_equal(inits["FC2Q"], expected_fc2_q)

    fc1_q_ref, fc1_s_ref, _ = _qmoe_quantize(fc1_w[:, keep, :], bits)
    reference = _qmoe_model(
        fc1_q_ref, fc1_s_ref, expected_fc2_q, fc2_s, bits, k=1, tokens=tokens
    )
    onnx.checker.check_model(reference)

    feed_rng = np.random.default_rng(223)
    feeds = {
        "X": (feed_rng.standard_normal((tokens, hidden)) * 0.2).astype(np.float32),
        "R": feed_rng.standard_normal((tokens, E)).astype(np.float32),
    }
    (out_pruned,) = _run(pruned, feeds)
    (out_ref,) = _run(reference, feeds)
    np.testing.assert_allclose(out_pruned, out_ref, rtol=1e-5, atol=1e-5)


def test_qmoe_expert_channel_pruning_cpp_survivor_count_floored_to_pack_multiple():
    # inter_size=10 at bits=4 (pack=2): sparsity=0.7 would naively ask for
    # 10 - round(10*0.7) = 3 survivors -- not a multiple of pack. The real
    # QMoE CPU kernel has no way to represent a partial trailing byte, so
    # this pass must round the survivor count DOWN to the nearest multiple
    # of pack (here, 2).
    E, hidden, inter, bits, tokens = 2, 6, 10, 4, 5
    rng = np.random.default_rng(229)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_q, fc1_s, fc1_dq = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s, fc2_dq = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_model(fc1_q, fc1_s, fc2_q, fc2_s, bits, k=1, tokens=tokens)
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.7)
    onnx.checker.check_model(pruned)  # would fail below if survivor count
    # weren't pack-aligned.

    inits = _qmoe_inits(pruned)
    assert inits["FC1Q"].shape == (E, 2, hidden // 2)  # floored 3 -> 2
    assert inits["FC2Q"].shape == (E, hidden, 1)

    sq = np.sum(fc1_dq**2, axis=(0, 2)) + np.sum(fc2_dq**2, axis=(0, 1))
    keep = np.sort(np.argsort(-np.sqrt(sq))[:2])
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q[:, keep, :])

    feed_rng = np.random.default_rng(233)
    feeds = {
        "X": (feed_rng.standard_normal((tokens, hidden)) * 0.2).astype(np.float32),
        "R": feed_rng.standard_normal((tokens, E)).astype(np.float32),
    }
    (out_pruned,) = _run(pruned, feeds)
    assert not np.isnan(out_pruned).any()


def test_qmoe_expert_channel_pruning_cpp_zero_sparsity_is_a_no_op():
    E, hidden, inter, bits = 2, 8, 8, 4
    rng = np.random.default_rng(240)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_q, fc1_s, _ = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s, _ = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_model(fc1_q, fc1_s, fc2_q, fc2_s, bits)
    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.0)
    inits = _qmoe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q)
    np.testing.assert_array_equal(inits["FC2Q"], fc2_q)


def test_qmoe_expert_channel_pruning_cpp_invalid_sparsity_raises():
    E, hidden, inter, bits = 2, 8, 8, 4
    rng = np.random.default_rng(241)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_q, fc1_s, _ = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s, _ = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_model(fc1_q, fc1_s, fc2_q, fc2_s, bits)
    with pytest.raises(Exception):
        onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=1.0)
    with pytest.raises(Exception):
        onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=-0.1)


def test_qmoe_expert_channel_pruning_cpp_multiple_nodes_pruned_independently():
    # Two independent QMoE nodes (disjoint weights) in one graph -- each
    # pruned to its own importance-ranked keep-set, confirming this pass
    # doesn't confuse one node's own tensors with the other's, and that the
    # touched-set bookkeeping doesn't spuriously skip the second node.
    E, hidden, inter, bits, tokens = 2, 8, 8, 4, 5
    rng = np.random.default_rng(250)
    fc1_w_a = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w_a = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_w_b = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w_b = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_q_a, fc1_s_a, fc1_dq_a = _qmoe_quantize(fc1_w_a, bits)
    fc2_q_a, fc2_s_a, fc2_dq_a = _qmoe_quantize(fc2_w_a, bits)
    fc1_q_b, fc1_s_b, fc1_dq_b = _qmoe_quantize(fc1_w_b, bits)
    fc2_q_b, fc2_s_b, fc2_dq_b = _qmoe_quantize(fc2_w_b, bits)

    node_a_inputs = ["X", "R", "FC1QA", "FC1SA", "", "FC2QA", "FC2SA", ""]
    node_b_inputs = ["Y0", "R", "FC1QB", "FC1SB", "", "FC2QB", "FC2SB", ""]
    node_a = onnx.helper.make_node(
        "QMoE",
        node_a_inputs,
        ["Y0"],
        domain="com.microsoft",
        name="qmoe_a",
        k=1,
        activation_type="relu",
        expert_weight_bits=bits,
        quant_type="int",
    )
    node_b = onnx.helper.make_node(
        "QMoE",
        node_b_inputs,
        ["Y"],
        domain="com.microsoft",
        name="qmoe_b",
        k=1,
        activation_type="relu",
        expert_weight_bits=bits,
        quant_type="int",
    )
    inits = [
        _u8(fc1_q_a, "FC1QA"),
        _f32(fc1_s_a, "FC1SA"),
        _u8(fc2_q_a, "FC2QA"),
        _f32(fc2_s_a, "FC2SA"),
        _u8(fc1_q_b, "FC1QB"),
        _f32(fc1_s_b, "FC1SB"),
        _u8(fc2_q_b, "FC2QB"),
        _f32(fc2_s_b, "FC2SB"),
    ]
    inputs = [
        onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [tokens, hidden]
        ),
        onnx.helper.make_tensor_value_info("R", onnx.TensorProto.FLOAT, [tokens, E]),
    ]
    outputs = [
        onnx.helper.make_tensor_value_info(
            "Y", onnx.TensorProto.FLOAT, [tokens, hidden]
        )
    ]
    graph = onnx.helper.make_graph(
        [node_a, node_b], "g", inputs, outputs, initializer=inits
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 18),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    sq_a = np.sum(fc1_dq_a**2, axis=(0, 2)) + np.sum(fc2_dq_a**2, axis=(0, 1))
    keep_a = np.sort(np.argsort(-np.sqrt(sq_a))[:4])
    sq_b = np.sum(fc1_dq_b**2, axis=(0, 2)) + np.sum(fc2_dq_b**2, axis=(0, 1))
    keep_b = np.sort(np.argsort(-np.sqrt(sq_b))[:4])

    out = _qmoe_inits(pruned)
    np.testing.assert_array_equal(out["FC1QA"], fc1_q_a[:, keep_a, :])
    np.testing.assert_array_equal(out["FC1QB"], fc1_q_b[:, keep_b, :])


# --- quant_type='int', blockwise ---------------------------------------


def test_qmoe_expert_channel_pruning_cpp_blockwise_int_matches_hand_built_presliced_reference():
    E, hidden, inter, bits, block_size, tokens = 2, 16, 32, 4, 16, 5
    rng = np.random.default_rng(260)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_q, fc1_s, fc1_dq = _qmoe_quantize_blockwise(fc1_w, bits, block_size)
    fc2_q, fc2_s, fc2_dq = _qmoe_quantize_blockwise(fc2_w, bits, block_size)
    model = _qmoe_model(
        fc1_q, fc1_s, fc2_q, fc2_s, bits, block_size=block_size, k=2, tokens=tokens
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.sum(fc1_dq**2, axis=(0, 2)) + np.sum(fc2_dq**2, axis=(0, 1))
    )
    # inter=32, block_size=16 -> 2 blocks; sparsity=0.5 -> keep 1 block.
    keep, keep_block_idx = _qmoe_block_keep_reference(
        importance, inter, block_size, 0.5
    )
    assert len(keep) == block_size  # block-aligned.

    inits = _qmoe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q[:, keep, :])
    np.testing.assert_array_equal(inits["FC1S"], fc1_s[:, keep, :])
    expected_fc2_q = _pack_vals(_unpack_vals(fc2_q, bits, inter)[:, :, keep], bits)
    np.testing.assert_array_equal(inits["FC2Q"], expected_fc2_q)
    np.testing.assert_array_equal(inits["FC2S"], fc2_s[:, :, keep_block_idx])

    fc1_q_ref, fc1_s_ref, _ = _qmoe_quantize_blockwise(
        fc1_w[:, keep, :], bits, block_size
    )
    reference = _qmoe_model(
        fc1_q_ref,
        fc1_s_ref,
        expected_fc2_q,
        fc2_s[:, :, keep_block_idx],
        bits,
        block_size=block_size,
        k=2,
        tokens=tokens,
    )
    onnx.checker.check_model(reference)

    feed_rng = np.random.default_rng(261)
    feeds = {
        "X": (feed_rng.standard_normal((tokens, hidden)) * 0.2).astype(np.float32),
        "R": feed_rng.standard_normal((tokens, E)).astype(np.float32),
    }
    (out_pruned,) = _run(pruned, feeds)
    (out_ref,) = _run(reference, feeds)
    np.testing.assert_allclose(out_pruned, out_ref, rtol=1e-5, atol=1e-5)


def test_qmoe_expert_channel_pruning_cpp_blockwise_int_keep_set_floored_to_block_multiple():
    # inter=48, block_size=16 (3 blocks), sparsity=0.4 -> naive channel-level
    # target = 48 - round(48*0.4) = 29 -- not a multiple of block_size (16).
    # The block-granularity computation (3 - round(3*0.4) = 2 blocks kept)
    # correctly resolves to 32 survivors, never 29.
    E, hidden, inter, bits, block_size, tokens = 2, 16, 48, 4, 16, 5
    rng = np.random.default_rng(262)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_q, fc1_s, fc1_dq = _qmoe_quantize_blockwise(fc1_w, bits, block_size)
    fc2_q, fc2_s, fc2_dq = _qmoe_quantize_blockwise(fc2_w, bits, block_size)
    model = _qmoe_model(
        fc1_q, fc1_s, fc2_q, fc2_s, bits, block_size=block_size, k=1, tokens=tokens
    )
    onnx.checker.check_model(model)

    naive_channel_target = inter - round(inter * 0.4)
    assert naive_channel_target % block_size != 0  # the hazard this test guards.

    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.4)
    onnx.checker.check_model(pruned)

    inits = _qmoe_inits(pruned)
    new_inter = inits["FC1Q"].shape[1]
    assert new_inter == 32
    assert new_inter % block_size == 0

    importance = np.sqrt(
        np.sum(fc1_dq**2, axis=(0, 2)) + np.sum(fc2_dq**2, axis=(0, 1))
    )
    keep, keep_block_idx = _qmoe_block_keep_reference(
        importance, inter, block_size, 0.4
    )
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q[:, keep, :])
    expected_fc2_q = _pack_vals(_unpack_vals(fc2_q, bits, inter)[:, :, keep], bits)
    np.testing.assert_array_equal(inits["FC2Q"], expected_fc2_q)
    np.testing.assert_array_equal(inits["FC2S"], fc2_s[:, :, keep_block_idx])


def test_qmoe_expert_channel_pruning_cpp_blockwise_int_zero_sparsity_is_a_no_op():
    E, hidden, inter, bits, block_size = 2, 16, 32, 4, 16
    rng = np.random.default_rng(263)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_q, fc1_s, _ = _qmoe_quantize_blockwise(fc1_w, bits, block_size)
    fc2_q, fc2_s, _ = _qmoe_quantize_blockwise(fc2_w, bits, block_size)
    model = _qmoe_model(fc1_q, fc1_s, fc2_q, fc2_s, bits, block_size=block_size)
    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.0)
    inits = _qmoe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q)
    np.testing.assert_array_equal(inits["FC2S"], fc2_s)


# --- declines (quant_type='int') ---------------------------------------


def test_qmoe_expert_channel_pruning_cpp_declines_fc3():
    E, hidden, inter, bits = 2, 8, 8, 4
    rng = np.random.default_rng(270)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc3_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc1_q, fc1_s, _ = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s, _ = _qmoe_quantize(fc2_w, bits)
    fc3_q, fc3_s, _ = _qmoe_quantize(fc3_w, bits)
    model = _qmoe_model(fc1_q, fc1_s, fc2_q, fc2_s, bits, fc3_q=fc3_q, fc3_scale=fc3_s)
    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _qmoe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q)


def test_qmoe_expert_channel_pruning_cpp_declines_swiglu_activation():
    E, hidden, inter, bits = 2, 8, 8, 4
    rng = np.random.default_rng(271)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_q, fc1_s, _ = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s, _ = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_model(fc1_q, fc1_s, fc2_q, fc2_s, bits, activation="swiglu")
    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _qmoe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q)


def test_qmoe_expert_channel_pruning_cpp_declines_block_size_below_kernel_floor():
    # block_size=8 is below the CPU kernel's own floor ("block_size must be
    # >= 16 when provided") -- declined outright.
    E, hidden, inter, bits = 2, 32, 16, 4
    rng = np.random.default_rng(272)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    block_size = 8
    fc1_q, fc1_s, _ = _qmoe_quantize_blockwise(fc1_w, bits, block_size)
    fc2_q, fc2_s, _ = _qmoe_quantize_blockwise(fc2_w, bits, block_size)
    model = _qmoe_model(fc1_q, fc1_s, fc2_q, fc2_s, bits, block_size=block_size)
    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _qmoe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q)


def test_qmoe_expert_channel_pruning_cpp_declines_block_size_not_dividing_evenly():
    # A non-block-aligned shape: block_size=16 doesn't divide inter_size=24
    # evenly (a partial/padded final block) -- declined the same way every
    # other shape mismatch in this pass is (fc2_scales' own expected
    # block-axis shape never matches).
    E, hidden, inter, bits, block_size = 2, 32, 24, 4, 16
    rng = np.random.default_rng(273)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_q, fc1_s, _ = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s, _ = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_model(fc1_q, fc1_s, fc2_q, fc2_s, bits, block_size=block_size)
    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _qmoe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q)


def test_qmoe_expert_channel_pruning_cpp_declines_non_int_nvfp4_quant_type():
    # 'fp8' is a genuinely different tensor format from either 'int' or
    # 'nvfp4' -- declined outright, an unsupported quantization variant.
    E, hidden, inter, bits = 2, 8, 8, 4
    rng = np.random.default_rng(274)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_q, fc1_s, _ = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s, _ = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_model(fc1_q, fc1_s, fc2_q, fc2_s, bits, quant_type="fp8")
    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _qmoe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q)


def test_qmoe_expert_channel_pruning_cpp_declines_weights_prepacked_nonraw():
    E, hidden, inter, bits = 2, 8, 8, 4
    rng = np.random.default_rng(275)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_q, fc1_s, _ = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s, _ = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_model(fc1_q, fc1_s, fc2_q, fc2_s, bits, weights_prepacked=1)
    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _qmoe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q)


def test_qmoe_expert_channel_pruning_cpp_declines_router_weights_present():
    E, hidden, inter, bits = 2, 8, 8, 4
    rng = np.random.default_rng(276)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_q, fc1_s, _ = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s, _ = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_model(fc1_q, fc1_s, fc2_q, fc2_s, bits, router_weights=True)
    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _qmoe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q)


def test_qmoe_expert_channel_pruning_cpp_declines_tied_fc1_weight():
    # fc1_experts_weights read by a second (Identity) consumer besides the
    # QMoE node itself -- an in-place resize would corrupt that other
    # consumer, so the whole chain is declined.
    E, hidden, inter, bits = 3, 6, 6, 4
    rng = np.random.default_rng(277)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_q, fc1_s, _ = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s, _ = _qmoe_quantize(fc2_w, bits)
    model = _qmoe_model(
        fc1_q, fc1_s, fc2_q, fc2_s, bits, tied_fc1_weight_elsewhere=True
    )
    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _qmoe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q)


# --- quant_type='nvfp4' -------------------------------------------------


def test_qmoe_nvfp4_expert_channel_pruning_cpp_matches_hand_built_presliced_reference():
    # Confirms production's own packed-bytes/block-scale slice is
    # byte-exact-identical to slicing the already-quantized reference
    # directly (never re-derived), AND that dequantizing the pruned tensors
    # reproduces exactly what slicing the full dequantized reference gives
    # -- the "slice commutes with dequant" property this format needs to
    # hold for pruning to be safe. fc1_global_scale/fc2_global_scale are
    # per-expert, not per-channel, so must come back completely untouched.
    E, hidden, inter, block_size, tokens = 3, 32, 64, 16, 6
    rng = np.random.default_rng(411)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float64)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float64)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    fc1_gs = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    fc2_gs = np.array([1.5, 1.0, 0.8], dtype=np.float32)

    fc1_q, fc1_s, fc1_dq = _qmoe_nvfp4_quantize_blockwise(fc1_w, fc1_gs, block_size)
    fc2_q, fc2_s, fc2_dq = _qmoe_nvfp4_quantize_blockwise(fc2_w, fc2_gs, block_size)

    model = _qmoe_nvfp4_model(
        fc1_q,
        fc1_s,
        fc2_q,
        fc2_s,
        fc1_gs,
        fc2_gs,
        fc1_bias=fc1_b,
        block_size=block_size,
        k=2,
        tokens=tokens,
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.sum(fc1_dq**2, axis=(0, 2))
        + np.sum(fc2_dq**2, axis=(0, 1))
        + np.sum(fc1_b.astype(np.float64) ** 2, axis=0)
    )
    keep, keep_block_idx = _qmoe_block_keep_reference(
        importance, inter, block_size, 0.5
    )
    assert len(keep) == 32

    inits = _qmoe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q[:, keep, :])
    np.testing.assert_array_equal(inits["FC1S"], fc1_s[:, keep, :])
    np.testing.assert_array_equal(inits["FC1B"], fc1_b[:, keep])
    expected_fc2_q = _pack_vals(_unpack_vals(fc2_q, 4, inter)[:, :, keep], 4)
    np.testing.assert_array_equal(inits["FC2Q"], expected_fc2_q)
    np.testing.assert_array_equal(inits["FC2S"], fc2_s[:, :, keep_block_idx])
    np.testing.assert_array_equal(inits["FC1GS"], fc1_gs)
    np.testing.assert_array_equal(inits["FC2GS"], fc2_gs)

    full_fc1_dequant = _qmoe_nvfp4_dequantize(fc1_q, fc1_s, fc1_gs, block_size)
    pruned_fc1_dequant = _qmoe_nvfp4_dequantize(
        inits["FC1Q"], inits["FC1S"], inits["FC1GS"], block_size
    )
    np.testing.assert_array_equal(pruned_fc1_dequant, full_fc1_dequant[:, keep, :])

    full_fc2_dequant = _qmoe_nvfp4_dequantize(fc2_q, fc2_s, fc2_gs, block_size)
    pruned_fc2_dequant = _qmoe_nvfp4_dequantize(
        inits["FC2Q"], inits["FC2S"], inits["FC2GS"], block_size
    )
    np.testing.assert_array_equal(pruned_fc2_dequant, full_fc2_dequant[:, :, keep])


def test_qmoe_nvfp4_expert_channel_pruning_cpp_keep_set_floored_to_block_multiple():
    # inter=48, block_size=16 (3 blocks), sparsity=0.4 -> naive channel-level
    # target = 48 - round(48*0.4) = 29 -- not a multiple of block_size. The
    # actual block-granularity computation resolves to 32 survivors.
    E, hidden, inter, block_size, tokens = 2, 32, 48, 16, 5
    rng = np.random.default_rng(419)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float64)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float64)
    fc1_gs = np.array([1.0, 0.7], dtype=np.float32)
    fc2_gs = np.array([1.2, 0.9], dtype=np.float32)
    fc1_q, fc1_s, fc1_dq = _qmoe_nvfp4_quantize_blockwise(fc1_w, fc1_gs, block_size)
    fc2_q, fc2_s, fc2_dq = _qmoe_nvfp4_quantize_blockwise(fc2_w, fc2_gs, block_size)
    model = _qmoe_nvfp4_model(
        fc1_q,
        fc1_s,
        fc2_q,
        fc2_s,
        fc1_gs,
        fc2_gs,
        block_size=block_size,
        k=1,
        tokens=tokens,
    )
    onnx.checker.check_model(model)

    naive_channel_target = inter - round(inter * 0.4)
    assert naive_channel_target % block_size != 0

    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.4)
    onnx.checker.check_model(pruned)

    inits = _qmoe_inits(pruned)
    new_inter = inits["FC1Q"].shape[1]
    assert new_inter == 32
    assert new_inter % block_size == 0

    importance = np.sqrt(
        np.sum(fc1_dq**2, axis=(0, 2)) + np.sum(fc2_dq**2, axis=(0, 1))
    )
    keep, keep_block_idx = _qmoe_block_keep_reference(
        importance, inter, block_size, 0.4
    )
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q[:, keep, :])
    expected_fc2_q = _pack_vals(_unpack_vals(fc2_q, 4, inter)[:, :, keep], 4)
    np.testing.assert_array_equal(inits["FC2Q"], expected_fc2_q)
    np.testing.assert_array_equal(inits["FC2S"], fc2_s[:, :, keep_block_idx])


def test_qmoe_expert_channel_pruning_cpp_declines_fp8_quant_type_with_nvfp4_shape():
    # quant_type='fp8' remains out of scope even when every other input is
    # shaped exactly like a valid nvfp4 chain -- the matcher's own
    # quant_type check runs before any of those shapes are even inspected.
    E, hidden, inter, block_size = 2, 32, 32, 16
    rng = np.random.default_rng(421)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float64)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float64)
    fc1_gs = np.array([1.0, 1.0], dtype=np.float32)
    fc2_gs = np.array([1.0, 1.0], dtype=np.float32)
    fc1_q, fc1_s, _ = _qmoe_nvfp4_quantize_blockwise(fc1_w, fc1_gs, block_size)
    fc2_q, fc2_s, _ = _qmoe_nvfp4_quantize_blockwise(fc2_w, fc2_gs, block_size)
    model = _qmoe_nvfp4_model(
        fc1_q,
        fc1_s,
        fc2_q,
        fc2_s,
        fc1_gs,
        fc2_gs,
        block_size=block_size,
        quant_type="fp8",
    )
    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _qmoe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q)


def test_qmoe_expert_channel_pruning_cpp_declines_nvfp4_zero_points_present():
    # A signed/symmetric E2M1 code has no zero-point concept -- fc1_zero_points
    # present on an otherwise-valid nvfp4 chain is declined outright.
    E, hidden, inter, block_size = 2, 16, 16, 16
    rng = np.random.default_rng(431)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float64)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float64)
    fc1_gs = np.array([1.0, 1.0], dtype=np.float32)
    fc2_gs = np.array([1.0, 1.0], dtype=np.float32)
    fc1_q, fc1_s, _ = _qmoe_nvfp4_quantize_blockwise(fc1_w, fc1_gs, block_size)
    fc2_q, fc2_s, _ = _qmoe_nvfp4_quantize_blockwise(fc2_w, fc2_gs, block_size)
    fc1_zp = _pack_vals(np.zeros((E, inter), dtype=np.uint8), 4)
    model = _qmoe_nvfp4_model(
        fc1_q,
        fc1_s,
        fc2_q,
        fc2_s,
        fc1_gs,
        fc2_gs,
        block_size=block_size,
        fc1_zp=fc1_zp,
    )
    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _qmoe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q)


def test_qmoe_expert_channel_pruning_cpp_declines_nvfp4_missing_global_scale():
    # fc1_global_scale/fc2_global_scale are required PRESENT for nvfp4 (the
    # schema's own "must be provided" language) -- absent, the whole chain
    # is declined.
    E, hidden, inter, block_size = 2, 16, 16, 16
    rng = np.random.default_rng(432)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float64)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float64)
    fc1_gs = np.array([1.0, 1.0], dtype=np.float32)
    fc2_gs = np.array([1.0, 1.0], dtype=np.float32)
    fc1_q, fc1_s, _ = _qmoe_nvfp4_quantize_blockwise(fc1_w, fc1_gs, block_size)
    fc2_q, fc2_s, _ = _qmoe_nvfp4_quantize_blockwise(fc2_w, fc2_gs, block_size)
    model = _qmoe_nvfp4_model(
        fc1_q,
        fc1_s,
        fc2_q,
        fc2_s,
        None,
        None,
        block_size=block_size,
    )
    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    inits = _qmoe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1Q"], fc1_q)


# --- subgraph recursion --------------------------------------------------


def test_qmoe_expert_channel_pruning_cpp_prunes_node_inside_if_branch():
    # The exact hand-built-presliced-reference fixture, nested inside an
    # `If` -- verifies IterSubgraphs reaches a QMoE node nested at depth 1,
    # checked via byte-level initializer equality (a full InferenceSession
    # round trip is disproportionate here -- the top-level test already
    # covers that byte-packing math end-to-end).
    E, hidden, inter, bits, tokens = 3, 8, 12, 4, 6
    rng = np.random.default_rng(211)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)

    fc1_q, fc1_s, fc1_dq = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s, fc2_dq = _qmoe_quantize(fc2_w, bits)
    inner = _qmoe_model(
        fc1_q, fc1_s, fc2_q, fc2_s, bits, fc1_bias=fc1_b, k=2, tokens=tokens
    )
    wrapped = _if_wrap(inner)

    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(wrapped, sparsity=0.5)
    onnx.checker.check_model(pruned)

    sq = (
        np.sum(fc1_dq**2, axis=(0, 2))
        + np.sum(fc2_dq**2, axis=(0, 1))
        + np.sum(fc1_b**2, axis=0)
    )
    keep = np.sort(np.argsort(-np.sqrt(sq))[:6])
    expected_fc2_q = _pack_vals(_unpack_vals(fc2_q, bits, inter)[:, :, keep], bits)

    then_g, else_g = _if_branches(pruned)
    for branch_g in (then_g, else_g):
        inits = {t.name: onnx.numpy_helper.to_array(t) for t in branch_g.initializer}
        np.testing.assert_array_equal(inits["FC1Q"], fc1_q[:, keep, :])
        np.testing.assert_array_equal(inits["FC1S"], fc1_s[:, keep])
        np.testing.assert_array_equal(inits["FC1B"], fc1_b[:, keep])
        np.testing.assert_array_equal(inits["FC2Q"], expected_fc2_q)


# --- FP16/BFloat16 weight support -------------------------------------------
#
# MatchQMoEProducer (structured_pruning_entry.cpp) used to hard-require
# ``onnx.TensorProto.FLOAT`` for `fc1_scales`/`fc2_scales` (and, via
# QMoEOptionalFloatInput, `fc1_experts_bias`/`fc2_experts_bias`), silently
# declining any QMoE node whose scale/bias tensors were stored as FLOAT16 or
# BFLOAT16 -- narrower than ``onnxsim.apply_qmoe_expert_channel_pruning``'s
# own ``_is_supported_float_dtype``. `fc1_experts_weights`/
# `fc2_experts_weights` themselves are unaffected either way -- always
# packed UINT8 regardless of the *activation*/scale dtype. Now widened via
# IsSupportedFloatDtype/ReadTensorAsF64/WriteF64TensorAs (see
# structured_pruning_entry.cpp's own "QMoE (com.microsoft, quantized-weight
# Mixture-of-Experts) expert-channel structured pruning" section top
# comment).
#
# FLOAT16: confirmed separately that onnxruntime's CPU QMoE kernel executes
# genuine FLOAT16 activations/scales/bias, so this runs a real session,
# mirroring test_qmoe_expert_channel_pruning_cpp_matches_hand_built_presliced_
# reference's own "re-quantize the sliced float weights from scratch,
# independently" cross-check. BFLOAT16 has no onnxruntime CPU execution
# support in this environment at all (see
# tests/test_moe_pruning_cpp.py's own identical note) -- checked at the
# array level (dtype preservation, exact per-element bfloat16 decode)
# instead.


def test_qmoe_expert_channel_pruning_cpp_fp16_matches_hand_built_presliced_reference():
    E, hidden, inter, bits, tokens = 4, 8, 12, 4, 6
    rng = np.random.default_rng(2001)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)

    fc1_q, fc1_s32, _ = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s32, _ = _qmoe_quantize(fc2_w, bits)
    # Scale/bias are stored as FLOAT16 in the model -- the importance ranking
    # below must use the SAME (rounded) FLOAT16 values the C++ port will
    # actually read back (via ReadTensorAsF64), not the internal FLOAT32
    # quantizer scale, or a boundary case could rank differently.
    fc1_s = fc1_s32.astype(np.float16)
    fc2_s = fc2_s32.astype(np.float16)
    fc1_b16 = fc1_b.astype(np.float16)

    model = _qmoe_model(
        fc1_q,
        fc1_s,
        fc2_q,
        fc2_s,
        bits,
        fc1_bias=fc1_b16,
        k=2,
        tokens=tokens,
        float_dtype=np.float16,
    )
    onnx.checker.check_model(model)
    inits_before = _qmoe_inits(model)
    assert inits_before["FC1S"].dtype == np.float16
    assert inits_before["FC1B"].dtype == np.float16

    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = _qmoe_inits(pruned)
    assert inits["FC1S"].dtype == np.float16
    assert inits["FC1B"].dtype == np.float16
    assert inits["FC2S"].dtype == np.float16  # untouched (indexes hidden_size)

    default_zp = 1 << (bits - 1)
    fc1_dq = (
        _unpack_vals(fc1_q, bits, hidden).astype(np.float64) - default_zp
    ) * fc1_s.astype(np.float64)[:, :, None]
    fc2_dq = (
        _unpack_vals(fc2_q, bits, inter).astype(np.float64) - default_zp
    ) * fc2_s.astype(np.float64)[:, :, None]
    sq = (
        np.sum(fc1_dq**2, axis=(0, 2))
        + np.sum(fc2_dq**2, axis=(0, 1))
        + np.sum(fc1_b16.astype(np.float64) ** 2, axis=0)
    )
    keep = np.sort(np.argsort(-np.sqrt(sq))[:6])  # 12 - round(12*0.5) = 6

    np.testing.assert_array_equal(inits["FC1Q"], fc1_q[:, keep, :])
    np.testing.assert_array_equal(
        inits["FC1S"].view(np.uint16), fc1_s[:, keep].view(np.uint16)
    )
    np.testing.assert_array_equal(
        inits["FC1B"].view(np.uint16), fc1_b16[:, keep].view(np.uint16)
    )
    expected_fc2_q = _pack_vals(_unpack_vals(fc2_q, bits, inter)[:, :, keep], bits)
    np.testing.assert_array_equal(inits["FC2Q"], expected_fc2_q)

    # Independent cross-check: re-quantize the *sliced* float32 weights from
    # scratch (never touching the C++ port's own pruned bytes) and confirm a
    # real onnxruntime CPU FLOAT16 session agrees, mirroring
    # test_qmoe_expert_channel_pruning_cpp_matches_hand_built_presliced_
    # reference's own FLOAT32 cross-check.
    fc1_q_ref, fc1_s_ref32, _ = _qmoe_quantize(fc1_w[:, keep, :], bits)
    reference = _qmoe_model(
        fc1_q_ref,
        fc1_s_ref32.astype(np.float16),
        expected_fc2_q,
        fc2_s,
        bits,
        fc1_bias=fc1_b16[:, keep],
        k=2,
        tokens=tokens,
        float_dtype=np.float16,
    )
    onnx.checker.check_model(reference)

    feed_rng = np.random.default_rng(2003)
    feeds = {
        "X": (feed_rng.standard_normal((tokens, hidden)) * 0.2).astype(np.float16),
        "R": feed_rng.standard_normal((tokens, E)).astype(np.float16),
    }
    (out_pruned,) = _run(pruned, feeds)
    (out_ref,) = _run(reference, feeds)
    np.testing.assert_allclose(
        out_pruned.astype(np.float64), out_ref.astype(np.float64), rtol=5e-2, atol=5e-2
    )


def test_qmoe_expert_channel_pruning_cpp_bfloat16_preserves_dtype_and_matches_array_oracle():
    E, hidden, inter, bits, tokens = 4, 8, 12, 4, 6
    rng = np.random.default_rng(2005)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)

    fc1_q, fc1_s32, _ = _qmoe_quantize(fc1_w, bits)
    fc2_q, fc2_s32, _ = _qmoe_quantize(fc2_w, bits)
    fc1_s = fc1_s32.astype(ml_dtypes.bfloat16)
    fc2_s = fc2_s32.astype(ml_dtypes.bfloat16)
    fc1_b16 = fc1_b.astype(ml_dtypes.bfloat16)

    model = _qmoe_model(
        fc1_q,
        fc1_s,
        fc2_q,
        fc2_s,
        bits,
        fc1_bias=fc1_b16,
        k=2,
        tokens=tokens,
        float_dtype=ml_dtypes.bfloat16,
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_qmoe_expert_channel_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = _qmoe_inits(pruned)
    assert inits["FC1S"].dtype == ml_dtypes.bfloat16
    assert inits["FC1B"].dtype == ml_dtypes.bfloat16
    assert inits["FC2S"].dtype == ml_dtypes.bfloat16

    default_zp = 1 << (bits - 1)
    fc1_dq = (
        _unpack_vals(fc1_q, bits, hidden).astype(np.float64) - default_zp
    ) * fc1_s.astype(np.float64)[:, :, None]
    fc2_dq = (
        _unpack_vals(fc2_q, bits, inter).astype(np.float64) - default_zp
    ) * fc2_s.astype(np.float64)[:, :, None]
    sq = (
        np.sum(fc1_dq**2, axis=(0, 2))
        + np.sum(fc2_dq**2, axis=(0, 1))
        + np.sum(fc1_b16.astype(np.float64) ** 2, axis=0)
    )
    keep = np.sort(np.argsort(-np.sqrt(sq))[:6])

    np.testing.assert_array_equal(inits["FC1Q"], fc1_q[:, keep, :])
    np.testing.assert_array_equal(
        inits["FC1S"].view(np.uint16), fc1_s[:, keep].view(np.uint16)
    )
    np.testing.assert_array_equal(
        inits["FC1B"].view(np.uint16), fc1_b16[:, keep].view(np.uint16)
    )
    expected_fc2_q = _pack_vals(_unpack_vals(fc2_q, bits, inter)[:, :, keep], bits)
    np.testing.assert_array_equal(inits["FC2Q"], expected_fc2_q)
