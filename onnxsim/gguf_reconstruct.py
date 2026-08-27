"""Reconstructs an ONNX graph *and* its weights directly from a GGUF LLM
checkpoint, for a small set of recognized decoder-only transformer
architectures (currently the Llama family: Llama/Llama2/Llama3, Mistral, and
Qwen2, which all share the same block shape -- RMSNorm, rotary position
embeddings, grouped-query attention, SwiGLU FFN -- differing only in things
:func:`read_gguf_metadata` already reports: head counts, RoPE base, whether
q/k/v projections carry a bias, and whether the LM head is tied to the token
embedding).

This is deliberately the "known architecture template" approach, not a
generic GGUF/ggml graph-structure reconstructor: vLLM and SGLang's own GGUF
support works the same way (match ``general.architecture`` against a
maintained, hand-written model implementation; fail clearly -- "architecture
X is not supported yet" -- rather than guess when it isn't recognized), and
their own issue trackers show that's a deliberate, load-bearing choice, not
a shortcut: coupling to a generic-but-unstable source (llama.cpp's internal,
non-public compute-graph construction, revised on nearly every commit) is a
worse tradeoff than a curated, explicit template per architecture family.

:func:`read_gguf_metadata` supplies everything this needs to know -- which
architecture, its hyperparameters, and its tensors' names/shapes -- without
reading any tensor byte data; :func:`import_gguf_weights` (reused here
unmodified) supplies the actual values, including its existing K-quant
(Q4_K/Q5_K/Q6_K/Q8_0) decode.

Scope note on shapes: the returned graph's ``batch_size``/``seq_len`` are
concrete, caller-chosen static dimensions, not dynamic axes. Real llama.cpp
inference builds a *different* concrete compute graph per call anyway (see
this feature's design discussion for why a cache-free, single-shape forward
graph is the right starting point at all) -- generalizing this to dynamic
axes, and to KV-cache-aware incremental decoding, is future work, not
something this first slice claims to solve.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.onnx_simplifier import import_gguf_weights, read_gguf_metadata

# Opset 17: modern enough for everything this builder needs (Trilu has been
# available since opset 14), while ReduceMean's reduction axes are still an
# *attribute* rather than an input (that migration happened at opset 18) --
# picking 17 avoids threading yet another small int64 constant through every
# RMSNorm call for no benefit here.
_OPSET = 17

# Mirrors onnxsim/gguf_dtype.h's GgmlType enum and ToOnnx/IsKQuant mapping --
# duplicated here rather than exposed through a new C++ binding, the same
# choice tests/test_import_gguf_weights.py already made for its own small,
# stable GGML_TYPE_* constants. See gguf_dtype.h's file comment: GGML never
# reassigns an existing type ID, so this mapping does not drift.
_GGML_RAW_TO_ONNX = {
    0: onnx.TensorProto.FLOAT,  # F32
    1: onnx.TensorProto.FLOAT16,  # F16
    24: onnx.TensorProto.INT8,  # I8
    25: onnx.TensorProto.INT16,  # I16
    26: onnx.TensorProto.INT32,  # I32
    27: onnx.TensorProto.INT64,  # I64
    28: onnx.TensorProto.DOUBLE,  # F64
    30: onnx.TensorProto.BFLOAT16,  # BF16
}
# Q8_0, Q4_K, Q5_K, Q6_K -- import_gguf_weights forces these to FLOAT
# regardless of what the initializer previously declared (see
# tensor_pool_gguf_bridge.h's HydrateTensorProtoFromGGUF), so that is what
# must be declared here too.
_GGML_KQUANT_TYPES = {8, 12, 13, 14}

_ONNX_DTYPE_ITEMSIZE = {
    onnx.TensorProto.FLOAT: 4,
    onnx.TensorProto.FLOAT16: 2,
    onnx.TensorProto.BFLOAT16: 2,
    onnx.TensorProto.DOUBLE: 8,
    onnx.TensorProto.INT8: 1,
    onnx.TensorProto.INT16: 2,
    onnx.TensorProto.INT32: 4,
    onnx.TensorProto.INT64: 8,
}

# general.architecture values this builder recognizes -- all share the same
# block shape (RMSNorm/RoPE/GQA/SwiGLU); see the module docstring.
_SUPPORTED_ARCHITECTURES = ("llama", "qwen2", "mistral")


class UnsupportedArchitectureError(NotImplementedError):
    """Raised for a GGUF checkpoint whose ``general.architecture`` (or a
    quantization format among the tensors this graph needs) this builder
    does not have a template for -- mirrors vLLM/SGLang's own "architecture
    X is not supported yet" failure mode: fail clearly rather than guess."""


class _Builder:
    """Accumulates nodes/initializers for one ONNX graph. Not reusable
    across graphs -- one instance per :func:`reconstruct_gguf_graph` call."""

    def __init__(self):
        self.nodes: List[onnx.NodeProto] = []
        self.initializers: List[onnx.TensorProto] = []
        self._counter = 0
        self._const_cache: Dict[Tuple, str] = {}

    def _name(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}.{self._counter}"

    def op(self, op_type: str, inputs: List[str], prefix: str, **attrs) -> str:
        out = self._name(prefix)
        self.nodes.append(onnx.helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def placeholder_weight(self, name: str, shape: List[int], onnx_dtype: int) -> None:
        """A zero-filled initializer with `name`'s exact GGUF-reported shape
        and the ONNX dtype import_gguf_weights will actually write into it
        (see _GGML_RAW_TO_ONNX/_GGML_KQUANT_TYPES) -- its *values* come from
        import_gguf_weights right after the graph this builds is assembled,
        but its declared dims/data_type must already be correct: hydration
        overwrites raw_data only (see tensor_pool_gguf_bridge.h's
        HydrateTensorProto/HydrateTensorProtoFromGGUF), never dims, and
        never data_type on the plain-raw-dtype path."""
        nbytes = _ONNX_DTYPE_ITEMSIZE[onnx_dtype]
        for d in shape:
            nbytes *= d
        t = onnx.helper.make_tensor(
            name, onnx_dtype, shape, vals=b"\x00" * nbytes, raw=True
        )
        self.initializers.append(t)

    def const(self, array: np.ndarray, prefix: str = "const") -> str:
        """A constant initializer from a numpy array this builder computed
        itself (RoPE's inv_freq, a causal mask, a reshape's target shape,
        ...) -- distinct from placeholder_weight, whose values come from the
        GGUF file, not from Python. Small integer-shape constants (Reshape
        targets, Slice bounds) recur often enough across layers to dedupe by
        value."""
        key = (array.shape, array.dtype.str, array.tobytes())
        cached = self._const_cache.get(key)
        if cached is not None:
            return cached
        name = self._name(prefix)
        self.initializers.append(onnx.numpy_helper.from_array(array, name=name))
        self._const_cache[key] = name
        return name

    def shape_const(self, dims: List[int]) -> str:
        return self.const(np.array(dims, dtype=np.int64), prefix="shape")


def _unsqueeze(b: _Builder, x: str, axes: List[int], prefix: str) -> str:
    # Unsqueeze's `axes` has been a second *input*, not an attribute, since
    # opset 13 -- unlike Concat's `axis` (always an attribute) or
    # ReduceMean's `axes` (still an attribute until opset 18, which _OPSET
    # predates).
    axes_c = b.const(np.array(axes, dtype=np.int64), prefix="unsqueeze_axes")
    return b.op("Unsqueeze", [x, axes_c], prefix)


def _linear(
    b: _Builder,
    x: str,
    weight_name: str,
    bias_name: Optional[str],
    prefix: str,
) -> str:
    """``x @ weight.T (+ bias)`` -- nn.Linear semantics. `weight_name` is a
    GGUF tensor already declared (via placeholder_weight) with its ORIGINAL
    [out_features, in_features] shape (GGUF/ggml round-trips a PyTorch
    nn.Linear.weight's shape unchanged -- see read_gguf_metadata's own
    dimension-order note), so this transposes at graph-build time via an
    explicit Transpose node rather than pre-transposing the (not-yet-known)
    weight values. A later `onnxsim.simplify()` constant-folds that
    Transpose away for free once the weight is actually hydrated."""
    wt = b.op("Transpose", [weight_name], f"{prefix}.wt", perm=[1, 0])
    out = b.op("MatMul", [x, wt], f"{prefix}.matmul")
    if bias_name is not None:
        out = b.op("Add", [out, bias_name], f"{prefix}.bias")
    return out


def _rmsnorm(b: _Builder, x: str, weight_name: str, eps: float, prefix: str) -> str:
    eps_c = b.const(np.array(eps, dtype=np.float32), prefix="eps")
    x2 = b.op("Mul", [x, x], f"{prefix}.sq")
    mean = b.op("ReduceMean", [x2], f"{prefix}.mean", axes=[-1], keepdims=1)
    var_eps = b.op("Add", [mean, eps_c], f"{prefix}.var_eps")
    rms = b.op("Sqrt", [var_eps], f"{prefix}.rms")
    normed = b.op("Div", [x, rms], f"{prefix}.normed")
    return b.op("Mul", [normed, weight_name], f"{prefix}.scaled")


def _slice_last_dim(b: _Builder, x: str, start: int, end: int, prefix: str) -> str:
    starts = b.const(np.array([start], dtype=np.int64), prefix="slice_start")
    ends = b.const(np.array([end], dtype=np.int64), prefix="slice_end")
    axes = b.const(np.array([-1], dtype=np.int64), prefix="slice_axis")
    return b.op("Slice", [x, starts, ends, axes], prefix)


def _rotate_half(b: _Builder, x: str, head_dim: int, prefix: str) -> str:
    half = head_dim // 2
    x1 = _slice_last_dim(b, x, 0, half, f"{prefix}.x1")
    x2 = _slice_last_dim(b, x, half, head_dim, f"{prefix}.x2")
    neg_x2 = b.op("Neg", [x2], f"{prefix}.negx2")
    return b.op("Concat", [neg_x2, x1], prefix, axis=-1)


def _apply_rope(
    b: _Builder, x: str, cos: str, sin: str, head_dim: int, prefix: str
) -> str:
    rotated = _rotate_half(b, x, head_dim, f"{prefix}.rot")
    a = b.op("Mul", [x, cos], f"{prefix}.a")
    c = b.op("Mul", [rotated, sin], f"{prefix}.c")
    return b.op("Add", [a, c], prefix)


def _reconstruct_llama_family(
    meta: dict, batch_size: int, seq_len: int
) -> onnx.GraphProto:
    kv = meta["kv"]
    tensors = {t["name"]: t for t in meta["tensors"]}
    arch = kv["general.architecture"]

    def key(suffix: str):
        return f"{arch}.{suffix}"

    n_embd = int(kv[key("embedding_length")])
    n_layer = int(kv[key("block_count")])
    n_ff = int(kv[key("feed_forward_length")])
    n_head = int(kv[key("attention.head_count")])
    n_head_kv = int(kv.get(key("attention.head_count_kv"), n_head))
    eps = float(
        kv.get(
            key("attention.layer_norm_rms_epsilon"),
            kv.get(key("attention.layer_norm_epsilon"), 1e-5),
        )
    )
    freq_base = float(kv.get(key("rope.freq_base"), 10000.0))

    if n_embd % n_head != 0:
        raise UnsupportedArchitectureError(
            f"{key('embedding_length')}={n_embd} is not divisible by "
            f"{key('attention.head_count')}={n_head}"
        )
    head_dim = n_embd // n_head
    if n_head % n_head_kv != 0:
        raise UnsupportedArchitectureError(
            f"{key('attention.head_count')}={n_head} is not a multiple of "
            f"{key('attention.head_count_kv')}={n_head_kv} (grouped-query "
            "attention requires an integer head-repeat factor)"
        )
    n_rep = n_head // n_head_kv

    rope_dims = kv.get(key("rope.dimension_count"))
    if rope_dims is not None and int(rope_dims) != head_dim:
        raise UnsupportedArchitectureError(
            f"{key('rope.dimension_count')}={rope_dims} != head_dim={head_dim} "
            "(partial rotary embeddings are not implemented)"
        )

    def require(name: str) -> dict:
        info = tensors.get(name)
        if info is None:
            raise UnsupportedArchitectureError(
                f"checkpoint is missing required tensor '{name}' for "
                f"architecture '{arch}'"
            )
        return info

    def optional(name: str) -> Optional[dict]:
        return tensors.get(name)

    b = _Builder()

    def declare(name: str, expected_shape: List[int]) -> str:
        info = require(name)
        if info["shape"] != expected_shape:
            raise UnsupportedArchitectureError(
                f"tensor '{name}' has shape {info['shape']}, expected "
                f"{expected_shape} for architecture '{arch}' with "
                f"embedding_length={n_embd}, feed_forward_length={n_ff}, "
                f"head_count={n_head}, head_count_kv={n_head_kv}"
            )
        ggml_type = info["ggml_type"]
        if ggml_type in _GGML_KQUANT_TYPES:
            onnx_dtype = onnx.TensorProto.FLOAT
        elif ggml_type in _GGML_RAW_TO_ONNX:
            onnx_dtype = _GGML_RAW_TO_ONNX[ggml_type]
        else:
            raise UnsupportedArchitectureError(
                f"tensor '{name}' uses ggml_type {ggml_type}, which "
                "import_gguf_weights cannot decode (only F32/F16/BF16/F64/"
                "I8/I16/I32/I64 and the Q4_K/Q5_K/Q6_K/Q8_0 K-quant family "
                "are supported)"
            )
        b.placeholder_weight(name, expected_shape, onnx_dtype)
        return name

    def declare_optional(name: str, expected_shape: List[int]) -> Optional[str]:
        return declare(name, expected_shape) if optional(name) is not None else None

    token_embd_info = require("token_embd.weight")
    if len(token_embd_info["shape"]) != 2 or token_embd_info["shape"][1] != n_embd:
        raise UnsupportedArchitectureError(
            f"token_embd.weight shape {token_embd_info['shape']} does not "
            f"match {key('embedding_length')}={n_embd}"
        )
    vocab_size = token_embd_info["shape"][0]
    token_embd = declare("token_embd.weight", [vocab_size, n_embd])

    input_ids = "input_ids"
    position_ids = "position_ids"
    graph_inputs = [
        onnx.helper.make_tensor_value_info(
            input_ids, onnx.TensorProto.INT64, [batch_size, seq_len]
        ),
        onnx.helper.make_tensor_value_info(
            position_ids, onnx.TensorProto.INT64, [batch_size, seq_len]
        ),
    ]

    x = b.op("Gather", [token_embd, input_ids], "embed", axis=0)

    # RoPE cos/sin: identical across every layer (same position_ids, same
    # freq_base/head_dim), so computed once here rather than per layer.
    inv_freq = 1.0 / (
        freq_base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    )
    inv_freq_c = b.const(
        inv_freq.reshape(1, 1, -1).astype(np.float32), prefix="inv_freq"
    )
    pos_f = b.op("Cast", [position_ids], "pos_f", to=onnx.TensorProto.FLOAT)
    pos_unsq = _unsqueeze(b, pos_f, [-1], "pos_unsq")
    freqs = b.op("Mul", [pos_unsq, inv_freq_c], "freqs")
    emb = b.op("Concat", [freqs, freqs], "rope_emb", axis=-1)
    cos = b.op("Cos", [emb], "rope_cos")
    sin = b.op("Sin", [emb], "rope_sin")
    # [B, S, D] -> [B, 1, S, D], broadcasting over both the H-head and the
    # HKV-head cases uniformly (dim 1 has size 1 either way).
    cos_b = _unsqueeze(b, cos, [1], "rope_cos_b")
    sin_b = _unsqueeze(b, sin, [1], "rope_sin_b")

    causal_mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
    mask_c = b.const(causal_mask, prefix="causal_mask")
    inv_sqrt_d = b.const(
        np.array(1.0 / math.sqrt(head_dim), dtype=np.float32), prefix="inv_sqrt_d"
    )

    def reshape(t: str, dims: List[int], prefix: str) -> str:
        return b.op("Reshape", [t, b.shape_const(dims)], prefix)

    n_embd_gqa = n_head_kv * head_dim
    for i in range(n_layer):
        p = f"blk.{i}"
        resid = x
        h = _rmsnorm(
            b, x, declare(f"{p}.attn_norm.weight", [n_embd]), eps, f"{p}.attn_norm"
        )

        q = _linear(
            b,
            h,
            declare(f"{p}.attn_q.weight", [n_embd, n_embd]),
            declare_optional(f"{p}.attn_q.bias", [n_embd]),
            f"{p}.q_proj",
        )
        k = _linear(
            b,
            h,
            declare(f"{p}.attn_k.weight", [n_embd_gqa, n_embd]),
            declare_optional(f"{p}.attn_k.bias", [n_embd_gqa]),
            f"{p}.k_proj",
        )
        v = _linear(
            b,
            h,
            declare(f"{p}.attn_v.weight", [n_embd_gqa, n_embd]),
            declare_optional(f"{p}.attn_v.bias", [n_embd_gqa]),
            f"{p}.v_proj",
        )

        q = reshape(q, [batch_size, seq_len, n_head, head_dim], f"{p}.q_r")
        q = b.op("Transpose", [q], f"{p}.q_t", perm=[0, 2, 1, 3])
        k = reshape(k, [batch_size, seq_len, n_head_kv, head_dim], f"{p}.k_r")
        k = b.op("Transpose", [k], f"{p}.k_t", perm=[0, 2, 1, 3])
        v = reshape(v, [batch_size, seq_len, n_head_kv, head_dim], f"{p}.v_r")
        v = b.op("Transpose", [v], f"{p}.v_t", perm=[0, 2, 1, 3])

        q = _apply_rope(b, q, cos_b, sin_b, head_dim, f"{p}.q_rope")
        k = _apply_rope(b, k, cos_b, sin_b, head_dim, f"{p}.k_rope")

        # Grouped-query attention via broadcasting, not an explicit
        # repeat/tile of k/v: split q's H heads into [HKV, REP] and give k/v
        # a size-1 REP axis, so MatMul's own batch-dimension broadcasting
        # does the repeat implicitly. See the module docstring's design note
        # and this function's own comment on _linear for the same
        # "let a later onnxsim.simplify() fold what it can" philosophy.
        q5 = reshape(q, [batch_size, n_head_kv, n_rep, seq_len, head_dim], f"{p}.q5")
        k5 = _unsqueeze(b, k, [2], f"{p}.k5")
        v5 = _unsqueeze(b, v, [2], f"{p}.v5")

        k5t = b.op("Transpose", [k5], f"{p}.k5t", perm=[0, 1, 2, 4, 3])
        scores = b.op("MatMul", [q5, k5t], f"{p}.scores")
        scores = b.op("Mul", [scores, inv_sqrt_d], f"{p}.scores_scaled")
        scores = b.op("Add", [scores, mask_c], f"{p}.scores_masked")
        attn = b.op("Softmax", [scores], f"{p}.softmax", axis=-1)
        out5 = b.op("MatMul", [attn, v5], f"{p}.attn_out5")

        out = reshape(out5, [batch_size, n_head, seq_len, head_dim], f"{p}.out_r")
        out = b.op("Transpose", [out], f"{p}.out_t", perm=[0, 2, 1, 3])
        out = reshape(out, [batch_size, seq_len, n_embd], f"{p}.out_flat")
        out = _linear(
            b,
            out,
            declare(f"{p}.attn_output.weight", [n_embd, n_embd]),
            declare_optional(f"{p}.attn_output.bias", [n_embd]),
            f"{p}.o_proj",
        )
        x = b.op("Add", [resid, out], f"{p}.attn_resid")

        resid = x
        h = _rmsnorm(
            b, x, declare(f"{p}.ffn_norm.weight", [n_embd]), eps, f"{p}.ffn_norm"
        )
        gate = _linear(
            b,
            h,
            declare(f"{p}.ffn_gate.weight", [n_ff, n_embd]),
            declare_optional(f"{p}.ffn_gate.bias", [n_ff]),
            f"{p}.gate_proj",
        )
        up = _linear(
            b,
            h,
            declare(f"{p}.ffn_up.weight", [n_ff, n_embd]),
            declare_optional(f"{p}.ffn_up.bias", [n_ff]),
            f"{p}.up_proj",
        )
        silu = b.op("Sigmoid", [gate], f"{p}.silu_sig")
        silu = b.op("Mul", [gate, silu], f"{p}.silu")
        act = b.op("Mul", [silu, up], f"{p}.act")
        down = _linear(
            b,
            act,
            declare(f"{p}.ffn_down.weight", [n_embd, n_ff]),
            declare_optional(f"{p}.ffn_down.bias", [n_embd]),
            f"{p}.down_proj",
        )
        x = b.op("Add", [resid, down], f"{p}.ffn_resid")

    x = _rmsnorm(b, x, declare("output_norm.weight", [n_embd]), eps, "output_norm")

    if optional("output.weight") is not None:
        lm_head = declare("output.weight", [vocab_size, n_embd])
    else:
        # Tied embeddings: some Llama-family checkpoints (small models
        # especially) have no separate LM head tensor at all and reuse
        # token_embd.weight for both -- token_embd was already declared
        # above, so there is nothing new to hydrate here.
        lm_head = token_embd
    logits = _linear(b, x, lm_head, None, "lm_head")

    graph = onnx.helper.make_graph(
        b.nodes,
        f"gguf_{arch}",
        graph_inputs,
        [
            onnx.helper.make_tensor_value_info(
                logits, onnx.TensorProto.FLOAT, [batch_size, seq_len, vocab_size]
            )
        ],
        initializer=b.initializers,
    )
    return graph


def reconstruct_gguf_graph(
    gguf_path: str, batch_size: int = 1, seq_len: int = 8
) -> Tuple[onnx.ModelProto, List[str]]:
    """
    Build a runnable ONNX graph -- structure *and* weights -- directly from
    a GGUF checkpoint, for a recognized architecture (currently the Llama
    family: ``llama``, ``qwen2``, ``mistral`` -- see the module docstring).

    Unlike :func:`import_gguf_weights`, which only ever fills in an
    existing graph's initializer *values*, this constructs the graph
    itself from the checkpoint's own declared hyperparameters
    (:func:`read_gguf_metadata`), then calls ``import_gguf_weights``
    internally to hydrate it -- so it reuses that function's existing
    K-quant (Q4_K/Q5_K/Q6_K/Q8_0) decode unchanged.

    :param gguf_path: path to the GGUF checkpoint
    :param batch_size: static batch dimension baked into the returned
            graph's input/output shapes (see the module docstring's scope
            note: this is not a dynamic axis)
    :param seq_len: static sequence-length dimension, likewise baked in
    :returns: ``(model, skipped)`` -- the constructed, hydrated model
            (inputs ``input_ids``/``position_ids``, both
            ``int64[batch_size, seq_len]``; output ``logits``,
            ``float32[batch_size, seq_len, vocab_size]``), and the names of
            any GGUF tensors present in the file but left un-hydrated
            (always empty in practice: every tensor this graph references
            is validated against the supported dtype set before the graph
            is even built, and ``import_gguf_weights`` never touches a
            tensor that is not present in ``model``'s initializers).
    :raises UnsupportedArchitectureError: if ``general.architecture`` is not
            one this builder has a template for, a required tensor is
            missing, or a required tensor's quantization format has no
            decoder (see :func:`import_gguf_weights`'s own scope note).
    """
    meta = read_gguf_metadata(gguf_path)
    arch = meta["kv"].get("general.architecture")
    if arch not in _SUPPORTED_ARCHITECTURES:
        raise UnsupportedArchitectureError(
            f"general.architecture={arch!r} has no graph template here -- "
            f"supported: {', '.join(_SUPPORTED_ARCHITECTURES)}"
        )

    graph = _reconstruct_llama_family(meta, batch_size, seq_len)
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", _OPSET)]
    )
    model.ir_version = onnx.IR_VERSION

    model, skipped = import_gguf_weights(model, gguf_path)
    return model, skipped
