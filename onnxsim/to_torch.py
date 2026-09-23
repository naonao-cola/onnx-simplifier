"""Convert an ONNX decoder LLM into a ``torch.export``-able ``torch.nn.Module``.

The primary consumer is TensorRT-LLM's AutoDeploy (``scripts/nvidia/trtllm_autodeploy_onnx.py``),
which takes a PyTorch module with ``forward(input_ids, position_ids) -> (logits, ...)``,
exports it to a ``torch.fx`` graph, pattern-matches attention / RoPE / RMSNorm onto its
canonical ops and replaces attention with TensorRT-LLM's paged-KV-cache kernels. That
gives an ONNX model a path onto the TensorRT-LLM runtime even though TensorRT-LLM itself
has no ONNX importer (its legacy TensorRT backend and ONNX tooling were removed in 1.3).

How the conversion works:

* The returned :class:`OnnxModule` *interprets* the ONNX graph op by op inside
  ``forward``. ``torch.export`` traces through that interpreter, so the exported graph is a
  flat aten graph with no trace of the interpreter loop.
* Only the nodes the kept outputs depend on are evaluated. Everything else -- the
  attention-mask construction, ``present.*`` outputs, KV-cache bookkeeping -- simply never
  runs and never appears in the exported graph.
* Integer shape arithmetic (``Shape`` -> ``Gather``/``Slice``/``Concat``/``Mul``... ->
  ``Reshape``/``Expand``/``Range``) is carried as Python tuples/ints rather than tensors.
  Under ``torch.export`` those are symbolic ints, which is what keeps batch and sequence
  length dynamic; turning them into int64 tensors would make every downstream ``reshape``
  data-dependent.
* Decoder self-attention is recognized on the ONNX graph --
  ``MatMul(q, kT) -> [Mul/Div scale] -> [Add/Where mask] -> [Cast] -> Softmax -> [Cast] ->
  MatMul(., v)`` or the opset-23 ``Attention`` op -- and, with ``attention="sdpa"`` (the
  default), emitted as ``F.scaled_dot_product_attention(q, k, v, is_causal=True)``: the
  mask input is dropped and causal masking assumed, since the cache-inserting runtime
  supplies its own. ``attention="exact"`` interprets the matched nodes as written instead
  (useful to validate a conversion against onnxruntime).
* With ``strip_kv_cache=True`` (the default), graph inputs named like a KV cache
  (``past_key_values.*``, ``past_key*``/``past_value*``) are removed: a ``Concat`` of a past
  input with the freshly computed key/value is replaced by the fresh value, i.e. the model
  becomes the cache-less prefill graph AutoDeploy expects. A live node that still needs a
  past input (e.g. positions derived from the past length) is reported as an error; pass
  ``position_ids`` explicitly instead.

Only what decoder-LLM exports use is implemented; unsupported ops raise
``NotImplementedError`` naming the op, rather than guessing.
"""

from __future__ import annotations

import contextvars
import operator
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

__all__ = ["onnx_to_torch", "onnx_state_dict"]

# Device of the current forward call's inputs, for tensors created inside forward
# (ConstantOfShape, Range). A ContextVar rather than a module attribute: torch.export
# rejects attributes set during forward.
_DEVICE = contextvars.ContextVar("onnxsim_to_torch_device", default=None)
# Per-forward-call memo of RoPE cos/sin tensors: q and k each have their own ONNX
# RotaryEmbedding node, but must share one cos/sin (unsqueezed) value for AutoDeploy's
# match_rope_pattern, which matches q's and k's rotation as a single pattern.
_ROPE_MEMO: contextvars.ContextVar = contextvars.ContextVar("onnxsim_to_torch_rope")
# forward's position_ids, for GroupQueryAttention do_rotary=1 graphs that have no
# position input of their own (ONNX Runtime GenAI builder, e.g. Llama-3.2).
_POSITIONS: contextvars.ContextVar = contextvars.ContextVar("onnxsim_to_torch_pos")

_KV_INPUT_RE = re.compile(r"^(past_key_values|past_key|past_value|past)[._]")

# Int initializers/constants up to this many elements are treated as static shape
# values (Python ints) instead of tensors.
_SHAPE_CONST_MAX = 64


class _Sh(tuple):
    """A 1-D int64 ONNX value carried as a Python tuple of (Sym)ints."""


def _torch():
    import torch

    return torch


def _dtype(onnx_type: int):
    torch = _torch()
    return {
        TensorProto.FLOAT: torch.float32,
        TensorProto.FLOAT16: torch.float16,
        TensorProto.BFLOAT16: torch.bfloat16,
        TensorProto.DOUBLE: torch.float64,
        TensorProto.INT64: torch.int64,
        TensorProto.INT32: torch.int32,
        TensorProto.INT16: torch.int16,
        TensorProto.INT8: torch.int8,
        TensorProto.UINT8: torch.uint8,
        TensorProto.BOOL: torch.bool,
    }[onnx_type]


def _attrs(node) -> Dict[str, Any]:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _is_int(v) -> bool:
    """A Python int or a ``torch.SymInt`` (which is *not* an ``int`` subclass)."""
    if isinstance(v, bool):
        return False
    return isinstance(v, int) or type(v).__name__ == "SymInt"


def _is_shape(v) -> bool:
    return isinstance(v, _Sh) or _is_int(v)


def _tensor(v, like=None):
    """Materialize a shape value (or Python scalar) as a tensor."""
    torch = _torch()
    if isinstance(v, torch.Tensor):
        return v
    device = like.device if like is not None else None
    if isinstance(v, _Sh):
        if all(isinstance(e, int) for e in v):
            return torch.tensor(list(v), dtype=torch.int64, device=device)
        return torch.stack(
            [torch.as_tensor(e, dtype=torch.int64, device=device) for e in v]
        )
    if isinstance(v, bool):
        return torch.tensor(v, device=device)
    if _is_int(v):
        return torch.as_tensor(v, dtype=torch.int64, device=device)
    if isinstance(v, float):
        return torch.tensor(
            v, dtype=like.dtype if like is not None else torch.float32, device=device
        )
    return torch.as_tensor(v, device=device)


def _ints(v) -> List:
    """A shape value (or static int tensor) as a list of (Sym)ints."""
    torch = _torch()
    if isinstance(v, _Sh):
        return list(v)
    if _is_int(v):
        return [v]
    if isinstance(v, torch.Tensor):
        return [int(e) for e in v.reshape(-1).tolist()]
    raise TypeError(f"expected a shape value, got {type(v)}")


def _const_value(arr: np.ndarray):
    """Static value of an initializer / Constant, or None if it must stay a tensor.

    Small int tensors become shape values; 0-d floats (attention scale, epsilon, the 2 in
    ``Pow(x, 2)``) become Python floats -- torch ops keep the tensor operand's dtype, and
    the attention matcher can read them without touching (possibly ``meta``) weights.
    """
    if arr.dtype.kind in "iu" and arr.size <= _SHAPE_CONST_MAX and arr.ndim <= 1:
        return int(arr) if arr.ndim == 0 else _Sh(int(e) for e in arr)
    if arr.dtype.kind == "f" and arr.ndim == 0:
        return float(arr)
    return None


def _norm_axis(axis: int, rank: int) -> int:
    return axis + rank if axis < 0 else axis


def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", name)


def _node_const(node) -> Optional[np.ndarray]:
    a = _attrs(node)
    if "value" in a:
        return numpy_helper.to_array(a["value"])
    for k, dt in (("value_int", np.int64), ("value_float", np.float32)):
        if k in a:
            return np.array(a[k], dtype=dt)
    for k, dt in (("value_ints", np.int64), ("value_floats", np.float32)):
        if k in a:
            return np.array(list(a[k]), dtype=dt)
    return None


def _dequant_matmulnbits(node, arrays) -> np.ndarray:
    """Dense ``[N, K]`` (``F.linear``-layout) weight of a ``com.microsoft::MatMulNBits`` node.

    ``B`` is ``[N, k_blocks, blob]`` uint8 with ``bits``-bit values packed low bits
    first; ``scales`` has one entry per (row, block); ``zero_points`` (optional) is
    either packed like ``B`` or in the scales' type, defaulting to ``2**(bits-1)``.
    """
    a = _attrs(node)
    k, n, bits, block = a["K"], a["N"], a.get("bits", 4), a["block_size"]
    if len(node.input) > 4 and node.input[4]:
        raise NotImplementedError("MatMulNBits with g_idx (act-order)")
    if 8 % bits:
        raise NotImplementedError(f"MatMulNBits bits={bits}")
    b = arrays[node.input[1]]
    scales = arrays[node.input[2]]
    k_blocks = -(-k // block)
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    b = b.reshape(n, -1)
    q = np.stack([(b >> (bits * i)) & mask for i in range(per_byte)], -1).reshape(n, -1)
    q = q[:, : k_blocks * block].astype(np.float32).reshape(n, k_blocks, block)
    sc = scales.astype(np.float32).reshape(n, k_blocks, 1)
    zp_name = node.input[3] if len(node.input) > 3 else ""
    if zp_name:
        zp = arrays[zp_name]
        if zp.dtype == np.uint8:
            zp = zp.reshape(n, -1)
            zp = np.stack([(zp >> (bits * i)) & mask for i in range(per_byte)], -1)
            zp = zp.reshape(n, -1)[:, :k_blocks].astype(np.float32)
        else:
            zp = zp.astype(np.float32).reshape(n, k_blocks)
        zp = zp[..., None]
    else:
        zp = np.float32(1 << (bits - 1))
    w = ((q - zp) * sc).reshape(n, -1)[:, :k]
    return np.ascontiguousarray(w.astype(scales.dtype))  # [N, K]: F.linear layout


class _AttnMatch:
    """One recognized ``softmax(scale * q @ kT [+ mask]) @ v`` block."""

    def __init__(self, out: str, q: str, kt: str, v: str, scale: float, nodes: List):
        self.out, self.q, self.kt, self.v, self.scale, self.nodes = (
            out,
            q,
            kt,
            v,
            scale,
            nodes,
        )


def _build_module_class():
    torch = _torch()
    import torch.nn.functional as F

    class OnnxModule(torch.nn.Module):  # noqa: D401 -- see module docstring
        """ONNX graph interpreter as an ``nn.Module``; see :func:`onnx_to_torch`."""

        def __init__(
            self,
            model: onnx.ModelProto,
            *,
            inputs: Sequence[str] = ("input_ids", "position_ids"),
            outputs: Optional[Sequence[str]] = None,
            strip_kv_cache: bool = True,
            attention: str = "sdpa",
            device: Optional[str] = None,
        ):
            super().__init__()
            if attention not in ("sdpa", "exact"):
                raise ValueError("attention must be 'sdpa' or 'exact'")
            g = model.graph
            self._opset = next(
                (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
                13,
            )
            self._attention = attention
            graph_inputs = [i.name for i in g.input]
            init_names = {t.name for t in g.initializer}
            real_inputs = [n for n in graph_inputs if n not in init_names]
            # GQA with in-op rotary derives positions from seqlens_k in ONNX Runtime;
            # a cache-inserting runtime supplies position_ids instead, so accept it as a
            # forward input even though the graph has none.
            self._synthetic_pos = (
                "position_ids" in inputs
                and "position_ids" not in real_inputs
                and attention == "sdpa"
                and any(
                    n.op_type == "GroupQueryAttention" and _attrs(n).get("do_rotary", 0)
                    for n in g.node
                )
            )
            missing = [
                n
                for n in inputs
                if n not in real_inputs
                and not (n == "position_ids" and self._synthetic_pos)
            ]
            if missing:
                raise ValueError(
                    f"inputs {missing} not among graph inputs {real_inputs}"
                )
            self._input_names = list(inputs)
            self._kv_inputs = (
                {n for n in real_inputs if _KV_INPUT_RE.match(n)}
                if strip_kv_cache
                else set()
            )
            self._output_names = list(outputs) if outputs else [g.output[0].name]

            # Static values: small int constants become shape values, the rest tensors.
            self._static: Dict[str, Any] = {}
            self._param_of: Dict[str, str] = {}
            # Weight-only-quantized MatMulNBits: dequantized once, here, to a dense
            # [K, N] parameter keyed "<B>::dequant"; the packed inputs are not kept.
            nbits = [
                n
                for n in g.node
                if n.domain == "com.microsoft" and n.op_type == "MatMulNBits"
            ]
            packed = {i for n in nbits for i in n.input[1:4] if i}
            other_uses = {i for n in g.node if n not in nbits for i in n.input}
            skip = packed - other_uses
            self._nbits_key: Dict[str, str] = {}
            if nbits:
                arrays = {
                    t.name: numpy_helper.to_array(t)
                    for t in g.initializer
                    if t.name in packed
                }
                for n in nbits:
                    key = n.input[1] + "::dequant"
                    if key not in self._param_of:
                        self._register(key, _dequant_matmulnbits(n, arrays), device)
                    self._nbits_key[n.output[0]] = key
            # MatMul(x, W) with a constant [K, N] weight becomes F.linear(x, W^T): the
            # [N, K] layout PyTorch / HF linears use, which is both AutoDeploy's
            # canonical linear and what cuBLAS's batch-1 decode GEMV is fast on (a
            # [K, N] weight sent one matmul per layer to a split-K kernel, ~28% of
            # Qwen3-0.6B's decode time). Weights with any other use keep their layout.
            mm_w = {
                n.input[1]
                for n in g.node
                if n.op_type == "MatMul" and n.domain in ("", "ai.onnx")
            }
            non_mm_uses = {
                i
                for n in g.node
                if not (n.op_type == "MatMul" and n.domain in ("", "ai.onnx"))
                for i in n.input
            } | {n.input[0] for n in g.node if n.op_type == "MatMul"}
            non_mm_uses |= {o.name for o in g.output}
            self._linear_key: Dict[str, str] = {}
            for t in g.initializer:
                if t.name in skip:
                    continue
                arr = numpy_helper.to_array(t)
                if (
                    t.name in mm_w
                    and t.name not in non_mm_uses
                    and arr.ndim == 2
                    and arr.dtype.kind == "f"
                ):
                    key = t.name + "::T"
                    self._register(key, np.ascontiguousarray(arr.T), device)
                    self._linear_key[t.name] = key
                    continue
                self._register(t.name, arr, device)
            self._nodes = []
            for n in g.node:
                if n.op_type == "Constant" and n.domain in ("", "ai.onnx"):
                    arr = _node_const(n)
                    if arr is not None:
                        self._register(n.output[0], arr, device)
                        continue
                self._nodes.append(n)

            producer = {o: n for n in self._nodes for o in n.output if o}
            consumers: Dict[str, List] = {}
            for n in self._nodes:
                for i in n.input:
                    consumers.setdefault(i, []).append(n)
            self._alias: Dict[str, str] = {}
            if self._kv_inputs:
                self._strip_kv(producer)
            self._attn: Dict[str, _AttnMatch] = {}
            if attention == "sdpa":
                self._match_attention(producer, consumers)
            self._plan = self._schedule(producer)
            self._rope_pair = self._pair_contrib_rope(producer)
            # Inputs each planned node actually reads (see _deps); the rest -- e.g. a
            # stripped past-KV input of GroupQueryAttention -- are passed as None.
            self._node_deps = {id(n): set(self._deps(n)) for n in self._plan}
            self.forward = self._make_forward()

        # ---- construction helpers ---------------------------------------------------

        def _register(self, name: str, arr: np.ndarray, device):
            sv = _const_value(arr)
            if sv is not None:
                self._static[name] = sv
                return
            attr = "w_" + _safe_name(name)
            while hasattr(self, attr):
                attr += "_"
            t = torch.from_numpy(np.ascontiguousarray(arr).copy())
            if device == "meta":
                t = torch.empty(t.shape, dtype=t.dtype, device="meta")
            elif device is not None:
                t = t.to(device)
            if t.is_floating_point():
                self.register_parameter(
                    attr, torch.nn.Parameter(t, requires_grad=False)
                )
            else:
                self.register_buffer(attr, t)
            self._param_of[name] = attr

        def _strip_kv(self, producer):
            for n in self._nodes:
                if n.op_type != "Concat" or len(n.input) != 2:
                    continue
                roots = [self._root_input(i, producer) for i in n.input]
                past = [r in self._kv_inputs for r in roots]
                if past[0] != past[1]:
                    self._alias[n.output[0]] = n.input[1 if past[0] else 0]
            # Values still derived from a past input once the concats are bypassed.
            dep = set(self._kv_inputs)
            for n in self._nodes:
                if n.output[0] in self._alias:
                    continue
                if any(self._resolve(i) in dep for i in n.input if i):
                    dep.update(o for o in n.output if o)
            # A constant table sliced to [:past_len + seq_len] before being indexed by
            # position_ids (HF rotary_emb's cos/sin cache) is the whole table: bounding
            # it by the *traced* length would break every later call with larger
            # positions (the decode steps of a cache-inserting runtime).
            for n in self._nodes:
                if (
                    n.op_type == "Slice"
                    and len(n.input) >= 3
                    and (n.input[0] in self._param_of or n.input[0] in self._static)
                    and n.input[2] in dep
                    and all(
                        v == 0 for v in _ints(self._static.get(n.input[1], _Sh((1,))))
                    )
                ):
                    self._alias[n.output[0]] = n.input[0]

        def _root_input(self, name, producer):
            # Follow single-input value-preserving ops back to a graph input.
            while name in producer and producer[name].op_type in (
                "Cast",
                "Identity",
                "Transpose",
            ):
                name = producer[name].input[0]
            return name

        def _match_attention(self, producer, consumers):
            def only_consumer(name):
                c = consumers.get(name, [])
                return c[0] if len(c) == 1 else None

            def scalar(name):
                v = self._static.get(name)
                return float(v) if isinstance(v, (int, float)) else None

            for sm in self._nodes:
                if sm.op_type != "Softmax":
                    continue
                if _norm_axis(_attrs(sm).get("axis", -1), 4) != 3:
                    continue
                chain, scale, cur = [sm], 1.0, sm.input[0]
                mm1 = None
                while cur in producer:
                    p = producer[cur]
                    if p.op_type == "MatMul":
                        mm1 = p
                        break
                    if p.op_type in ("Cast", "Identity"):
                        nxt = p.input[0]
                    elif p.op_type in ("Mul", "Div") and (
                        scalar(p.input[1]) is not None or scalar(p.input[0]) is not None
                    ):
                        k = 1 if scalar(p.input[1]) is not None else 0
                        c = scalar(p.input[k])
                        scale = scale / c if p.op_type == "Div" else scale * c
                        nxt = p.input[1 - k]
                    elif p.op_type == "Add":
                        # score + mask: follow the operand that leads to the QK MatMul.
                        nxt = next(
                            (i for i in p.input if self._leads_to_matmul(i, producer)),
                            None,
                        )
                    elif p.op_type == "Where":
                        nxt = next(
                            (
                                i
                                for i in p.input[1:]
                                if self._leads_to_matmul(i, producer)
                            ),
                            None,
                        )
                    else:
                        nxt = None
                    if nxt is None:
                        break
                    chain.append(p)
                    cur = nxt
                if mm1 is None:
                    continue
                cur = sm.output[0]
                mm2 = None
                while True:
                    c = only_consumer(cur)
                    if c is None:
                        break
                    if c.op_type == "MatMul" and c.input[0] == cur:
                        mm2 = c
                        break
                    if c.op_type in ("Cast", "Identity", "Dropout"):
                        chain.append(c)
                        cur = c.output[0]
                        continue
                    break
                if mm2 is None:
                    continue
                self._attn[mm2.output[0]] = _AttnMatch(
                    mm2.output[0],
                    mm1.input[0],
                    mm1.input[1],
                    mm2.input[1],
                    scale,
                    chain,
                )

        def _pair_contrib_rope(self, producer):
            """q/k com.microsoft::RotaryEmbedding pairs feeding one GroupQueryAttention.

            AutoDeploy replaces a matched (q, k) rotation with one op inserted where q's
            rotation starts, so k's input must already exist there; the converter
            therefore rotates each pair together, at whichever of the two nodes comes
            first. Maps id(node) -> (q_node, k_node).
            """
            pairs = {}
            for g in self._nodes:
                if g.op_type != "GroupQueryAttention" or len(g.input) < 2:
                    continue
                q, k = producer.get(g.input[0]), producer.get(g.input[1])
                if not all(
                    x is not None
                    and x.domain == "com.microsoft"
                    and x.op_type == "RotaryEmbedding"
                    for x in (q, k)
                ):
                    continue
                if list(q.input[1:4]) != list(k.input[1:4]) or _attrs(q) != _attrs(k):
                    continue
                pairs[id(q)] = pairs[id(k)] = (q, k)
            return pairs

        def _leads_to_matmul(self, name, producer, depth=4):
            while depth and name in producer:
                p = producer[name]
                if p.op_type == "MatMul":
                    return True
                if p.op_type not in ("Mul", "Div", "Cast", "Identity"):
                    return False
                name = p.input[0] if p.input[0] in producer else p.input[1]
                depth -= 1
            return False

        def _deps(self, n) -> List[str]:
            if n.output and n.output[0] in self._attn:
                m = self._attn[n.output[0]]
                return [m.q, m.kt, m.v]
            if n.op_type == "Attention" and self._attention == "sdpa":
                return [i for i in n.input[:3]]
            if (
                n.op_type == "MatMul"
                and n.domain in ("", "ai.onnx")
                and n.input[1] in self._linear_key
            ):
                return [n.input[0], self._linear_key[n.input[1]]]
            if n.op_type == "MatMulNBits" and n.output[0] in self._nbits_key:
                bias = n.input[5] if len(n.input) > 5 and n.input[5] else None
                return [n.input[0], self._nbits_key[n.output[0]]] + (
                    [bias] if bias else []
                )
            if n.op_type == "GroupQueryAttention" and self._attention == "sdpa":
                # past KV and the seqlens_k / total_sequence_length bookkeeping (derived
                # from attention_mask) are the cache-inserting runtime's business.
                rotary = (
                    [i for i in n.input[7:9] if i]
                    if _attrs(n).get("do_rotary", 0)
                    else []
                )
                return [i for i in n.input[:3] if i] + rotary
            return [i for i in n.input if i]

        def _schedule(self, producer):
            need, order, seen = [], [], set()
            stack = [self._resolve(o) for o in self._output_names]
            while stack:
                name = stack.pop()
                if name in seen:
                    continue
                seen.add(name)
                if name in producer:
                    n = producer[name]
                    need.append(n)
                    stack.extend(self._resolve(d) for d in self._deps(n))
                elif name in self._kv_inputs:
                    raise ValueError(
                        f"output depends on stripped KV-cache input {name!r}; "
                        "pass position_ids explicitly or use strip_kv_cache=False"
                    )
                elif (
                    name not in self._static
                    and name not in self._param_of
                    and name not in self._input_names
                ):
                    raise ValueError(
                        f"value {name!r} needed by the kept outputs is not produced by the "
                        f"graph and is not one of inputs={self._input_names}"
                    )
            keep = {id(n) for n in need}
            for n in self._nodes:  # original order is already topological
                if id(n) in keep and n not in order:
                    order.append(n)
            return order

        def _resolve(self, name):
            while name in self._alias:
                name = self._alias[name]
            return name

        # ---- forward ----------------------------------------------------------------

        @property
        def param_names(self) -> Dict[str, str]:
            """ONNX value name -> state-dict key of every weight tensor."""
            return dict(self._param_of)

        def _make_forward(self):
            # torch.export (and AutoDeploy, which calls the model by keyword) binds inputs
            # and dynamic_shapes to forward's declared parameter names, so generate a
            # forward whose parameters are the ONNX input names.
            params = []
            for name in self._input_names:
                p = _safe_name(name)
                if p[0].isdigit() or p in params:
                    p = "in_" + p
                params.append(p)
            src = (
                f"def forward(self, {', '.join(p + '=None' for p in params)}):\n"
                f"    return self._forward_impl(({', '.join(params)},))\n"
            )
            ns: Dict[str, Any] = {}
            exec(src, ns)  # noqa: S102 -- identifiers are sanitized above
            return ns["forward"].__get__(self)

        def _forward_impl(self, args):
            _DEVICE.set(
                next((a.device for a in args if isinstance(a, torch.Tensor)), None)
            )
            _ROPE_MEMO.set({})
            if self._synthetic_pos:
                _POSITIONS.set(args[self._input_names.index("position_ids")])
            env: Dict[str, Any] = dict(self._static)
            for k, attr in self._param_of.items():
                env[k] = getattr(self, attr)
            for name, val in zip(self._input_names, args):
                if val is not None:
                    env[name] = val

            def get(name):
                return env[self._resolve(name)] if name else None

            for n in self._plan:
                out0 = n.output[0]
                if out0 in self._attn:
                    m = self._attn[out0]
                    q, kt, v = get(m.q), get(m.kt), get(m.v)
                    env[out0] = F.scaled_dot_product_attention(
                        q, kt.transpose(-1, -2), v, is_causal=True, scale=m.scale
                    )
                    continue
                pair = self._rope_pair.get(id(n))
                if pair is not None:
                    qn, kn = pair
                    if qn.output[0] in env:  # already rotated with its partner
                        continue
                    if (
                        self._resolve(kn.input[0]) in env
                        and self._resolve(qn.input[0]) in env
                    ):
                        env[qn.output[0]], env[kn.output[0]] = self._contrib_rope_pair(
                            get(qn.input[0]),
                            get(kn.input[0]),
                            *(get(i) for i in qn.input[1:4]),
                            _attrs(qn),
                        )
                        continue
                deps = self._node_deps[id(n)]
                ins = [get(i) if i in deps else None for i in n.input]
                res = self._run(n, ins)
                if not isinstance(res, tuple) or isinstance(res, _Sh):
                    res = (res,)
                for name, val in zip(n.output, res):
                    if name:
                        env[name] = val
            outs = tuple(get(o) for o in self._output_names)
            return outs

        # ---- op implementations ------------------------------------------------------

        def _run(self, n, ins):
            if n.domain == "com.microsoft":
                # ONNX Runtime contrib ops (ORT GenAI builder / optimum exports).
                fn = getattr(self, "_ms_" + n.op_type, None) or getattr(
                    self, "_op_" + n.op_type, None
                )
                if fn is None:
                    raise NotImplementedError(f"contrib op com.microsoft::{n.op_type}")
                return fn(ins, _attrs(n), n)
            if n.domain not in ("", "ai.onnx"):
                raise NotImplementedError(f"custom-domain op {n.domain}::{n.op_type}")
            fn = getattr(self, "_op_" + n.op_type, None)
            if fn is None:
                raise NotImplementedError(f"ONNX op {n.op_type} (node {n.name!r})")
            return fn(ins, _attrs(n), n)

        # shape-producing / shape-arithmetic ops

        def _op_Shape(self, ins, a, n):
            s = tuple(ins[0].shape)
            start, end = a.get("start", 0), a.get("end", None)
            return _Sh(s[start:end])

        def _op_Size(self, ins, a, n):
            return ins[0].numel()

        def _op_Identity(self, ins, a, n):
            return ins[0]

        _op_Dropout = _op_Identity

        def _op_Cast(self, ins, a, n):
            x, to = ins[0], a["to"]
            if _is_shape(x) and to in (TensorProto.INT64, TensorProto.INT32):
                return x
            if to == TensorProto.BOOL and _is_int(x):
                return x != 0
            return _tensor(x).to(_dtype(to))

        def _op_CastLike(self, ins, a, n):
            return _tensor(ins[0]).to(ins[1].dtype)

        def _op_Unsqueeze(self, ins, a, n):
            x = ins[0]
            axes = _ints(ins[1]) if len(ins) > 1 else list(a["axes"])
            if _is_int(x) and axes in ([0], [-1]):
                return _Sh((x,))
            x = _tensor(x)
            rank = x.dim() + len(axes)
            for ax in sorted(_norm_axis(ax, rank) for ax in axes):
                x = x.unsqueeze(ax)
            return x

        def _op_Squeeze(self, ins, a, n):
            x = ins[0]
            axes = (
                _ints(ins[1]) if len(ins) > 1 and ins[1] is not None else a.get("axes")
            )
            if isinstance(x, _Sh) and len(x) == 1:
                return x[0]
            x = _tensor(x)
            if axes is None:
                return x.squeeze()
            for ax in sorted((_norm_axis(ax, x.dim()) for ax in axes), reverse=True):
                x = x.squeeze(ax)
            return x

        def _op_Concat(self, ins, a, n):
            if all(_is_shape(v) for v in ins):
                out: List = []
                for v in ins:
                    out.extend(_ints(v))
                return _Sh(out)
            ref = next(v for v in ins if not _is_shape(v))
            return torch.cat([_tensor(v, ref) for v in ins], dim=a["axis"])

        def _op_Gather(self, ins, a, n):
            data, idx = ins
            axis = a.get("axis", 0)
            if isinstance(data, _Sh):
                if _is_int(idx):
                    return data[idx]
                return _Sh(data[i] for i in _ints(idx))
            if _is_int(idx):
                return data.select(_norm_axis(axis, data.dim()), idx)
            idx = _tensor(idx, data)
            ax = _norm_axis(axis, data.dim())
            if ax == 0 and data.dim() == 2 and idx.dtype in (torch.int64, torch.int32):
                return F.embedding(idx, data)
            idx = torch.where(idx < 0, idx + data.shape[ax], idx)
            out = data.index_select(ax, idx.reshape(-1))
            return out.reshape(
                tuple(data.shape[:ax]) + tuple(idx.shape) + tuple(data.shape[ax + 1 :])
            )

        def _op_Slice(self, ins, a, n):
            data = ins[0]
            if len(ins) > 1:
                starts, ends = _ints(ins[1]), _ints(ins[2])
                axes = _ints(ins[3]) if len(ins) > 3 and ins[3] is not None else None
                steps = _ints(ins[4]) if len(ins) > 4 and ins[4] is not None else None
            else:
                starts, ends, axes = list(a["starts"]), list(a["ends"]), a.get("axes")
                steps = None
            axes = list(axes) if axes is not None else list(range(len(starts)))
            steps = steps or [1] * len(starts)
            if isinstance(data, _Sh):
                ((s, e, st),) = zip(starts, ends, steps)
                return _Sh(data[slice(s, e, st)])
            idx = [slice(None)] * data.dim()
            for s, e, ax, st in zip(starts, ends, axes, steps):
                if st <= 0:
                    raise NotImplementedError("Slice with non-positive step")
                idx[_norm_axis(ax, data.dim())] = slice(s, e, st)
            return data[tuple(idx)]

        def _op_Reshape(self, ins, a, n):
            x, shape = ins[0], _ints(ins[1])
            if isinstance(x, _Sh):
                return x if len(shape) == 1 else _tensor(x).reshape(shape)
            if not a.get("allowzero", 0):
                shape = [
                    x.shape[i] if (isinstance(d, int) and d == 0) else d
                    for i, d in enumerate(shape)
                ]
            return x.reshape(shape)

        def _op_Flatten(self, ins, a, n):
            x = ins[0]
            ax = _norm_axis(a.get("axis", 1), x.dim())
            lead = 1
            for d in x.shape[:ax]:
                lead = lead * d
            return x.reshape(lead, -1)

        def _op_Expand(self, ins, a, n):
            x, shape = _tensor(ins[0]), _ints(ins[1])
            return x.expand(torch.broadcast_shapes(tuple(x.shape), tuple(shape)))

        def _op_Tile(self, ins, a, n):
            return _tensor(ins[0]).repeat(_ints(ins[1]))

        def _op_ConstantOfShape(self, ins, a, n):
            val = a.get("value")
            arr = (
                numpy_helper.to_array(val)
                if val is not None
                else np.zeros(1, np.float32)
            )
            return torch.full(
                _ints(ins[0]),
                arr.reshape(-1)[0].item(),
                dtype=torch.from_numpy(arr).dtype,
                device=_DEVICE.get(),
            )

        def _op_Range(self, ins, a, n):
            start, limit, delta = ins
            if all(_is_shape(v) for v in ins):
                return torch.arange(
                    start, limit, delta, dtype=torch.int64, device=_DEVICE.get()
                )
            ref = next(v for v in ins if not _is_shape(v))
            return torch.arange(
                _tensor(start).item(),
                _tensor(limit).item(),
                _tensor(delta).item(),
                dtype=ref.dtype,
            )

        def _op_Transpose(self, ins, a, n):
            x = ins[0]
            perm = a.get("perm", list(range(x.dim()))[::-1])
            return x.permute(perm)

        def _op_Split(self, ins, a, n):
            x = ins[0]
            ax = _norm_axis(a.get("axis", 0), x.dim())
            split = (
                _ints(ins[1]) if len(ins) > 1 and ins[1] is not None else a.get("split")
            )
            if split is None:
                k = a.get("num_outputs", len(n.output))
                return tuple(torch.chunk(x, k, dim=ax))
            return tuple(torch.split(x, list(split), dim=ax))

        # elementwise, with a pure-Python path for shape arithmetic

        def _binary(self, ins, py, th, op=None):
            x, y = ins
            if _is_shape(x) and _is_shape(y):
                if _is_int(x) and _is_int(y):
                    return py(x, y)
                xs = list(x) if isinstance(x, _Sh) else None
                ys = list(y) if isinstance(y, _Sh) else None
                if xs is None:
                    return _Sh(py(x, e) for e in ys)
                if ys is None:
                    return _Sh(py(e, y) for e in xs)
                if len(xs) == len(ys):
                    return _Sh(py(p, q) for p, q in zip(xs, ys))
            ref = next((v for v in (x, y) if isinstance(v, torch.Tensor)), None)

            def scalar(v):
                return isinstance(v, (float, bool)) or _is_int(v)

            if ref is not None and op is not None:
                # tensor (op) Python number: let torch take the scalar directly --
                # materializing it with torch.tensor() inside forward breaks
                # torch.export on meta-device models (AutoDeploy's export path).
                if isinstance(x, torch.Tensor) and scalar(y):
                    if not (op is operator.truediv and not x.is_floating_point()):
                        return op(x, y)
                if isinstance(y, torch.Tensor) and scalar(x):
                    if not (op is operator.truediv and not y.is_floating_point()):
                        return op(x, y)
            return th(_tensor(x, ref), _tensor(y, ref))

        def _op_Add(self, ins, a, n):
            return self._binary(ins, lambda p, q: p + q, torch.add, op=operator.add)

        def _op_Sub(self, ins, a, n):
            return self._binary(ins, lambda p, q: p - q, torch.sub, op=operator.sub)

        def _op_Mul(self, ins, a, n):
            return self._binary(ins, lambda p, q: p * q, torch.mul, op=operator.mul)

        def _op_Div(self, ins, a, n):
            def th(p, q):
                if not p.is_floating_point():
                    return torch.div(p, q, rounding_mode="trunc")
                return p / q

            return self._binary(ins, lambda p, q: p // q, th, op=operator.truediv)

        def _op_Mod(self, ins, a, n):
            return self._binary(ins, lambda p, q: p % q, torch.remainder)

        def _op_Pow(self, ins, a, n):
            x, y = ins
            if not _is_shape(x) and isinstance(y, (int, float)):
                return torch.pow(x, y)
            if (
                not _is_shape(x)
                and isinstance(y, torch.Tensor)
                and y.numel() == 1
                and y.device.type != "meta"
            ):
                return torch.pow(x, y.to(x.dtype))
            return self._binary(ins, lambda p, q: p**q, torch.pow, op=operator.pow)

        def _op_Max(self, ins, a, n):
            out = ins[0]
            for v in ins[1:]:
                out = self._binary([out, v], max, torch.maximum)
            return out

        def _op_Min(self, ins, a, n):
            out = ins[0]
            for v in ins[1:]:
                out = self._binary([out, v], min, torch.minimum)
            return out

        def _op_Equal(self, ins, a, n):
            return self._binary(ins, lambda p, q: p == q, torch.eq)

        def _op_Less(self, ins, a, n):
            return self._binary(ins, lambda p, q: p < q, torch.lt)

        def _op_LessOrEqual(self, ins, a, n):
            return self._binary(ins, lambda p, q: p <= q, torch.le)

        def _op_Greater(self, ins, a, n):
            return self._binary(ins, lambda p, q: p > q, torch.gt)

        def _op_GreaterOrEqual(self, ins, a, n):
            return self._binary(ins, lambda p, q: p >= q, torch.ge)

        def _op_And(self, ins, a, n):
            return torch.logical_and(_tensor(ins[0]), _tensor(ins[1]))

        def _op_Or(self, ins, a, n):
            return torch.logical_or(_tensor(ins[0]), _tensor(ins[1]))

        def _op_Xor(self, ins, a, n):
            return torch.logical_xor(_tensor(ins[0]), _tensor(ins[1]))

        def _op_Not(self, ins, a, n):
            return torch.logical_not(_tensor(ins[0]))

        def _op_Where(self, ins, a, n):
            c, x, y = ins
            if all(_is_shape(v) or isinstance(v, bool) for v in ins):
                # e.g. Where(Equal(shape, -1), 1, shape): select per element in Python.
                cs, xs, ys = (list(v) if isinstance(v, _Sh) else v for v in (c, x, y))
                size = next(len(v) for v in (cs, xs, ys) if isinstance(v, list))

                def at(v, i):
                    return v[i] if isinstance(v, list) else v

                return _Sh(at(xs, i) if at(cs, i) else at(ys, i) for i in range(size))
            ref = next((v for v in (x, y) if isinstance(v, torch.Tensor)), None)

            def side(v):
                return v if isinstance(v, (float, bool)) else _tensor(v, ref)

            if not isinstance(c, torch.Tensor):
                c = _tensor(c, ref)
            return torch.where(c.to(torch.bool), side(x), side(y))

        def _unary(name):
            def fn(self, ins, a, n):
                return getattr(torch, name)(_tensor(ins[0]))

            return fn

        _op_Sqrt = _unary("sqrt")
        _op_Reciprocal = _unary("reciprocal")
        _op_Neg = _unary("neg")
        _op_Exp = _unary("exp")
        _op_Log = _unary("log")
        _op_Sin = _unary("sin")
        _op_Cos = _unary("cos")
        _op_Tanh = _unary("tanh")
        _op_Sigmoid = _unary("sigmoid")
        _op_Erf = _unary("erf")
        _op_Abs = _unary("abs")
        _op_Floor = _unary("floor")
        _op_Ceil = _unary("ceil")
        _op_Relu = _unary("relu")
        _op_IsNaN = _unary("isnan")
        _op_IsInf = _unary("isinf")

        def _op_Softmax(self, ins, a, n):
            return torch.softmax(
                ins[0], dim=a.get("axis", -1 if self._opset >= 13 else 1)
            )

        def _op_ScatterND(self, ins, a, n):
            if a.get("reduction", "none") != "none":
                raise NotImplementedError("ScatterND with reduction")
            data, idx, upd = _tensor(ins[0]), _tensor(ins[1]), _tensor(ins[2])
            k = idx.shape[-1]
            idx = idx.reshape(-1, k)
            upd = upd.reshape((-1,) + tuple(data.shape[k:]))
            return torch.index_put(data, tuple(idx.unbind(-1)), upd)

        def _op_Trilu(self, ins, a, n):
            k = _ints(ins[1])[0] if len(ins) > 1 and ins[1] is not None else 0
            return torch.triu(ins[0], k) if a.get("upper", 1) else torch.tril(ins[0], k)

        def _op_CumSum(self, ins, a, n):
            if a.get("exclusive", 0) or a.get("reverse", 0):
                raise NotImplementedError("CumSum exclusive/reverse")
            return torch.cumsum(ins[0], dim=_ints(ins[1])[0])

        def _reduce(self, ins, a, fn):
            x = ins[0]
            axes = (
                _ints(ins[1]) if len(ins) > 1 and ins[1] is not None else a.get("axes")
            )
            keep = bool(a.get("keepdims", 1))
            if not axes:
                if a.get("noop_with_empty_axes", 0):
                    return x
                axes = list(range(x.dim()))
            return fn(x, [_norm_axis(ax, x.dim()) for ax in axes], keep)

        def _op_ReduceMean(self, ins, a, n):
            return self._reduce(ins, a, lambda x, d, k: x.mean(dim=d, keepdim=k))

        def _op_ReduceSum(self, ins, a, n):
            return self._reduce(ins, a, lambda x, d, k: x.sum(dim=d, keepdim=k))

        def _op_ReduceMax(self, ins, a, n):
            return self._reduce(ins, a, lambda x, d, k: x.amax(dim=d, keepdim=k))

        def _op_ReduceMin(self, ins, a, n):
            return self._reduce(ins, a, lambda x, d, k: x.amin(dim=d, keepdim=k))

        def _op_MatMul(self, ins, a, n):
            if n.input[1] in self._linear_key:
                w = getattr(self, self._param_of[self._linear_key[n.input[1]]])
                return F.linear(ins[0], w)
            return torch.matmul(ins[0], ins[1])

        def _op_Gemm(self, ins, a, n):
            x, w = ins[0], ins[1]
            x = x.t() if a.get("transA", 0) else x
            w = w.t() if a.get("transB", 0) else w
            y = a.get("alpha", 1.0) * (x @ w)
            if len(ins) > 2 and ins[2] is not None:
                y = y + a.get("beta", 1.0) * ins[2]
            return y

        def _op_LayerNormalization(self, ins, a, n):
            x, w = ins[0], ins[1]
            b = ins[2] if len(ins) > 2 else None
            ax = _norm_axis(a.get("axis", -1), x.dim())
            return F.layer_norm(x, tuple(x.shape[ax:]), w, b, a.get("epsilon", 1e-5))

        @staticmethod
        def _rms(x, w, eps, stash=TensorProto.FLOAT):
            # Same arithmetic order as RMSNormalization's reference body and HF's
            # *RMSNorm.forward -- also the shape AutoDeploy's rmsnorm matcher looks for.
            xs = x.to(_dtype(stash))
            var = xs.pow(2).mean(-1, keepdim=True)
            return w * (xs * torch.rsqrt(var + eps)).to(x.dtype)

        def _op_RMSNormalization(self, ins, a, n):
            x, w = ins
            ax = _norm_axis(a.get("axis", -1), x.dim())
            if ax != x.dim() - 1:
                raise NotImplementedError(
                    "RMSNormalization over more than the last axis"
                )
            return self._rms(
                x, w, a.get("epsilon", 1e-5), a.get("stash_type", TensorProto.FLOAT)
            )

        def _op_SimplifiedLayerNormalization(self, ins, a, n):
            # ONNX Runtime's RMSNorm (registered in the default domain).
            x, w = ins[0], ins[1]
            if _norm_axis(a.get("axis", -1), x.dim()) != x.dim() - 1:
                raise NotImplementedError("SimplifiedLayerNormalization, non-last axis")
            return (self._rms(x, w, a.get("epsilon", 1e-5)),)

        def _op_SkipSimplifiedLayerNormalization(self, ins, a, n):
            # outputs: (RMSNorm(input + skip [+ bias]), mean, inv_std_var, the sum)
            x = ins[0] + ins[1]
            if len(ins) > 3 and ins[3] is not None:
                x = x + ins[3]
            return (self._rms(x, ins[2], a.get("epsilon", 1e-5)), None, None, x)

        @staticmethod
        def _rope_bnsd(x4, cos_c, sin_c, pos):
            """HF rotate-half RoPE of ``x4`` [B, N, S, D] from a [max_pos, rd/2] cos/sin
            cache gathered at ``pos`` [B, S]; the first ``rd`` dims rotate.

            cos/sin are unsqueezed at dim 1 and memoized per forward call, so q and k
            share one node each -- the shape AutoDeploy's match_rope_pattern needs.
            """
            rd, d = 2 * cos_c.shape[-1], x4.shape[-1]
            key = (id(cos_c), id(sin_c), id(pos), x4.dtype)
            memo = _ROPE_MEMO.get({})
            # The gathered [B, S, rd] cos/sin are shared by every layer (outside the
            # pattern); the unsqueeze is *inside* AutoDeploy's pattern, so it must be
            # shared by exactly one q/k pair -- a node also used by other layers makes
            # the matcher refuse the replacement. Hand each unsqueeze out twice.
            if ("base",) + key not in memo:
                cos = torch.cat([cos_c[pos], cos_c[pos]], -1).to(x4.dtype)  # [B, S, rd]
                sin = torch.cat([sin_c[pos], sin_c[pos]], -1).to(x4.dtype)
                memo[("base",) + key] = (cos, sin)
            if key not in memo:
                cos, sin = memo[("base",) + key]
                memo[key] = [cos.unsqueeze(1), sin.unsqueeze(1), 0]
            entry = memo[key]
            cos, sin = entry[0], entry[1]
            entry[2] += 1
            if entry[2] == 2:
                del memo[key]
            # Full rotary: no slicing at all -- a no-op x[..., :d] exports as aten.alias.
            xr, xp = (x4[..., :rd], x4[..., rd:]) if rd < d else (x4, None)
            h = rd // 2
            rot = torch.cat([-xr[..., h:], xr[..., :h]], -1)
            y = xr * cos + rot * sin
            return torch.cat([y, xp], -1) if rd < d else y

        @staticmethod
        def _contrib_rope_prep(x, pos, cos_c, a):
            if a.get("interleaved", 0):
                raise NotImplementedError("com.microsoft::RotaryEmbedding interleaved")
            if a.get("scale", 1.0) != 1.0:
                raise NotImplementedError("com.microsoft::RotaryEmbedding scale != 1")
            rd = 2 * cos_c.shape[-1]
            if a.get("rotary_embedding_dim", 0) not in (0, rd):
                raise NotImplementedError("rotary_embedding_dim != cos_cache width")
            if (
                pos.dim() == 1
                and pos.shape[0] == 1
                and x.dim() == 3
                and x.shape[1] != 1
            ):
                pos = pos + torch.arange(x.shape[1], device=pos.device)[None]
            if x.dim() == 3:
                # Rotate in [B, N, S, D]: the only layout AutoDeploy's rope matcher has.
                nh = a.get("num_heads", 0)
                d = x.shape[-1] // nh if nh else rd
                return x.reshape(x.shape[0], x.shape[1], -1, d).transpose(1, 2), pos
            return x, pos

        @staticmethod
        def _contrib_rope_finish(y4, x):
            return y4.transpose(1, 2).reshape(x.shape) if x.dim() == 3 else y4

        def _contrib_rope_pair(self, xq, xk, pos, cos_c, sin_c, a):
            q4, pos = self._contrib_rope_prep(xq, pos, cos_c, a)
            k4, _ = self._contrib_rope_prep(xk, pos, cos_c, a)
            yq = self._rope_bnsd(q4, cos_c, sin_c, pos)
            yk = self._rope_bnsd(k4, cos_c, sin_c, pos)
            return (
                self._contrib_rope_finish(yq, xq),
                self._contrib_rope_finish(yk, xk),
            )

        def _ms_RotaryEmbedding(self, ins, a, n):
            # com.microsoft::RotaryEmbedding(input, position_ids, cos_cache, sin_cache):
            # input [B, S, N*D] or [B, N, S, D]; cos/sin cache [max_pos, rotary_dim/2].
            x, pos, cos_c, sin_c = ins[:4]
            x4, pos = self._contrib_rope_prep(x, pos, cos_c, a)
            return self._contrib_rope_finish(self._rope_bnsd(x4, cos_c, sin_c, pos), x)

        def _ms_MatMulNBits(self, ins, a, n):
            w = getattr(self, self._param_of[self._nbits_key[n.output[0]]])
            y = F.linear(ins[0], w)
            if len(ins) > 5 and ins[5] is not None:
                y = y + ins[5]
            return y

        def _ms_GroupQueryAttention(self, ins, a, n):
            # com.microsoft::GroupQueryAttention(query, key, value, past_key, past_value,
            #   seqlens_k, total_sequence_length, [cos_cache, sin_cache, ...]):
            # q/k/v [B, S, H*D] (key/value empty: query is packed QKV), past [B, Hkv, P, D].
            if a.get("local_window_size", -1) != -1 or a.get("softcap", 0.0):
                raise NotImplementedError(
                    "GroupQueryAttention sliding window / softcap"
                )
            if a.get("do_rotary", 0) and a.get("rotary_interleaved", 0):
                raise NotImplementedError("GroupQueryAttention interleaved rotary")
            hq, hk = a["num_heads"], a["kv_num_heads"]
            q = ins[0]
            k = ins[1] if len(ins) > 1 else None
            v = ins[2] if len(ins) > 2 else None
            b, s = q.shape[0], q.shape[1]
            if k is None:
                d = q.shape[-1] // (hq + 2 * hk)
                q, k, v = torch.split(q, [hq * d, hk * d, hk * d], -1)
            d = q.shape[-1] // hq
            q = q.reshape(b, s, hq, d).transpose(1, 2)
            k = k.reshape(b, s, hk, d).transpose(1, 2)
            v = v.reshape(b, s, hk, d).transpose(1, 2)
            past = 0
            if self._attention == "exact" and len(ins) > 4 and ins[3] is not None:
                past = ins[3].shape[2]
            if a.get("do_rotary", 0):
                # In-op RoPE from the cos/sin cache inputs. ONNX Runtime derives the
                # positions from seqlens_k (past_len + arange(S), no left padding);
                # "sdpa" mode takes them from forward's position_ids instead, which the
                # cache-inserting runtime supplies (the graph itself has no such input).
                cos_c, sin_c = ins[7], ins[8]
                if self._attention == "exact":
                    pos = past + torch.arange(s, device=q.device)[None].expand(b, s)
                else:
                    pos = _POSITIONS.get()
                q = self._rope_bnsd(q, cos_c, sin_c, pos)
                k = self._rope_bnsd(k, cos_c, sin_c, pos)
            if self._attention == "exact" and len(ins) > 4 and ins[3] is not None:
                k, v = torch.cat([ins[3], k], 2), torch.cat([ins[4], v], 2)
            present_k, present_v = k, v
            if hk != hq:
                # HF repeat_kv spelling, which AutoDeploy's match_repeat_kv recognizes.
                rep, t = hq // hk, k.shape[2]
                k = k[:, :, None].expand(b, hk, rep, t, d).reshape(b, hq, t, d)
                v = v[:, :, None].expand(b, hk, rep, t, d).reshape(b, hq, t, d)
            scale = a.get("scale", 0.0) or None
            if self._attention == "exact" and not (isinstance(past, int) and past == 0):
                t = k.shape[2]
                mask = torch.ones(s, t, dtype=torch.bool, device=q.device).tril(past)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
            return (y.transpose(1, 2).reshape(b, s, hq * d), present_k, present_v)

        def _op_RotaryEmbedding(self, ins, a, n):
            x, cos, sin = ins[0], ins[1], ins[2]
            pos = ins[3] if len(ins) > 3 else None
            if a.get("interleaved", 0) or a.get("rotary_embedding_dim", 0):
                raise NotImplementedError(
                    "RotaryEmbedding interleaved / partial rotary"
                )
            if x.dim() != 4:
                raise NotImplementedError("RotaryEmbedding on 3-D input")
            key = (id(cos), id(sin), id(pos), 1, x.dtype)
            memo = _ROPE_MEMO.get({})
            if key not in memo:
                if pos is not None:
                    cos, sin = cos[pos], sin[pos]
                # cos/sin: [B, S, D/2] -> broadcast over heads of x [B, N, S, D].
                memo[key] = (
                    torch.cat([cos, cos], -1).unsqueeze(1).to(x.dtype),
                    torch.cat([sin, sin], -1).unsqueeze(1).to(x.dtype),
                )
            cos, sin = memo[key]
            h = x.shape[-1] // 2
            rot = torch.cat([-x[..., h:], x[..., :h]], -1)
            return x * cos + rot * sin

        def _op_Attention(self, ins, a, n):
            # Opset-23 Attention. "sdpa" mode: causal, mask and past ignored (the cache-
            # inserting runtime supplies both). "exact" mode: honor mask, past and is_causal.
            q, k, v = ins[0], ins[1], ins[2]
            mask = ins[3] if len(ins) > 3 else None
            exact = self._attention == "exact"
            if exact and len(ins) > 5 and ins[4] is not None:
                k, v = torch.cat([ins[4], k], 2), torch.cat([ins[5], v], 2)
            three_d = q.dim() == 3
            if three_d:
                hq, hk = a["q_num_heads"], a["kv_num_heads"]
                b, s = q.shape[0], q.shape[1]
                q = q.reshape(b, s, hq, -1).transpose(1, 2)
                k = k.reshape(b, k.shape[1], hk, -1).transpose(1, 2)
                v = v.reshape(b, v.shape[1], hk, -1).transpose(1, 2)
            present_k, present_v = k, v
            if k.shape[1] != q.shape[1]:
                rep = q.shape[1] // k.shape[1]
                k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
            if exact:
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask,
                    is_causal=bool(a.get("is_causal", 0)),
                    scale=a.get("scale"),
                )
            else:
                y = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=a.get("scale")
                )
            if three_d:
                y = y.transpose(1, 2).reshape(y.shape[0], y.shape[2], -1)
            return (y, present_k, present_v)

    return OnnxModule


_MODULE_CLS = None


def onnx_to_torch(model, **kwargs):
    """Build an :class:`OnnxModule` from an ONNX model (a ``ModelProto`` or a path).

    Keyword arguments:
        inputs: graph inputs that become ``forward``'s positional/keyword arguments, in order
            (default ``("input_ids", "position_ids")``); all other real inputs must be dead once
            the KV cache is stripped (``attention_mask`` usually is, with ``attention="sdpa"``).
        outputs: graph outputs to return (default: the first one, e.g. ``logits``).
        strip_kv_cache: drop ``past_key_values.*`` inputs, see the module docstring.
        attention: ``"sdpa"`` (recognize attention, emit causal SDPA) or ``"exact"``.
        device: ``"meta"`` to create weight placeholders without data (load them later with
            :func:`onnx_state_dict`), or a device to put the weights on.
    """
    global _MODULE_CLS
    if _MODULE_CLS is None:
        _MODULE_CLS = _build_module_class()
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(model)
    return _MODULE_CLS(model, **kwargs)


def onnx_state_dict(module, model) -> Dict[str, Any]:
    """The ONNX initializer data of ``model`` keyed like ``module.state_dict()``.

    Lets a module built with ``device="meta"`` be filled in with ``load_state_dict`` --
    the loading path TensorRT-LLM AutoDeploy's model factories use. ``module`` is the
    module :func:`onnx_to_torch` returned, or its ``param_names`` mapping (ONNX value
    name -> state-dict key): AutoDeploy loads weights into the *exported* GraphModule,
    whose state-dict keys are the same but which no longer carries the mapping.
    """
    torch = _torch()
    names = module if isinstance(module, dict) else module.param_names
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(model)
    arrays = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "Constant" and n.output[0] in names:
            arrays[n.output[0]] = _node_const(n)
        if n.op_type == "MatMulNBits" and n.input[1] + "::dequant" in names:
            arrays[n.input[1] + "::dequant"] = _dequant_matmulnbits(n, arrays)
    for name in names:
        if name.endswith("::T") and name[:-3] in arrays:
            arrays[name] = np.ascontiguousarray(arrays[name[:-3]].T)
    return {
        attr: torch.from_numpy(np.ascontiguousarray(arrays[name]).copy())
        for name, attr in names.items()
    }
