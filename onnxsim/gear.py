"""GEAR (Kang et al., 2024, "GEAR: An Efficient KV Cache Compression Recipe
for Near-Lossless Generative Inference of LLM",
https://arxiv.org/abs/2403.05527). onnxsim ports the algorithm, not any
framework's code, per the same rationale as
:mod:`onnxsim.kv_cache_quantization`/:mod:`onnxsim.rotatekv` (GEAR's own
reference implementation quantizes live PyTorch KV-cache tensors inside a
custom generation loop, with no ONNX export path).

Distinguishing this module from its two nearest siblings, both already in
this repo:

- :mod:`onnxsim.kv_cache_quantization` (KIVI/KVQuant) quantizes a matched
  ``Concat(past, new, axis=seq)`` KV-cache stream to INT8 -- per-channel for
  Key, per-token for Value -- and stops there. Whatever reconstruction error
  that quantization leaves behind is simply accepted; nothing in that module
  measures or corrects it.
- :mod:`onnxsim.low_rank_compensation` (LoRC, from ZeroQuant-V2) *does*
  correct a quantizer's leftover reconstruction error, via the exact same
  Eckart-Young truncated-SVD argument this module reuses below -- but only
  for a **weight** matrix's error, which is a static, offline-known
  quantity (``float_weight - dequantized_weight``, computed once with no
  calibration data at all), and only with a **low-rank** term -- no sparse
  component.

GEAR's own distinguishing contribution, applied here to the KV cache (an
*activation*, per :mod:`onnxsim.kv_cache_quantization`'s own docstring on
why that already makes every KV-cache technique in this repo fundamentally
different from a weight quantizer) rather than to a weight matrix: decompose
the KV-cache quantizer's *own* reconstruction error into (1) a **low-rank**
component -- the same truncated-SVD, Eckart-Young-optimal idea
:mod:`onnxsim.low_rank_compensation` already uses, capturing the
*coherent*, structured part of the error that is shared across many
tokens/channels -- plus (2) a **sparse** component -- the error's own
top-magnitude, *individually* large entries, which a low-rank approximation
alone represents poorly (the same "outliers a shared correction can't
reach" problem :mod:`onnxsim.spqr` already solves for weights, just fit
here per-channel rather than per-element -- see below).

**Why this has to be calibration-fit and static, not a genuine per-step
SVD.** GEAR's own reference algorithm recomputes this whole decomposition
fresh, from the *actual* current residual, at every decode step, over the
*entire* growing cache -- an expensive, data-dependent recomputation the
paper's own eager PyTorch loop can afford but a single static ONNX graph
rewrite cannot express (there is no ONNX op for "run SVD on this runtime
tensor"). Root cause, and the same one :mod:`onnxsim.kv_cache_quantization`
already documents for why a past token's stored reconstruction is never
revised: once a token is quantized into the INT8 cache, its original float
value is gone -- for an *already-cached* ("past") token, there is no live
tensor left in the graph to compute an exact residual from, so no runtime
op could recompute GEAR's decomposition for it even in principle. This
module therefore fits the low-rank subspace (a fixed ``[head_dim,
head_dim]`` projector ``P``, rank ``r``) and the outlier-channel selection
(a fixed ``[head_dim]`` 0/1 mask) **once, from calibration data** -- the
same calibration-driven approach :mod:`onnxsim.kv_cache_quantization`
already uses for its own static per-channel scale -- and applies that fixed
operator, at every step, only to the **freshly produced ("new") token(s)**
own *true* residual, which -- unlike a past token's -- genuinely is still a
live tensor in the graph at the moment it is quantized, no calibration
needed for *that* part. A past token's already-written reconstruction is
never revisited or improved after the fact, exactly like
:mod:`onnxsim.kv_cache_quantization`/:mod:`onnxsim.rotatekv`'s own past-cache
notes. This is a documented, reasonable simplification given onnxsim's own
static-graph-rewrite constraints, the same kind of honest scope note
:mod:`onnxsim.billm`'s own docstring uses for its own simplification
relative to its source paper.

A second simplification, also documented: the paper applies its own
quantizer (uniform or a custom "residual-friendly" scheme) to Key and Value
alike, with no KIVI-style asymmetric per-channel-vs-per-token treatment.
This module follows that: every matched
:mod:`onnxsim.kv_cache_quantization` ``Concat(past, new, axis=seq)`` stream
(Key- and Value-shaped alike, found via that module's own
:func:`onnxsim.kv_cache_quantization._find_kv_cache_candidates` matcher,
reused rather than reimplemented) gets the same static, per-channel INT8
base quantization plus this module's own low-rank+sparse residual
correction -- there is no separate Value-style code path here.

**Fitting, from calibration data (see** :func:`generate_random_calibration_data`
**/** :func:`load_huggingface_calibration_data` **):**

1. Run calibration data through the float model, probing each matched
   stream's ``new`` activation (this step's own freshly computed K/V, before
   quantization) -- the same probe technique
   :mod:`onnxsim.kv_cache_quantization`/:mod:`onnxsim.rotatekv` already use.
2. Compute the same per-channel absmax/127 scale
   :mod:`onnxsim.kv_cache_quantization`'s Key-style path uses, and, from it,
   each calibration token's exact quantization residual
   ``y = x - dequantize(quantize(x))`` (using numpy's round-half-to-even,
   matching ``QuantizeLinear``'s own rounding mode exactly).
3. **Low-rank:** the truncated SVD of the calibration residual matrix ``y``
   (``[num_calibration_tokens, head_dim]``) -- the exact same Eckart-Young
   argument :mod:`onnxsim.low_rank_compensation` already uses, here applied
   to a KV-cache activation's calibration-measured residual instead of a
   weight's exact one. Keeping the top ``rank`` right-singular vectors
   ``V_r`` (``[head_dim, rank]``) gives the calibration data's own
   dominant, shared error directions; ``P = V_r @ V_r.T`` is the orthogonal
   projector onto that subspace, baked into the graph as one ``MatMul``
   weight.
4. **Sparse:** what the low-rank projection above still leaves behind on
   that same calibration data (``y - y @ P``), averaged per channel. The
   ``outlier_fraction`` channels with the largest leftover magnitude --
   i.e. the ones the shared low-rank subspace represents worst -- become a
   fixed 0/1 per-channel mask, baked into the graph as one elementwise
   ``Mul``. Unlike :mod:`onnxsim.spqr` (whose outliers are individual,
   scattered ``(row, col)`` weight positions, needing ``ScatterND`` plus
   explicit index/value pairs to store efficiently), an outlier here is a
   whole **channel** -- and a KV-cache stream's ``head_dim`` is always
   small (a single attention head's own dimension, typically well under a
   few hundred) -- so a plain dense ``[head_dim]`` mask is already as
   compact as any sparse encoding would be, with far less graph machinery.

Graph rewrite, per matched stream (see
:mod:`onnxsim.kv_cache_quantization` for the exact ``Concat(past, new,
axis=seq)`` match):

    Before:
      past: graph input, float32 [..., seq_past, head_dim]
      new:  float32 [..., seq_new, head_dim]        -- this step's own K/V
      present = Concat(past, new, axis=seq)          -- graph output, and
                consumed by the attention math

    After:
      past: graph input, INT8 [..., seq_past, head_dim]     -- dtype changed
      scale: initializer, float32 [head_dim]                   -- per-channel
      zero_point: initializer, INT8 [head_dim], all zero       -- symmetric
      p: initializer, float32 [head_dim, head_dim]      -- fitted rank-r
         projector (present only when rank > 0)
      sparse_mask: initializer, float32 [head_dim], 0/1  -- fitted outlier
         mask (present only when outlier_fraction selects at least one
         channel)
      new_q = QuantizeLinear(new, scale, zero_point, axis=-1)
      present = Concat(past, new_q, axis=seq)         -- INT8 graph output,
                unchanged from onnxsim.kv_cache_quantization's own rewrite
      past_dequant = DequantizeLinear(past, scale, zero_point, axis=-1)
      new_dequant = DequantizeLinear(new_q, scale, zero_point, axis=-1)
      new_residual = new - new_dequant                -- this step's *true*
                     residual, exact (still has the live float `new` tensor)
      low_rank_term = new_residual @ p                 -- coherent part
      remainder = new_residual - low_rank_term
      sparse_term = remainder * sparse_mask             -- outlier-channel part
      new_corrected = new_dequant + low_rank_term + sparse_term
      present_corrected = Concat(past_dequant, new_corrected, axis=seq)
      <every other consumer of the old float present now reads
       present_corrected instead>

``past_dequant`` never gets ``low_rank_term``/``sparse_term`` added -- it is
plain per-channel dequantization, identical to
:mod:`onnxsim.kv_cache_quantization`'s own baseline, because (as above)
there is no live float value left to compute a past token's true residual
from. Only ``new_corrected`` -- this step's own freshly produced token(s) --
ever receives the low-rank/sparse compensation, and whatever reconstruction
a token gets when it is "new" is what it keeps forever afterward, the same
"past is never revised" property :mod:`onnxsim.kv_cache_quantization`/
:mod:`onnxsim.rotatekv` already document for their own calibrated rewrites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.kv_cache_quantization import _find_kv_cache_candidates, _KvCacheCandidate


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


@dataclass
class _GearFit:
    scale: np.ndarray  # float32 [head_dim]
    projector: Optional[np.ndarray]  # float32 [head_dim, head_dim] or None
    sparse_mask: Optional[np.ndarray]  # float32 [head_dim] or None


def _fit_gear(
    candidates: Sequence[_KvCacheCandidate],
    model: onnx.ModelProto,
    calibration_data: Sequence[Tensors],
    providers: Optional[Sequence[str]],
    rank: int,
    outlier_fraction: float,
) -> Dict[str, _GearFit]:
    probe_names = sorted({c.new_name for c in candidates})
    probe = _add_probe_outputs(model, probe_names)

    samples: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        outputs = backend.run_model(probe, batch, providers=providers)
        for name in probe_names:
            arr = np.asarray(outputs[name], dtype=np.float64)
            if arr.ndim == 0:
                continue
            samples[name].append(arr.reshape(-1, arr.shape[-1]))

    fits: Dict[str, _GearFit] = {}
    for name, batches in samples.items():
        if not batches:
            continue
        x = np.concatenate(batches, axis=0)  # [N, head_dim]
        head_dim = x.shape[1]

        channel_absmax = np.abs(x).max(axis=0)
        scale = (np.maximum(channel_absmax, 1e-12) / 127.0).astype(np.float32)

        codes = np.clip(np.round(x / scale), -128, 127)
        dequant = codes * scale
        residual = x - dequant  # [N, head_dim], exact simulated QDQ residual

        r = max(0, min(rank, head_dim, x.shape[0]))
        projector = None
        remainder = residual
        if r > 0:
            _u, _s, vt = np.linalg.svd(residual, full_matrices=False)
            v_r = vt[:r].T  # [head_dim, r]
            projector = (v_r @ v_r.T).astype(np.float32)
            remainder = residual - residual @ projector

        sparse_mask = None
        num_outliers = int(round(outlier_fraction * head_dim))
        if num_outliers > 0:
            channel_score = np.abs(remainder).mean(axis=0)
            outlier_channels = np.argsort(channel_score)[-num_outliers:]
            mask = np.zeros(head_dim, dtype=np.float32)
            mask[outlier_channels] = 1.0
            sparse_mask = mask

        fits[name] = _GearFit(scale=scale, projector=projector, sparse_mask=sparse_mask)

    return fits


def _rewire_consumers(
    graph: onnx.GraphProto, c: _KvCacheCandidate, replacement_name: str
) -> None:
    # See onnxsim.kv_cache_quantization._rewire_consumers for why this must
    # run before any new node referencing c.present_name is inserted.
    for node in graph.node:
        if node is c.concat_node:
            continue
        for i, inp in enumerate(node.input):
            if inp == c.present_name:
                node.input[i] = replacement_name


def _apply_gear(
    graph: onnx.GraphProto,
    c: _KvCacheCandidate,
    fit: _GearFit,
    taken_names: Set[str],
    input_by_name: Dict[str, onnx.ValueInfoProto],
    output_by_name: Dict[str, onnx.ValueInfoProto],
) -> None:
    prefix = f"{c.present_name}_gear"
    num_channels = fit.scale.shape[0]

    scale_name = _unique_name(f"{prefix}_scale", taken_names)
    graph.initializer.append(onnx.numpy_helper.from_array(fit.scale, name=scale_name))
    zp_name = _unique_name(f"{prefix}_zero_point", taken_names)
    zp = np.zeros(num_channels, dtype=np.int8)
    graph.initializer.append(onnx.numpy_helper.from_array(zp, name=zp_name))

    # past: FLOAT -> INT8 (same shape) -- unchanged from
    # onnxsim.kv_cache_quantization's own Key-style rewrite.
    past_input = input_by_name[c.past_name]
    past_input.type.tensor_type.elem_type = onnx.TensorProto.INT8

    new_nodes: List[onnx.NodeProto] = []

    def _new(op_type: str, inputs: List[str], out_suffix: str, **attrs) -> str:
        out_name = _unique_name(f"{prefix}_{out_suffix}", taken_names)
        node = onnx.helper.make_node(
            op_type,
            inputs,
            [out_name],
            name=_unique_name(f"{prefix}_{out_suffix}_node", taken_names),
            **attrs,
        )
        new_nodes.append(node)
        return out_name

    new_q_name = _new(
        "QuantizeLinear",
        [c.new_name, scale_name, zp_name],
        "new_q",
        axis=c.channel_axis,
    )
    quantize_node = new_nodes.pop()

    if c.new_is_first_input:
        c.concat_node.input[0] = new_q_name
    else:
        c.concat_node.input[1] = new_q_name
    present_output = output_by_name[c.present_name]
    present_output.type.tensor_type.elem_type = onnx.TensorProto.INT8

    past_dequant_name = _new(
        "DequantizeLinear",
        [c.past_name, scale_name, zp_name],
        "past_dequant",
        axis=c.channel_axis,
    )
    new_dequant_name = _new(
        "DequantizeLinear",
        [new_q_name, scale_name, zp_name],
        "new_dequant",
        axis=c.channel_axis,
    )
    residual_name = _new("Sub", [c.new_name, new_dequant_name], "new_residual")

    correction_terms = []
    remainder_name = residual_name
    if fit.projector is not None:
        p_name = _unique_name(f"{prefix}_p", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(fit.projector, name=p_name)
        )
        low_rank_name = _new("MatMul", [residual_name, p_name], "low_rank")
        correction_terms.append(low_rank_name)
        if fit.sparse_mask is not None:
            remainder_name = _new("Sub", [residual_name, low_rank_name], "remainder")

    if fit.sparse_mask is not None:
        mask_name = _unique_name(f"{prefix}_sparse_mask", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(fit.sparse_mask, name=mask_name)
        )
        sparse_name = _new("Mul", [remainder_name, mask_name], "sparse")
        correction_terms.append(sparse_name)

    new_corrected_name = new_dequant_name
    for term_name in correction_terms:
        new_corrected_name = _new("Add", [new_corrected_name, term_name], "corrected")

    if c.new_is_first_input:
        concat_inputs = [new_corrected_name, past_dequant_name]
    else:
        concat_inputs = [past_dequant_name, new_corrected_name]
    present_corrected_name = _new(
        "Concat", concat_inputs, "present_corrected", axis=c.seq_axis
    )

    _rewire_consumers(graph, c, present_corrected_name)

    concat_idx = next(i for i, n in enumerate(graph.node) if n is c.concat_node)
    graph.node.insert(concat_idx, quantize_node)
    for offset, node in enumerate(new_nodes):
        graph.node.insert(concat_idx + 2 + offset, node)


def apply_gear(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    rank: int = 4,
    outlier_fraction: float = 0.05,
) -> onnx.ModelProto:
    """Applies GEAR-style low-rank-plus-sparse residual compensation (see
    this module's own docstring) on top of static per-channel INT8
    quantization, to every ``Concat(past, new, axis=seq)`` KV-cache stream
    :mod:`onnxsim.kv_cache_quantization` can find (Key- and Value-shaped
    streams alike -- see the module docstring for why there is no
    asymmetric treatment here).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches (each a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs) used to fit each matched stream's own per-channel scale,
            low-rank projector, and outlier-channel mask -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and :func:`onnxsim.load_huggingface_calibration_data`
            (real data, a more representative fit than random input)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to calibrate on
    :param rank: the low-rank correction's rank (clamped to
            ``min(rank, head_dim, num_calibration_tokens)`` per stream);
            ``0`` disables the low-rank term entirely (sparse-only)
    :param outlier_fraction: fraction of each stream's channels (by count,
            rounded to the nearest whole channel) kept as an explicit sparse
            correction after the low-rank term is subtracted; ``0.0``
            disables the sparse term entirely (low-rank-only)
    :returns: ``model`` with every matched stream's ``past_*`` graph input
            and ``present_*`` graph output changed to INT8 (identical to
            :func:`onnxsim.quantize_kv_cache`'s own Key-style rewrite), and
            every other consumer of the old float ``present_*`` rewired to
            a low-rank-plus-sparse-corrected reconstruction -- see the
            module docstring's diagram; a stream whose calibration
            activation never appeared in any batch is left untouched, as is
            the whole model when no stream matches at all, or when
            ``model``'s opset is older than 13 (``QuantizeLinear``/
            ``DequantizeLinear``'s per-channel ``axis`` needs opset 13)
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    if not _has_min_opset(model, 13):
        return model

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    candidates = _find_kv_cache_candidates(graph)
    if not candidates:
        return out

    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    fits = _fit_gear(
        candidates, model, calibration_data, providers, rank, outlier_fraction
    )
    if not fits:
        return out

    taken_names: Set[str] = _all_names(graph)
    input_by_name = {i.name: i for i in graph.input}
    output_by_name = {o.name: o for o in graph.output}

    for c in candidates:
        fit = fits.get(c.new_name)
        if fit is None:
            continue
        _apply_gear(graph, c, fit, taken_names, input_by_name, output_by_name)

    return out
