"""PTQ4ViT (Yuan, Xie, Chen, Xu, Suo, Ma, 2022, ECCV 2022, "PTQ4ViT:
Post-Training Quantization for Vision Transformers with Twin Uniform
Quantization", https://arxiv.org/abs/2111.12293) -- the paper's own **twin
uniform quantization** piece specifically. onnxsim ports the *algorithm*,
not any framework's code, per the same rationale as
:mod:`onnxsim.ibert_gelu`/:mod:`onnxsim.bwa_ptq` (the paper's own reference
implementation quantizes live PyTorch modules with no ONNX export path).

**The problem twin uniform quantization targets.** Two activations inside
a Vision Transformer block have a distribution an ordinary single-scale
uniform quantizer represents badly:

- A ``Softmax`` output (the attention probabilities) lies in ``[0, 1]`` by
  construction, but is heavily concentrated near ``0`` with a thin, important
  tail near ``1`` -- the handful of tokens each query actually attends to
  strongly. :mod:`onnxsim.attention_quantization` already quantizes this
  exact tensor, but with a *fixed* ``1/255`` per-tensor scale that spends
  the same resolution everywhere in ``[0, 1]`` regardless of where the real
  mass sits -- see that module's own docstring. This module is a
  complementary, calibration-driven alternative for the same tensor, not a
  replacement for it.
- A ``Gelu``/``Erf``-decomposed-GELU output is concentrated in two separate
  clusters: a small negative dip (GELU's own behavior for slightly-negative
  inputs) and a much wider spread of positive values -- an asymmetric,
  roughly bimodal shape.

A single uniform quantizer covering the whole observed range wastes most of
its levels on the sparse in-between region and under-resolves the two
regions that actually carry the data's mass.

**Twin uniform quantization, the paper's own fix**: split the value range
at one threshold ``t`` into two sub-ranges, ``[lo, t]`` and ``[t, hi]``, and
quantize each with its *own* independent uniform quantizer (own scale, own
zero point) at the same per-side bit width -- doubling the usable
resolution exactly where the real distribution concentrates its mass, at
the cost of one extra bit of selector information per element (which side
of ``t`` it falls on).

This module finds ``t`` (and each side's ``lo``/``hi``) directly from
calibration data by a small grid search minimizing the mean squared
reconstruction error twin quantization introduces, the same
"search a scalar threshold against a directly-measured reconstruction
error" idea :func:`onnxsim.calibration._mse_threshold` already uses for
ordinary single-scale calibration -- **not** the paper's own reported
split-point/percentile constants, which this module does not try to
reproduce (this project has previously shipped a wrong recalled numeric
constant that only direct verification caught -- see
:mod:`onnxsim.ibert_gelu`'s own docstring for the precedent this follows).
See ``tests/test_ptq4vit.py`` for the empirical check that twin
quantization actually beats an equal-per-side-resolution single quantizer
on synthetic post-Softmax/post-GELU-shaped data.

**Where this is applied**: right after a standalone ``Softmax`` node's
output, and right after a standalone ``Gelu`` node's output or the final
``Mul`` of the standard ``0.5 * x * (1 + Erf(x / sqrt(2)))`` GELU export
decomposition (matched structurally: an ``Erf`` node feeding an ``Add``,
feeding a ``Mul``, feeding a second ``Mul`` -- the same decomposition
:mod:`onnxsim.ibert_gelu` targets, though that module rewrites ``Erf``
itself rather than wrapping the whole GELU's output). The twin
quantize/dequantize is inserted as new ``Less``/``Where``/``Sub``/``Div``/
``Round``/``Clip``/``Mul``/``Add`` nodes immediately after the matched
node's own output, exactly like :mod:`onnxsim.attention_quantization`'s own
Softmax-output quantization and :func:`onnxsim.calibration.quantize_static`'s
QDQ insertion -- the rest of the graph is untouched. Everything computes in
ordinary float32; onnxsim has no lower-than-float32 arithmetic ONNX op, so
this represents the *twin-uniform-ness* of the scheme (two independent
scale/zero-point pairs, selected per element) rather than a literal packed
sub-byte storage format, the same simplification :mod:`onnxsim.ibert_gelu`
documents for its own polynomial.

**What this module does not claim to reproduce**: the paper's own separate
Hessian-guided search for ordinary (non-twin) per-channel *weight*
quantization scales elsewhere in the network -- that is a different piece
of the paper's full pipeline and out of scope here; only the twin-uniform
quantization of Softmax/GELU *activations* is ported. Also not reproduced:
the paper's own reported ViT/DeiT/Swin end-task accuracy numbers, and any
literal low-bit hardware storage format -- see above.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data

_EPS = 1e-12


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


def _twin_quantize_dequantize(
    values: np.ndarray, lo: float, split: float, hi: float, n_levels: int
) -> np.ndarray:
    """Simulates the exact reconstruction the ONNX graph this module
    inserts will compute: independently quantize-dequantize ``values <=
    split`` against ``[lo, split]`` and ``values > split`` against
    ``[split, hi]``, each to ``n_levels`` uniform levels. Used both by the
    calibration search below and directly verifiable against numpy in
    tests (onnxruntime is not bit-exact across CPU architectures, so a
    tight numeric assertion belongs here, not in an onnxruntime round
    trip).
    """
    scale_lo = max(split - lo, _EPS) / (n_levels - 1)
    scale_hi = max(hi - split, _EPS) / (n_levels - 1)
    q_lo = np.clip(np.round((values - lo) / scale_lo), 0, n_levels - 1)
    dq_lo = q_lo * scale_lo + lo
    q_hi = np.clip(np.round((values - split) / scale_hi), 0, n_levels - 1)
    dq_hi = q_hi * scale_hi + split
    return np.where(values <= split, dq_lo, dq_hi)


def _single_uniform_quantize_dequantize(
    values: np.ndarray, lo: float, hi: float, n_levels: int
) -> np.ndarray:
    """An ordinary single-scale uniform quantizer over ``[lo, hi]`` -- the
    baseline twin uniform quantization is compared against, both by
    :func:`_search_twin_split` (to confirm splitting is actually worth it
    on this tensor's data before touching the graph) and by
    ``tests/test_ptq4vit.py`` (to confirm the win empirically rather than
    assuming it by construction).
    """
    scale = max(hi - lo, _EPS) / (n_levels - 1)
    q = np.clip(np.round((values - lo) / scale), 0, n_levels - 1)
    return q * scale + lo


def _search_twin_split(
    values: np.ndarray,
    lo: float,
    hi: float,
    n_levels: int = 256,
    num_candidates: int = 97,
) -> Optional[float]:
    """Grid search over candidate split points ``t`` strictly between
    ``lo`` and ``hi``, minimizing the mean squared error
    :func:`_twin_quantize_dequantize` introduces against ``values`` --
    PTQ4ViT's own "search the split minimizing reconstruction error"
    idea, applied directly (see this module's own docstring) rather than
    via the paper's own reported percentile constants.

    The bar a candidate split has to clear is not an ordinary single
    quantizer at the *same* per-side level count (twin quantization,
    spending one extra selector bit, has roughly double that quantizer's
    raw level count and would then win on almost any data, which would be
    an unfair, not-really-informative comparison) but one at ``2 *
    n_levels`` levels -- the *equal total bit budget* comparison (one
    extra bit spent uniformly on every level, vs. spent on a side
    selector). ``tests/test_ptq4vit.py`` verifies directly that this bar
    is only cleared by a wide margin on distributions with real
    concentration to exploit (a skewed Beta or a bimodal mixture), and is
    roughly a toss-up on a flat distribution with nothing to exploit --
    twin quantization is not a free win "by construction".

    Returns ``None`` when no split clears that bar on this data (e.g.
    ``values`` too small/degenerate to search meaningfully, or a
    distribution flat enough that a split buys nothing beyond what the
    extra bit alone already would) -- the caller then leaves that tensor
    unquantized by this module rather than inserting machinery that would
    only add complexity for no measured benefit.
    """
    v = values.astype(np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2 * n_levels or hi - lo <= _EPS:
        return None

    baseline = _single_uniform_quantize_dequantize(v, lo, hi, 2 * n_levels)
    baseline_mse = float(np.mean((v - baseline) ** 2))

    candidates = np.linspace(lo, hi, num_candidates + 2)[1:-1]
    best_t: Optional[float] = None
    best_mse = baseline_mse
    for t in candidates:
        recon = _twin_quantize_dequantize(v, lo, float(t), hi, n_levels)
        mse = float(np.mean((v - recon) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_t = float(t)
    return best_t


def _find_softmax_targets(graph: onnx.GraphProto) -> List[onnx.NodeProto]:
    return [n for n in graph.node if n.op_type == "Softmax"]


def _find_gelu_targets(graph: onnx.GraphProto) -> List[onnx.NodeProto]:
    """Finds every node whose *output* is a finished GELU activation:
    a standalone ``Gelu`` node, or the final ``Mul`` of the standard
    ``0.5 * x * (1 + Erf(x / sqrt(2)))`` export decomposition (an ``Erf``
    feeding an ``Add``, feeding a ``Mul``, feeding a second ``Mul`` --
    matched structurally, the same decomposition
    :mod:`onnxsim.ibert_gelu` targets by its ``Erf`` node, though this
    module wraps the whole GELU output rather than rewriting ``Erf``
    itself). Matching by structure alone (not also checking the ``Add``'s
    and second ``Mul``'s constant operands are actually ~1.0/~0.5) is a
    deliberate simplification: it can in principle match a non-GELU
    ``Erf``-based expression with the same node shape, which is why this
    is only ever used to *wrap* the existing output with a quantize-
    dequantize round trip, never to change what the graph computes.
    """
    consumers_by_input: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            consumers_by_input.setdefault(inp, []).append(node)

    targets = [n for n in graph.node if n.op_type == "Gelu"]

    for node in graph.node:
        if node.op_type != "Erf" or len(node.input) != 1:
            continue
        add_node = next(
            (
                c
                for c in consumers_by_input.get(node.output[0], [])
                if c.op_type == "Add"
            ),
            None,
        )
        if add_node is None:
            continue
        mul1 = next(
            (
                c
                for c in consumers_by_input.get(add_node.output[0], [])
                if c.op_type == "Mul"
            ),
            None,
        )
        if mul1 is None:
            continue
        mul2 = next(
            (
                c
                for c in consumers_by_input.get(mul1.output[0], [])
                if c.op_type == "Mul"
            ),
            None,
        )
        if mul2 is None:
            continue
        targets.append(mul2)

    return targets


def _insert_twin_quantize(
    graph: onnx.GraphProto,
    target_output: str,
    lo: float,
    split: float,
    hi: float,
    n_levels: int,
    tag: str,
    taken_names: set,
) -> None:
    """Rewires every consumer of ``target_output`` to instead read the
    twin-uniform quantize-dequantize round trip of it, and appends the new
    nodes/initializers implementing that round trip -- the same
    "quantize/dequantize wraps the value in place" splice
    :mod:`onnxsim.attention_quantization` uses for its own Softmax-output
    quantization. ``target_output`` must not itself be a graph output name
    (the caller filters those out) -- rewiring a graph output would need
    renaming the output's ``ValueInfoProto``, not a node input.
    """
    # Snapshot the consumers *before* creating any new node -- the new
    # nodes below (mask/shifted_lo/shifted_hi) themselves read
    # ``target_output``, and would otherwise get caught by the same
    # rewiring loop and rewired into a self-referential cycle.
    old_consumers = [n for n in graph.node if target_output in n.input]

    new_nodes: List[onnx.NodeProto] = []

    def _const(value: float, suffix: str) -> str:
        name = _unique_name(f"ptq4vit_{tag}_{suffix}", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(np.array(value, dtype=np.float32), name=name)
        )
        return name

    def _node(op_type: str, inputs: List[str], suffix: str, **attrs) -> str:
        out_name = _unique_name(f"ptq4vit_{tag}_{suffix}", taken_names)
        new_nodes.append(
            onnx.helper.make_node(
                op_type,
                inputs,
                [out_name],
                name=_unique_name(f"ptq4vit_{tag}_{suffix}_node", taken_names),
                **attrs,
            )
        )
        return out_name

    lo_name = _const(lo, "lo")
    split_name = _const(split, "split")
    scale_lo_name = _const(max(split - lo, _EPS) / (n_levels - 1), "scale_lo")
    scale_hi_name = _const(max(hi - split, _EPS) / (n_levels - 1), "scale_hi")
    zero_name = _const(0.0, "zero")
    max_level_name = _const(float(n_levels - 1), "max_level")

    mask_name = _node("Less", [target_output, split_name], "mask")

    shifted_lo = _node("Sub", [target_output, lo_name], "shifted_lo")
    scaled_lo = _node("Div", [shifted_lo, scale_lo_name], "scaled_lo")
    rounded_lo = _node("Round", [scaled_lo], "rounded_lo")
    clipped_lo = _node("Clip", [rounded_lo, zero_name, max_level_name], "clipped_lo")
    dq_lo = _node("Mul", [clipped_lo, scale_lo_name], "dq_lo_scaled")
    dq_lo = _node("Add", [dq_lo, lo_name], "dq_lo")

    shifted_hi = _node("Sub", [target_output, split_name], "shifted_hi")
    scaled_hi = _node("Div", [shifted_hi, scale_hi_name], "scaled_hi")
    rounded_hi = _node("Round", [scaled_hi], "rounded_hi")
    clipped_hi = _node("Clip", [rounded_hi, zero_name, max_level_name], "clipped_hi")
    dq_hi = _node("Mul", [clipped_hi, scale_hi_name], "dq_hi_scaled")
    dq_hi = _node("Add", [dq_hi, split_name], "dq_hi")

    result_name = _node("Where", [mask_name, dq_lo, dq_hi], "result")

    # Splice the new nodes in right after target_output's own producer --
    # appending them at the end of graph.node instead would leave any
    # existing consumer that now reads `result_name` (rewired below)
    # pointing at a node that appears *later* in the list than it does,
    # breaking the topological order onnx.checker requires.
    producer_index = next(
        i for i, n in enumerate(graph.node) if target_output in n.output
    )
    for offset, new_node in enumerate(new_nodes):
        graph.node.insert(producer_index + 1 + offset, new_node)

    for node in old_consumers:
        for i, inp in enumerate(node.input):
            if inp == target_output:
                node.input[i] = result_name


def apply_ptq4vit_quantization(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    n_levels: int = 256,
) -> onnx.ModelProto:
    """Applies PTQ4ViT's own twin uniform quantization (see this module's
    docstring) to every standalone ``Softmax`` output and every GELU
    output (a standalone ``Gelu`` node, or the standard ``Erf``-decomposed
    GELU's final ``Mul``) in ``model``.

    For each matched tensor, ``model`` is run over ``calibration_data``
    (falling back to :func:`onnxsim.calibration.generate_random_calibration_data`
    when omitted, the same default every other calibration-based
    ``quantize_*``/``apply_*`` function in this package uses) to observe
    its actual values, then :func:`_search_twin_split` finds the split
    point minimizing reconstruction error directly against those observed
    values. A tensor whose search finds no split that beats a single
    ordinary uniform quantizer (see :func:`_search_twin_split`) is left
    unquantized by this module.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to observe each
            matched tensor's real values from. Each batch is a
            ``{input_name: np.ndarray}`` dict matching the model's graph
            inputs -- see :func:`onnxsim.calibration.generate_random_calibration_data`
            (the default, a quick smoke test) and
            :func:`onnxsim.calibration.load_huggingface_calibration_data`
            (real data, a much better calibration source for real
            deployment).
    :param num_calibration_samples: number of random batches to generate
            when ``calibration_data`` is not supplied
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run calibration on
    :param n_levels: number of uniform levels *each* sub-quantizer uses
            (default 256, an 8-bit round trip per side -- the paper's own
            "twin" scheme costs one extra selector bit versus a single
            8-bit quantizer at this setting)
    :returns: ``model`` with every matched tensor's consuming nodes
            rewired to read the twin-uniform quantize-dequantize round
            trip of it instead. A model with no matching ``Softmax``/GELU
            pattern, or an opset older than 11 (the 3-input ``Clip`` form
            this module's quantize-dequantize round trip needs), is
            returned unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if not _has_min_opset(model, 11):
        return model

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    graph_output_names = {o.name for o in graph.output}
    softmax_targets = [
        n for n in _find_softmax_targets(graph) if n.output[0] not in graph_output_names
    ]
    gelu_targets = [
        n for n in _find_gelu_targets(graph) if n.output[0] not in graph_output_names
    ]
    candidate_names = [n.output[0] for n in softmax_targets] + [
        n.output[0] for n in gelu_targets
    ]
    if not candidate_names:
        return out

    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_calibration_samples, seed=seed
        )

    probe_model = _add_probe_outputs(model, candidate_names)
    collected: Dict[str, List[np.ndarray]] = {name: [] for name in candidate_names}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in candidate_names:
            arr = np.asarray(result[name])
            if arr.size:
                collected[name].append(arr.ravel())

    taken_names = _all_names(graph)
    for i, node in enumerate(softmax_targets):
        name = node.output[0]
        values = collected.get(name, [])
        if not values:
            continue
        split = _search_twin_split(
            np.concatenate(values), lo=0.0, hi=1.0, n_levels=n_levels
        )
        if split is None:
            continue
        _insert_twin_quantize(
            graph, name, 0.0, split, 1.0, n_levels, f"softmax{i}", taken_names
        )

    for i, node in enumerate(gelu_targets):
        name = node.output[0]
        values = collected.get(name, [])
        if not values:
            continue
        v = np.concatenate(values)
        lo = float(v.min())
        hi = float(v.max())
        split = _search_twin_split(v, lo=lo, hi=hi, n_levels=n_levels)
        if split is None:
            continue
        _insert_twin_quantize(
            graph, name, lo, split, hi, n_levels, f"gelu{i}", taken_names
        )

    return out
