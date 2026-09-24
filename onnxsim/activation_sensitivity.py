"""Which activations can an NPU-quantized model afford to quantize?

:func:`onnxsim.full_qdq.quantize_full_qdq` applies a mixed-precision *policy*
(nodes kept float, 16-bit tensors in an otherwise 8-bit graph), and
:func:`onnxsim.pick_calibration` picks the calibration method, but finding
the policy was left to per-model scripts (``scripts/android/vision_models/
{bevformer_tiny,sam,streampetr}/sensitivity.py``) that each re-implemented an
"only this op type in int8" / "everything but this op type in int8" sweep.
:mod:`onnxsim.mixed_precision` does the analogous search for *weights*
(block INT4 vs INT8); this module does it for *activations*, which is what
limits int8 on an NPU (ViT outliers, sampling coordinates, detector scores).

- :func:`group_nodes` partitions the quantizable compute nodes into groups:
  per node, per op type, per block (node-name prefixes such as
  ``/blocks.3/...``, with a topological-window fallback), or user groups.
  Data-movement nodes (Reshape, Transpose, ...) are not groups of their own:
  they share their input's quantization, so they follow their producer.
  GridSample is a group: its sampling grid is an activation of its own.
- :func:`analyze_activation_sensitivity` calibrates **once**
  (:func:`onnxsim.calibration.collect_calibration_stats`), then builds one
  :func:`~onnxsim.full_qdq.quantize_full_qdq` model per group from those
  ranges -- ``mode="only"``: quantize just that group; ``"all_but"``: keep
  just that group float -- and scores each against the float model with a
  metric (default :func:`onnxsim.calibration_pick.worst_output_sqnr`, or a
  task metric).
- :func:`search_activation_precision_for_budget` starts with every group at
  the cheapest level of a ladder (``uint8 -> uint16 -> float``) and promotes
  the most sensitive group one step at a time, re-scoring the whole model,
  until the score meets a budget. It returns the ``quantize_full_qdq``
  policy (``exclude_nodes`` + ``tensor_dtypes``) and the quantized model.

Every quantized model is run through ONNX Runtime at its *basic*
optimization level (:func:`onnxsim.calibration_pick.run_outputs`): the
extended level's fused u8s8 kernels saturate on CPUs without VNNI and would
make the ranking depend on the host CPU.

Validation against the policies the Android model ports found by hand
(``scripts/quantization/activation_sensitivity_validate.py``, host only, MSE
ranges, worst-output SQNR, a 20 dB search budget):

- BEVFormer-tiny encoder (op-type groups): all-uint8 13.2 dB; the Gemms
  alone reach 14.0 dB, i.e. the int8 Linears carry nearly all of the loss
  (the port's ``lin8`` vs ``all8`` cosines, 0.985 vs 0.971), with the
  grid-construction Add/Sub and GridSample next (24-27 dB each alone).
- EdgeSAM encoder (35 blocks): all-uint8 9.9 dB, and keeping any single
  block float buys back at most 1.6 dB -- the error is spread over the
  network; the search promotes 33/35 blocks (188/204 nodes) to reach 20 dB,
  matching the port's "post-training quantization can't fix it".
- RF-DETR-Nano (DINOv2): all-uint8 1.8 dB and every backbone block alone
  already ~3 dB; the search keeps 81/82 groups float -- the port's "no int8
  policy beats fp16". (Its u8 export has unnamed Linear nodes, which block
  grouping puts in topological windows; pass ``block_regex`` or explicit
  groups for such graphs.)
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import onnx

from onnxsim.calibration import Tensors, collect_calibration_stats
from onnxsim.calibration_pick import Outputs, run_outputs, worst_output_sqnr
from onnxsim.full_qdq import (
    _SHARED_QPARAM_OPS,
    _constants_to_initializers,
    _data_inputs,
    _float_tensor_names,
    _is_quantized_node,
    quantize_full_qdq,
)

__all__ = [
    "GroupSensitivity",
    "ActivationSensitivityReport",
    "ActivationPrecisionSearch",
    "group_nodes",
    "analyze_activation_sensitivity",
    "search_activation_precision_for_budget",
]

LEVELS = ("uint8", "uint16", "float")
Metric = Callable[[Outputs, Outputs], float]
GroupSpec = Union[str, Mapping[str, Sequence[str]]]

# Data-movement ops whose output shares their input's quantization, so they follow their
# producer's level instead of forming groups. GridSample is not one of them here: its grid
# (sampling-coordinate) input is a real activation of its own, and quantizing it is exactly
# the error a deformable-attention model is sensitive to.
_FOLLOWERS = frozenset(_SHARED_QPARAM_OPS) - {"GridSample"}

# First node-name path component that ends in an index: "blocks.3", "layers_2", "layer1".
_BLOCK_COMPONENT = re.compile(r"[A-Za-z_]*[._]?\d+$")


def _key(n: onnx.NodeProto) -> str:
    """The name ``quantize_full_qdq``'s ``exclude_nodes`` matches a node by."""
    return n.name or n.output[0]


def _prepared(model: onnx.ModelProto) -> onnx.ModelProto:
    m = onnx.ModelProto()
    m.CopyFrom(model)
    _constants_to_initializers(m)
    return onnx.shape_inference.infer_shapes(m)


def _quantizable(m: onnx.ModelProto) -> List[onnx.NodeProto]:
    return [n for n in m.graph.node if _is_quantized_node(n, None, set(), set())]


def _activation_tensors(m: onnx.ModelProto) -> List[str]:
    """Every tensor ``quantize_full_qdq`` may need a range for: the float,
    non-constant data inputs and outputs of every quantizable node."""
    floats = _float_tensor_names(m)
    inits = {i.name for i in m.graph.initializer}
    out, seen = [], set()
    for n in _quantizable(m):
        for x in _data_inputs(n) + [o for o in n.output if o]:
            if x in floats and x not in inits and x not in seen:
                seen.add(x)
                out.append(x)
    return out


def _block_key(name: str, regex: Optional[str]) -> Optional[str]:
    if regex is not None:
        mt = re.search(regex, name)
        if not mt:
            return None
        return mt.group(1) if mt.groups() else mt.group(0)
    parts = [p for p in name.split("/") if p]
    for i, p in enumerate(parts[:-1]):  # the last component is the op itself
        if _BLOCK_COMPONENT.fullmatch(p) and any(c.isdigit() for c in p):
            return "/" + "/".join(parts[: i + 1])
    return None


def group_nodes(
    model: Union[str, onnx.ModelProto],
    groups: GroupSpec = "block",
    block_regex: Optional[str] = None,
    window: int = 8,
) -> Dict[str, List[str]]:
    """Partition ``model``'s quantizable compute nodes into named groups.

    :param groups: ``"node"`` (one group per node), ``"op_type"``,
            ``"block"`` (the node-name prefix up to the first indexed path
            component -- ``/backbone/blocks.3/attn/qkv/MatMul`` ->
            ``/backbone/blocks.3`` -- or ``block_regex``'s first group;
            nodes without one go into topological windows of ``window``
            nodes), or a ``{group: [node names]}`` mapping (nodes it leaves
            out stay at the cheapest level and are never analyzed)
    :returns: ``{group: [node keys]}`` in topological order; a node key is
            the node's name or, if unnamed, its first output name
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    nodes = [n for n in _quantizable(_prepared(model)) if n.op_type not in _FOLLOWERS]
    if isinstance(groups, Mapping):
        known = {_key(n) for n in nodes}
        given: Dict[str, List[str]] = {}
        for g, members in groups.items():
            unknown = [x for x in members if x not in known]
            if unknown:
                raise ValueError(
                    f"group {g!r}: not quantizable compute nodes: {unknown[:3]}"
                )
            given[str(g)] = list(members)
        return given
    out: Dict[str, List[str]] = defaultdict(list)
    if groups == "node":
        for n in nodes:
            out[_key(n)].append(_key(n))
    elif groups == "op_type":
        for n in nodes:
            out[n.op_type].append(_key(n))
    elif groups == "block":
        pending: List[str] = []

        def flush():
            if pending:
                out[f"window:{pending[0]}"].extend(pending)
                pending.clear()

        for n in nodes:
            b = _block_key(n.name, block_regex) if n.name else None
            if b is None:
                pending.append(_key(n))
                if len(pending) >= window:
                    flush()
            else:
                flush()
                out[b].append(_key(n))
        flush()
    else:
        raise ValueError(f"unknown groups: {groups!r}")
    return dict(out)


def _policy(
    m: onnx.ModelProto, level_of: Dict[str, str], default: str
) -> Tuple[List[str], Dict[str, str]]:
    """``quantize_full_qdq`` arguments for a node -> level assignment:
    ``exclude_nodes`` (float nodes) and ``tensor_dtypes`` (uint16 outputs).
    Data-movement nodes follow the producer of their data input (else their
    first assigned consumer), so a float group's Reshape stays float too."""
    qnodes = _quantizable(m)
    producer = {o: n for n in m.graph.node for o in n.output}
    consumers = defaultdict(list)
    for n in m.graph.node:
        for x in n.input:
            consumers[x].append(n)
    level = dict(level_of)
    movement = [n for n in qnodes if n.op_type in _FOLLOWERS]
    for n in movement:  # topological: producers are resolved first
        ins = _data_inputs(n)
        p = producer.get(ins[0]) if ins else None
        if p is not None and _key(p) in level:
            level[_key(n)] = level[_key(p)]
    for n in reversed(movement):  # graph-input-fed chains: take a consumer's level
        if _key(n) not in level:
            for c in consumers.get(n.output[0], []):
                if _key(c) in level:
                    level[_key(n)] = level[_key(c)]
                    break
    exclude, dtypes = [], {}
    floats = _float_tensor_names(m)
    for n in qnodes:
        lv = level.get(_key(n), default)
        if lv == "float":
            exclude.append(_key(n))
        elif lv == "uint16":
            for o in n.output:
                if o and o in floats:
                    dtypes[o] = "uint16"
    return exclude, dtypes


def _quantize(
    m: onnx.ModelProto,
    ranges: Dict[str, Tuple[float, float]],
    level_of: Dict[str, str],
    default: str,
    per_channel: bool,
) -> onnx.ModelProto:
    exclude, dtypes = _policy(m, level_of, default)
    return quantize_full_qdq(
        m,
        None,
        activation_dtype="uint8",
        per_channel=per_channel,
        exclude_nodes=exclude,
        ranges=ranges,
        tensor_dtypes=dtypes,
    )


class _Scorer:
    """One calibration (per fold), then any number of level assignments
    scored against the float model on held-out data."""

    def __init__(
        self,
        model: onnx.ModelProto,
        calibration_data: Sequence[Tensors],
        eval_data: Optional[Sequence[Tensors]],
        metric: Optional[Metric],
        method: str,
        folds: int,
        providers: Optional[Sequence[str]],
        per_channel: bool,
    ):
        self.m = _prepared(model)
        self.metric = metric or worst_output_sqnr
        self.providers = providers
        self.per_channel = per_channel
        names = _activation_tensors(self.m)
        calibration_data = list(calibration_data)
        if folds >= 2:
            if eval_data is not None:
                raise ValueError(
                    "folds cross-fits on calibration_data: pass no eval_data"
                )
            if len(calibration_data) < folds:
                raise ValueError(
                    f"{len(calibration_data)} batches are too few for {folds} folds"
                )
            splits = [
                (
                    [b for i, b in enumerate(calibration_data) if i % folds != f],
                    [b for i, b in enumerate(calibration_data) if i % folds == f],
                )
                for f in range(folds)
            ]
        else:
            splits = [(calibration_data, list(eval_data or calibration_data))]
        self.n = sum(len(ev) for _, ev in splits)
        self.folds = []
        for cal, ev in splits:
            stats = collect_calibration_stats(
                self.m, cal, providers=providers, tensor_names=names
            )
            ranges = stats.ranges(method)
            self.folds.append((ranges, ev, run_outputs(self.m, ev, providers)))
        self.float_score = self._mean(lambda fo, ev, r: self.metric(fo, fo))

    def _mean(self, fn) -> float:
        return float(sum(fn(fo, ev, r) * len(ev) for r, ev, fo in self.folds) / self.n)

    def score(
        self, level_of: Dict[str, str], default: str, max_batches: Optional[int] = None
    ) -> float:
        def one(fo, ev, r):
            if max_batches is not None:
                fo, ev = fo[:max_batches], ev[:max_batches]
            q = _quantize(self.m, r, level_of, default, self.per_channel)
            return self.metric(fo, run_outputs(q, ev, self.providers))

        return self._mean(one)

    def model(self, level_of: Dict[str, str], default: str) -> onnx.ModelProto:
        return _quantize(self.m, self.folds[-1][0], level_of, default, self.per_channel)


@dataclass
class GroupSensitivity:
    group: str
    nodes: List[str]
    dtype: str  # the activation dtype the group was quantized to
    score: float  # metric score of this variant (higher is better)
    delta_vs_float: float  # score - float model's score (<= 0 usually)
    delta_vs_all_quantized: float  # score - every group at ``dtype``'s score
    sensitivity: float  # how much this group hurts; the report is sorted by it


@dataclass
class ActivationSensitivityReport:
    """:func:`analyze_activation_sensitivity`'s result, most sensitive first."""

    mode: str
    float_score: float
    all_quantized_score: Dict[str, float]  # dtype -> every group quantized
    groups: List[GroupSensitivity]

    def top(self, k: int = 10) -> List[GroupSensitivity]:
        return self.groups[:k]


def _resolve(
    model, groups, block_regex, window
) -> Tuple[onnx.ModelProto, Dict[str, List[str]]]:
    if isinstance(model, str):
        model = onnx.load(model)
    return model, group_nodes(model, groups, block_regex=block_regex, window=window)


def analyze_activation_sensitivity(
    model: Union[str, onnx.ModelProto],
    calibration_data: Sequence[Tensors],
    eval_data: Optional[Sequence[Tensors]] = None,
    groups: GroupSpec = "block",
    mode: str = "only",
    activation_dtypes: Sequence[str] = ("uint8",),
    metric: Optional[Metric] = None,
    method: str = "minmax",
    folds: int = 0,
    providers: Optional[Sequence[str]] = None,
    per_channel: bool = True,
    block_regex: Optional[str] = None,
    window: int = 8,
    prescreen_batches: Optional[int] = None,
    refine_top: int = 0,
    verbose: bool = False,
) -> ActivationSensitivityReport:
    """Score every group's activation quantization (see the module docstring).

    ``mode="only"`` quantizes just the group (every other node float):
    ``sensitivity = float_score - score``. ``mode="all_but"`` quantizes every
    other group and keeps this one float: ``sensitivity = score -
    all_quantized_score`` (what keeping it float buys back). The model is
    calibrated once (``folds >= 2``: once per fold, each fold scored by
    ranges from the others, as :func:`onnxsim.pick_calibration`); every
    variant reuses those ranges.

    :param metric: ``metric(float_outputs, quant_outputs) -> float``, higher
            is better (default: worst output's SQNR in dB)
    :param method: calibration method for the shared ranges (any
            :meth:`onnxsim.calibration.CalibrationStats.ranges` method)
    :param prescreen_batches: early-stop for big sweeps: score every group
            on this many eval batches only, then (``refine_top``) re-score
            the ``refine_top`` most sensitive on all of them
    """
    if mode not in ("only", "all_but"):
        raise ValueError(f"unknown mode: {mode!r}")
    for dt in activation_dtypes:
        if dt not in ("uint8", "uint16"):
            raise ValueError(f"unsupported activation dtype: {dt!r}")
    model, grp = _resolve(model, groups, block_regex, window)
    sc = _Scorer(
        model,
        calibration_data,
        eval_data,
        metric,
        method,
        folds,
        providers,
        per_channel,
    )
    return _analyze(
        sc, grp, mode, activation_dtypes, prescreen_batches, refine_top, verbose
    )


def _analyze(
    sc: "_Scorer",
    grp: Dict[str, List[str]],
    mode: str,
    activation_dtypes: Sequence[str],
    prescreen_batches: Optional[int],
    refine_top: int,
    verbose: bool,
) -> ActivationSensitivityReport:
    everything = [x for members in grp.values() for x in members]
    all_q = {
        dt: sc.score({x: dt for x in everything}, "float") for dt in activation_dtypes
    }
    rows: List[GroupSensitivity] = []

    def variant(g: str, dt: str) -> Tuple[Dict[str, str], str]:
        if mode == "only":
            return {x: dt for x in grp[g]}, "float"
        lv = {x: dt for x in everything}
        lv.update({x: "float" for x in grp[g]})
        return lv, dt

    def row(g: str, dt: str, s: float) -> GroupSensitivity:
        sens = sc.float_score - s if mode == "only" else s - all_q[dt]
        return GroupSensitivity(
            g, grp[g], dt, s, s - sc.float_score, s - all_q[dt], sens
        )

    for dt in activation_dtypes:
        for g in grp:
            s = sc.score(*variant(g, dt), max_batches=prescreen_batches)
            rows.append(row(g, dt, s))
            if verbose:
                print(f"  {mode} {g} [{dt}]: {s:.6g}")
    rows.sort(key=lambda r: -r.sensitivity)
    if prescreen_batches is not None and refine_top > 0:
        for i, r in enumerate(rows[:refine_top]):
            rows[i] = row(r.group, r.dtype, sc.score(*variant(r.group, r.dtype)))
        rows[:refine_top] = sorted(rows[:refine_top], key=lambda r: -r.sensitivity)
    return ActivationSensitivityReport(mode, sc.float_score, all_q, rows)


@dataclass
class ActivationPrecisionSearch:
    """:func:`search_activation_precision_for_budget`'s result."""

    meets_budget: bool
    score: float
    float_score: float
    levels: Dict[str, str]  # group -> ladder level
    exclude_nodes: List[str]  # the quantize_full_qdq policy ...
    tensor_dtypes: Dict[str, str]  # ... (activation_dtype = ladder[0])
    model: onnx.ModelProto
    promoted_nodes: int  # compute nodes above the cheapest level
    trace: List[Tuple[str, str, float]] = field(
        default_factory=list
    )  # (group, new level, score)


def estimate_group_macs(
    model: onnx.ModelProto, grp: Dict[str, List[str]]
) -> Dict[str, float]:
    """Rough per-group multiply-accumulates (Conv/Gemm/MatMul from inferred
    shapes, output elements for anything else): a cost hint for the search."""
    m = _prepared(model)
    shape = {}
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        dims = [
            d.dim_value if d.HasField("dim_value") else 1
            for d in vi.type.tensor_type.shape.dim
        ]
        shape[vi.name] = dims
    for t in m.graph.initializer:
        shape[t.name] = list(t.dims)
    by_key = {_key(n): n for n in m.graph.node}
    cost = {}
    for g, members in grp.items():
        c = 0.0
        for k in members:
            n = by_key[k]
            out = shape.get(n.output[0], [1])
            o = float(np.prod(out)) if out else 1.0
            if n.op_type in ("MatMul", "Gemm") and n.input[0] in shape:
                a = shape[n.input[0]]
                kdim = (
                    a[0]
                    if n.op_type == "Gemm"
                    and any(at.name == "transA" and at.i for at in n.attribute)
                    else a[-1]
                )
                c += o * kdim
            elif n.op_type in ("Conv", "ConvTranspose") and n.input[1] in shape:
                w = shape[n.input[1]]
                c += o * float(np.prod(w[1:]))
            else:
                c += o
        cost[g] = c
    return cost


def search_activation_precision_for_budget(
    model: Union[str, onnx.ModelProto],
    calibration_data: Sequence[Tensors],
    eval_data: Optional[Sequence[Tensors]] = None,
    budget: float = 30.0,
    ladder: Sequence[str] = LEVELS,
    groups: GroupSpec = "block",
    metric: Optional[Metric] = None,
    method: str = "minmax",
    folds: int = 0,
    rerank: bool = False,
    costs: Union[None, str, Mapping[str, float]] = None,
    max_steps: Optional[int] = None,
    providers: Optional[Sequence[str]] = None,
    per_channel: bool = True,
    block_regex: Optional[str] = None,
    window: int = 8,
    verbose: bool = False,
) -> ActivationPrecisionSearch:
    """Greedy mixed-precision search: every group starts at ``ladder[0]``;
    while ``metric`` (higher is better) is below ``budget``, promote one
    group one step up the ladder and re-score the whole model.

    Which group: with ``rerank=False`` (default) groups are ranked once by
    :func:`analyze_activation_sensitivity` (``mode="all_but"`` at
    ``ladder[0]``: what keeping each one float buys back) and promoted in
    that order, each to the top of the ladder before the next (one scored
    model per step). ``rerank=True`` tries a one-step promotion of every
    group at every step and keeps the best (one scored model per group per
    step; slower, better when sensitivities interact). Ties prefer the
    cheaper group under ``costs`` (``"macs"`` for
    :func:`estimate_group_macs`, or a ``{group: cost}`` mapping).

    :param budget: the minimum acceptable metric score, e.g. 30 (dB) for the
            default worst-output SQNR, or a matched-detection count
    :param ladder: precision levels, cheapest first; each is ``"uint8"``,
            ``"uint16"`` or ``"float"``
    :returns: an :class:`ActivationPrecisionSearch` -- its ``exclude_nodes``
            / ``tensor_dtypes`` reproduce the model through
            :func:`onnxsim.full_qdq.quantize_full_qdq`
    """
    ladder = list(ladder)
    if not ladder or any(lv not in LEVELS for lv in ladder):
        raise ValueError(f"ladder levels must be among {LEVELS}: {ladder}")
    model, grp = _resolve(model, groups, block_regex, window)
    sc = _Scorer(
        model,
        calibration_data,
        eval_data,
        metric,
        method,
        folds,
        providers,
        per_channel,
    )
    base = ladder[0]
    if costs == "macs":
        cost = estimate_group_macs(model, grp)
    elif costs is None:
        cost = {g: 0.0 for g in grp}
    else:
        cost = {g: float(costs.get(g, 0.0)) for g in grp}  # type: ignore[union-attr]
    idx = {g: 0 for g in grp}

    def levels_of() -> Dict[str, str]:
        return {x: ladder[idx[g]] for g, members in grp.items() for x in members}

    score = sc.score(levels_of(), base)
    trace: List[Tuple[str, str, float]] = [("<start>", base, score)]
    if verbose:
        print(
            f"  start [{base}]: {score:.6g} (float {sc.float_score:.6g}, budget {budget})"
        )
    order: List[str] = []
    if not rerank and score < budget:
        rep = _analyze(  # reuses this search's one calibration
            sc,
            grp,
            "all_but",
            (base,) if base != "float" else ("uint8",),
            None,
            0,
            False,
        )
        order = [
            r.group
            for r in sorted(rep.groups, key=lambda r: (-r.sensitivity, cost[r.group]))
        ]
    steps = 0
    while score < budget and (max_steps is None or steps < max_steps):
        open_groups = [g for g in grp if idx[g] < len(ladder) - 1]
        if not open_groups:
            break
        if rerank:
            best = None
            for g in open_groups:
                idx[g] += 1
                s = sc.score(levels_of(), base)
                idx[g] -= 1
                key = (s, -cost[g])
                if best is None or key > best[0]:
                    best = (key, g, s)
            assert best is not None
            _, g, score = best
            idx[g] += 1
        else:
            g = next(x for x in order if idx[x] < len(ladder) - 1)
            idx[g] += 1
            score = sc.score(levels_of(), base)
        steps += 1
        trace.append((g, ladder[idx[g]], score))
        if verbose:
            print(f"  promote {g} -> {ladder[idx[g]]}: {score:.6g}")
    lv = levels_of()
    exclude, dtypes = _policy(sc.m, lv, base)
    return ActivationPrecisionSearch(
        meets_budget=score >= budget,
        score=score,
        float_score=sc.float_score,
        levels={g: ladder[idx[g]] for g in grp},
        exclude_nodes=exclude,
        tensor_dtypes=dtypes,
        model=sc.model(lv, base),
        promoted_nodes=sum(len(grp[g]) for g in grp if idx[g] > 0),
        trace=trace,
    )
