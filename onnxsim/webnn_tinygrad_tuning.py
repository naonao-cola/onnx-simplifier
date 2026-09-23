"""Native, per-node "WebNN vs. tuned tinygrad" tuning: times each node of a
(simplified) ONNX model on rustnn (a native W3C WebNN implementation, see
``onnxsim.rustnn_runtime``) and on tinygrad with and without tinygrad's own
``BEAM`` kernel search, checks both against onnx's reference evaluator, and
records the winner on the node.

This is the native counterpart of the browser-only comparison
``scripts/convertmodel/test/webgpu_kernel_tuning_vs_webnn.test.mjs`` makes
(tinygrad's best tuned WebGPU Conv2D vs. onnxruntime-web's WebNN EP). The
browser flow can't use tinygrad's ``BEAM=N`` because its search compiles and
times each candidate on ``Device[...]`` locally, which is why
``onnxsim.webgpu_kernel_tuning`` only *generates* candidates and leaves
timing to ``webgpu_kernel_tuner.mjs``. Here both runtimes are native
libraries in the same process, so tinygrad's own search can run as-is on any
tinygrad device (``CPU``, ``METAL``, ``CUDA``, ``WEBGPU`` through Dawn, ...)
and be compared against WebNN on the same machine.

Each node is timed in isolation, as a one-node model
(:func:`extract_node_model`), not in the context of the whole graph: that
matches how onnxsim already reasons about per-node placement
(``onnxsim._ep_fragmentation``, ``webgpu_kernel_tuner.mjs``'s
``tuneNodeKernel``), but it ignores fusion across node boundaries, which
tinygrad in particular relies on. :func:`benchmark_model` times a whole
model the same way for that comparison.

The result is advisory. :func:`tune_node` writes a JSON summary to the
node's ``metadata_props`` under :data:`TUNING_METADATA_KEY` (read it back
with :func:`read_tuning_result`); nothing in onnxsim or onnxruntime acts on
it automatically.

Both backends are optional: without ``pywebnn`` (or when
:func:`onnxsim.rustnn_runtime.probe_rustnn` fails) the WebNN side is
recorded as skipped with the reason, and without ``tinygrad`` the tinygrad
side is. Verified against tinygrad 0.14.0 and pywebnn 0.5.12.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import onnx

from onnxsim.rustnn_runtime import RustnnSession, WebnnLoweringError, probe_rustnn

__all__ = [
    "TUNING_METADATA_KEY",
    "BackendTiming",
    "NodeTuningResult",
    "extract_node_model",
    "random_feeds",
    "time_tinygrad",
    "time_webnn",
    "tune_node",
    "tune_model",
    "benchmark_model",
    "read_tuning_result",
]

#: ``NodeProto.metadata_props`` key :func:`tune_node` writes its result to.
TUNING_METADATA_KEY = "onnxsim.webnn_tinygrad_tuning"

#: Op types :func:`tune_model` looks at by default -- the compute-heavy ones
#: where a tuned kernel can plausibly beat the platform library.
DEFAULT_TUNED_OP_TYPES = ("Conv", "ConvTranspose", "MatMul", "Gemm")


@dataclass(frozen=True)
class BackendTiming:
    """One backend's latency for one node (or model).

    :param backend: ``"webnn"`` or ``"tinygrad"``.
    :param config: what ran: ``"<device_type>/<rustnn backend>"`` for WebNN,
            ``"<device> BEAM=<n>"`` for tinygrad.
    :param median_ms: median wall-clock time of one run, host to host.
    :param max_abs_error: max absolute difference from onnx's reference
            evaluator over all outputs (``None`` when no reference ran).
    :param error: why this backend didn't produce a timing (then the timing
            fields are ``nan``).
    """

    backend: str
    config: str
    median_ms: float = float("nan")
    min_ms: float = float("nan")
    runs: int = 0
    max_abs_error: Optional[float] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class NodeTuningResult:
    """What :func:`tune_node` measured for one node.

    ``winner`` is the fastest :class:`BackendTiming` (among ``timings``) that
    ran and stayed within tolerance of the reference, or ``None`` when none
    did.
    """

    node_name: str
    op_type: str
    timings: List[BackendTiming] = field(default_factory=list)
    winner: Optional[BackendTiming] = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "node_name": self.node_name,
                "op_type": self.op_type,
                "timings": [asdict(t) for t in self.timings],
                "winner": asdict(self.winner) if self.winner else None,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> "NodeTuningResult":
        d = json.loads(text)
        return cls(
            node_name=d["node_name"],
            op_type=d["op_type"],
            timings=[BackendTiming(**t) for t in d["timings"]],
            winner=BackendTiming(**d["winner"]) if d["winner"] else None,
        )


def _find_node(graph: onnx.GraphProto, node_name: str) -> onnx.NodeProto:
    if not node_name:
        raise ValueError("node_name must be a non-empty NodeProto.name")
    for node in graph.node:
        if node.name == node_name:
            return node
    raise ValueError(f"no node named {node_name!r} in the graph")


def extract_node_model(model: onnx.ModelProto, node_name: str) -> onnx.ModelProto:
    """A standalone model holding just ``node_name``: its constant inputs
    (initializers and ``Constant`` outputs) become initializers, every other
    input a graph input, with shapes/dtypes from ONNX shape inference.

    :raises ValueError: no such node, or a non-constant input/output whose
            shape inference can't give a fully static shape (WebNN and a
            fixed tinygrad kernel both need one).
    """
    node = _find_node(model.graph, node_name)
    inferred = onnx.shape_inference.infer_shapes(model)
    graph = inferred.graph
    value_infos = {
        vi.name: vi
        for vi in list(graph.input) + list(graph.value_info) + list(graph.output)
    }
    consts = {init.name: init for init in graph.initializer}
    for n in graph.node:
        if (
            n.op_type == "Constant"
            and n.output
            and n.attribute
            and n.attribute[0].name == "value"
        ):
            t = onnx.TensorProto()
            t.CopyFrom(n.attribute[0].t)
            t.name = n.output[0]
            consts[n.output[0]] = t

    def static_vi(name: str) -> onnx.ValueInfoProto:
        vi = value_infos.get(name)
        if (
            vi is None
            or not vi.type.tensor_type.HasField("shape")
            or any(not d.HasField("dim_value") for d in vi.type.tensor_type.shape.dim)
        ):
            raise ValueError(
                f"node {node_name!r}: {name!r} has no static shape after shape "
                "inference; simplify the model (with fixed input shapes) first"
            )
        return vi

    inputs, initializers = [], []
    for name in dict.fromkeys(i for i in node.input if i):
        if name in consts:
            initializers.append(consts[name])
        else:
            inputs.append(static_vi(name))
    outputs = [static_vi(o) for o in node.output if o]
    sub_node = onnx.NodeProto()
    sub_node.CopyFrom(node)
    del sub_node.metadata_props[:]
    sub_graph = onnx.helper.make_graph(
        [sub_node], f"{node_name}_isolated", inputs, outputs, initializers
    )
    sub = onnx.helper.make_model(sub_graph, opset_imports=list(model.opset_import))
    sub.ir_version = model.ir_version
    return sub


def random_feeds(model: onnx.ModelProto, seed: int = 0) -> Dict[str, np.ndarray]:
    """Deterministic random feeds for every non-initializer graph input
    (floats ~ N(0, 1); integers in ``[0, 4)``, which keeps index-like inputs
    in range for small shapes)."""
    rng = np.random.default_rng(seed)
    inits = {i.name for i in model.graph.initializer}
    feeds = {}
    for vi in model.graph.input:
        if vi.name in inits:
            continue
        tt = vi.type.tensor_type
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
        shape = [d.dim_value for d in tt.shape.dim]
        if np.issubdtype(dtype, np.floating):
            feeds[vi.name] = rng.standard_normal(shape).astype(dtype)
        elif dtype == np.bool_:
            feeds[vi.name] = rng.integers(0, 2, shape).astype(dtype)
        else:
            feeds[vi.name] = rng.integers(0, 4, shape).astype(dtype)
    return feeds


def _reference(
    model: onnx.ModelProto, feeds: Mapping[str, np.ndarray]
) -> Optional[List[np.ndarray]]:
    try:
        from onnx.reference import ReferenceEvaluator

        return list(ReferenceEvaluator(model).run(None, dict(feeds)))
    except Exception:  # the reference evaluator doesn't cover every op/dtype
        return None


def _max_abs_error(
    got: Sequence[np.ndarray], ref: Optional[Sequence[np.ndarray]]
) -> Optional[float]:
    if ref is None:
        return None
    worst = 0.0
    for g, r in zip(got, ref):
        g, r = np.asarray(g), np.asarray(r)
        if g.shape != r.shape:
            return float("inf")
        if g.size:
            worst = max(
                worst,
                float(np.max(np.abs(g.astype(np.float64) - r.astype(np.float64)))),
            )
    return worst


def _median_min(samples: List[float]) -> Tuple[float, float]:
    return statistics.median(samples), min(samples)


def time_webnn(
    model: onnx.ModelProto,
    feeds: Mapping[str, np.ndarray],
    *,
    device_type: str = "auto",
    backend: str = "auto",
    warmup: int = 2,
    runs: int = 10,
) -> Tuple[BackendTiming, Optional[List[np.ndarray]]]:
    """Lowers ``model`` onto rustnn (see ``onnxsim.rustnn_runtime``) and times
    it. Returns the timing and outputs (in ``model.graph.output`` order), or
    an errored :class:`BackendTiming` and ``None`` when rustnn is unavailable
    or can't lower/run the model."""
    config = f"{device_type}/{backend}"
    ok, reason = probe_rustnn(device_type, backend)
    if not ok:
        return BackendTiming(
            "webnn", config, error=f"rustnn unavailable: {reason}"
        ), None
    try:
        session = RustnnSession(model, device_type=device_type, backend=backend)
        timing, out = session.benchmark(feeds, warmup=warmup, runs=runs)
    except WebnnLoweringError as e:
        return BackendTiming(
            "webnn", config, error=f"not lowerable to WebNN: {e}"
        ), None
    except Exception as e:
        return BackendTiming("webnn", config, error=f"{type(e).__name__}: {e}"), None
    return (
        BackendTiming("webnn", config, timing.median_ms, timing.min_ms, timing.runs),
        [out[name] for name in session.output_names],
    )


def time_tinygrad(
    model: onnx.ModelProto,
    feeds: Mapping[str, np.ndarray],
    *,
    device: Optional[str] = None,
    beam: int = 0,
    warmup: int = 2,
    runs: int = 10,
) -> Tuple[BackendTiming, Optional[List[np.ndarray]]]:
    """Runs ``model`` through tinygrad's ``OnnxRunner`` under ``TinyJit`` on
    ``device`` (tinygrad's default when ``None``) with ``BEAM=beam`` -- so
    ``beam > 0`` runs tinygrad's own kernel search, compiling and timing
    candidates on that device -- and times the jitted call host to host
    (feeds copied in, outputs read back), the same span
    :func:`time_webnn`'s ``MLContext.compute`` covers.

    ``IGNORE_BEAM_CACHE`` is set so an earlier run's cached search result
    for the same kernel doesn't stand in for this one's.
    """
    try:
        from tinygrad import Device, Tensor, TinyJit
        from tinygrad.helpers import Context
        from tinygrad.nn.onnx import OnnxRunner
    except ImportError as e:
        return BackendTiming(
            "tinygrad", f"{device} BEAM={beam}", error=f"tinygrad unavailable: {e}"
        ), None

    dev = device or Device.DEFAULT
    config = f"{dev} BEAM={beam}"
    fd, path = tempfile.mkstemp(suffix=".onnx")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(model.SerializeToString())
        with Context(BEAM=beam, IGNORE_BEAM_CACHE=int(beam > 0)):
            runner = OnnxRunner(path)
            if device is not None:
                runner = runner.to(device)
            names = [o.name for o in model.graph.output]

            @TinyJit
            def step(**tensors):
                out = runner(tensors)
                return [out[n].realize() for n in names]

            def call():
                # Host to host, like MLContext.compute on the WebNN side:
                # copy feeds in, run, and read every output back.
                Device[dev].synchronize()
                start = time.perf_counter()
                tensors = {k: Tensor(v, device=dev) for k, v in feeds.items()}
                out = [t.numpy() for t in step(**tensors)]
                return (time.perf_counter() - start) * 1e3, out

            # TinyJit needs two calls to capture; BEAM search happens then too.
            for _ in range(max(2, warmup)):
                call()
            samples, outputs = [], None
            for _ in range(max(1, runs)):
                ms, outputs = call()
                samples.append(ms)
    except Exception as e:
        return BackendTiming("tinygrad", config, error=f"{type(e).__name__}: {e}"), None
    finally:
        os.unlink(path)
    median, fastest = _median_min(samples)
    return BackendTiming("tinygrad", config, median, fastest, len(samples)), outputs


def _set_node_metadata(node: onnx.NodeProto, key: str, value: str) -> None:
    for entry in node.metadata_props:
        if entry.key == key:
            entry.value = value
            return
    entry = node.metadata_props.add()
    entry.key = key
    entry.value = value


def _measure(
    model: onnx.ModelProto,
    feeds: Mapping[str, np.ndarray],
    *,
    webnn_device_types: Sequence[str],
    webnn_backend: str,
    tinygrad_device: Optional[str],
    beams: Sequence[int],
    warmup: int,
    runs: int,
    atol: float,
    rtol: float,
) -> Tuple[List[BackendTiming], Optional[BackendTiming]]:
    ref = _reference(model, feeds)
    timings = []
    for device_type in webnn_device_types:
        t, out = time_webnn(
            model,
            feeds,
            device_type=device_type,
            backend=webnn_backend,
            warmup=warmup,
            runs=runs,
        )
        timings.append(_with_error(t, out, ref))
    for beam in beams:
        t, out = time_tinygrad(
            model, feeds, device=tinygrad_device, beam=beam, warmup=warmup, runs=runs
        )
        timings.append(_with_error(t, out, ref))

    def within_tolerance(t: BackendTiming) -> bool:
        if t.max_abs_error is None:
            return True  # no reference to judge against
        scale = (
            max((float(np.max(np.abs(r))) for r in ref if r.size), default=0.0)
            if ref
            else 0.0
        )
        return t.max_abs_error <= atol + rtol * scale

    candidates = [t for t in timings if t.ok and within_tolerance(t)]
    winner = min(candidates, key=lambda t: t.median_ms) if candidates else None
    return timings, winner


def _with_error(t: BackendTiming, out, ref) -> BackendTiming:
    if not t.ok or out is None:
        return t
    return BackendTiming(
        t.backend, t.config, t.median_ms, t.min_ms, t.runs, _max_abs_error(out, ref)
    )


def tune_node(
    model: onnx.ModelProto,
    node_name: str,
    *,
    feeds: Optional[Mapping[str, np.ndarray]] = None,
    webnn_device_types: Sequence[str] = ("auto",),
    webnn_backend: str = "auto",
    tinygrad_device: Optional[str] = None,
    beams: Sequence[int] = (0, 2),
    warmup: int = 2,
    runs: int = 10,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    write_back: bool = True,
) -> NodeTuningResult:
    """Times node ``node_name`` in isolation (:func:`extract_node_model`) on
    rustnn for each of ``webnn_device_types`` and on tinygrad for each BEAM
    width in ``beams`` (``0`` = tinygrad's untuned kernels), validates every
    output against onnx's reference evaluator, and picks the fastest one
    within ``atol + rtol * max|reference|``.

    :param feeds: inputs for the isolated node, keyed by the node's own
            input names; random (:func:`random_feeds`) when omitted.
    :param write_back: also store the result as JSON on the node's
            ``metadata_props[TUNING_METADATA_KEY]`` (``model`` is mutated).
    """
    node = _find_node(model.graph, node_name)
    sub = extract_node_model(model, node_name)
    feeds = dict(feeds) if feeds is not None else random_feeds(sub)
    timings, winner = _measure(
        sub,
        feeds,
        webnn_device_types=webnn_device_types,
        webnn_backend=webnn_backend,
        tinygrad_device=tinygrad_device,
        beams=beams,
        warmup=warmup,
        runs=runs,
        atol=atol,
        rtol=rtol,
    )
    result = NodeTuningResult(node_name, node.op_type, timings, winner)
    if write_back:
        _set_node_metadata(node, TUNING_METADATA_KEY, result.to_json())
    return result


def tune_model(
    model: onnx.ModelProto,
    *,
    node_names: Optional[Iterable[str]] = None,
    op_types: Sequence[str] = DEFAULT_TUNED_OP_TYPES,
    **kwargs,
) -> List[NodeTuningResult]:
    """:func:`tune_node` over ``node_names``, or else every named node whose
    op type is in ``op_types``. ``kwargs`` go to :func:`tune_node`
    (except ``feeds``, which is per-node: random feeds are used)."""
    if "feeds" in kwargs:
        raise TypeError(
            "tune_model uses random per-node feeds; call tune_node to pass feeds"
        )
    if node_names is None:
        node_names = [
            n.name for n in model.graph.node if n.name and n.op_type in op_types
        ]
    return [tune_node(model, name, **kwargs) for name in node_names]


def benchmark_model(
    model: onnx.ModelProto,
    *,
    feeds: Optional[Mapping[str, np.ndarray]] = None,
    webnn_device_types: Sequence[str] = ("auto",),
    webnn_backend: str = "auto",
    tinygrad_device: Optional[str] = None,
    beams: Sequence[int] = (0, 2),
    warmup: int = 2,
    runs: int = 10,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> NodeTuningResult:
    """The same comparison as :func:`tune_node`, for the whole model at once
    (so tinygrad can fuse across nodes). Nothing is written back; the
    result's ``node_name`` is ``""`` and ``op_type`` is ``"<model>"``."""
    feeds = dict(feeds) if feeds is not None else random_feeds(model)
    timings, winner = _measure(
        model,
        feeds,
        webnn_device_types=webnn_device_types,
        webnn_backend=webnn_backend,
        tinygrad_device=tinygrad_device,
        beams=beams,
        warmup=warmup,
        runs=runs,
        atol=atol,
        rtol=rtol,
    )
    return NodeTuningResult("", "<model>", timings, winner)


def read_tuning_result(
    model: Union[onnx.ModelProto, str], node_name: str
) -> Optional[NodeTuningResult]:
    """The :class:`NodeTuningResult` :func:`tune_node` stored on
    ``node_name``, or ``None`` if it has none."""
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    node = _find_node(model.graph, node_name)
    for entry in node.metadata_props:
        if entry.key == TUNING_METADATA_KEY:
            return NodeTuningResult.from_json(entry.value)
    return None
