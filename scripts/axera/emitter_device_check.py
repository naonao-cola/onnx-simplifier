"""Device check for the record-level emitters: do models emitted at a
calibration Pulsar2 never built compute what that calibration means?

Every emitter here was validated against native Pulsar2 builds record for
record (``reshape_record_emit``, ``misc_op_record_emit``,
``matmul_record_emit``, the Relu zero-point retarget in
``elementwise_scale_emit``). This module asks the hardware: each emitted
model runs on the AX650 and is compared with a quantized reference, i.e.
the float graph with a QuantizeLinear/DequantizeLinear pair on every
quantized (consumer, tensor) at the target calibration, run in onnxruntime
(``qdq_reference``). The error is measured in output LSBs (units of the
output scale).

The native template runs the same way first (its own calibration, its own
reference): that is the control, and its error is the baseline an emitted
model is held to. A health run of the native template after each emitted
model must reproduce the control bit for bit, so a model that wedges the
NPU cannot pass silently (``docs/axera-emitter-device-check.md``).

Targets are the ResNet18 step's predicted calibration
(``step_calibration.py``) wherever the emitter serves the step node there,
and a perturbed calibration where it refuses (Conv and the dense-head Gemm).

Usage::

    emitter_device_check.py list
    emitter_device_check.py device OUT.json [CASE ...]   # needs axcl-vm + the card
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, parser

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import elementwise_scale_emit as ew  # noqa: E402
import matmul_record_emit as mre  # noqa: E402
import misc_op_record_emit as misc  # noqa: E402
import reshape_record_emit as rre  # noqa: E402

FIX = os.path.join(_HERE, "fixtures")
OWN = os.path.join(FIX, "emitter_device_check")
STEP_CALIB = os.path.join(FIX, "step_calibration", "resnet18_step_calibration.json.gz")
STEP_OPS = os.path.join(FIX, "tinygrad_ax_backend", "resnet18_step_ops.json.gz")
_LOCK = "/tmp/axcl-device.lock"

# (scale, zero_point, qmin, qmax)
QParam = tuple[float, int, int, int]


@dataclass
class QConfig:
    """Where the reference quantizes: each node input ``(node, tensor)`` and
    each graph output ``tensor``, as Pulsar2's ``tensor_configs`` do."""

    inputs: dict[tuple[str, str], QParam] = field(default_factory=dict)
    outputs: dict[str, QParam] = field(default_factory=dict)


# --------------------------------------------------------------------------
# quantized reference


def qdq_model(model: onnx.ModelProto, cfg: QConfig) -> onnx.ModelProto:
    """``model`` with a Q/DQ pair in front of every configured node input and
    behind every configured graph output."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    new_nodes: list[onnx.NodeProto] = []
    inits: list[TensorProto] = []

    def qdq(src: str, tag: str, q: QParam) -> tuple[str, list[onnx.NodeProto]]:
        s, zp, qmin, _ = q
        dt = np.uint8 if qmin >= 0 else np.int8
        sn, zn = f"{tag}__s", f"{tag}__zp"
        inits.append(numpy_helper.from_array(np.array(s, np.float32), sn))
        inits.append(numpy_helper.from_array(np.array(zp, dt), zn))
        qn, dn = f"{tag}__q", f"{tag}__dq"
        return dn, [
            helper.make_node("QuantizeLinear", [src, sn, zn], [qn]),
            helper.make_node("DequantizeLinear", [qn, sn, zn], [dn]),
        ]

    for i, node in enumerate(g.node):
        for j, t in enumerate(node.input):
            q = cfg.inputs.get((node.name, t))
            if q is None or not t:
                continue
            name, nodes = qdq(t, f"in{i}_{j}", q)
            new_nodes += nodes
            node.input[j] = name
        new_nodes.append(node)
    for k, out in enumerate(g.output):
        q = cfg.outputs.get(out.name)
        if q is None:
            continue
        # rename the producer's output so the graph output keeps its name
        raw = f"{out.name}__raw"
        for node in new_nodes:
            for j, t in enumerate(node.output):
                if t == out.name:
                    node.output[j] = raw
        name, nodes = qdq(raw, f"out{k}", q)
        nodes[-1].output[0] = out.name
        new_nodes += nodes
    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.extend(inits)
    return m


def qdq_reference(model: onnx.ModelProto, cfg: QConfig, feeds: Mapping) -> list:
    import onnxruntime as ort

    m = qdq_model(model, cfg)
    del m.opset_import[:]
    m.opset_import.extend([helper.make_opsetid("", 17)])
    m.ir_version = 8
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        m.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    names = [o.name for o in m.graph.output]
    return sess.run(names, {k: np.asarray(v) for k, v in feeds.items()})


def config_from_quant(quant: Mapping) -> QConfig:
    """Pulsar2 ``quant_axmodel.json`` -> ``QConfig`` (inputs keyed by the
    consuming op's name; graph outputs filled by ``_outputs_from``)."""
    cfg = QConfig()
    values = quant["values"]
    for op, per in quant["tensor_configs"].items():
        for t, c in per.items():
            v = values.get(str(c.get("dominator", c.get("hash"))))
            if not isinstance(v, dict) or not isinstance(v.get("scale"), list):
                continue
            q = (
                float(v["scale"][0]),
                int(round(v["zero_point"][0])),
                int(c["quant_min"]),
                int(c["quant_max"]),
            )
            cfg.inputs[(op, t)] = q
    return cfg


def _outputs_from(cfg: QConfig, model: onnx.ModelProto) -> QConfig:
    """Graph outputs quantize with their producer's config for that tensor."""
    for out in model.graph.output:
        prod = next(n for n in model.graph.node if out.name in n.output)
        q = cfg.inputs.pop((prod.name, out.name), None)
        if q is not None:
            cfg.outputs[out.name] = q
    return cfg


def retarget_config(cfg: QConfig, new: Mapping[str, tuple[float, int]]) -> QConfig:
    """``cfg`` with every configured tensor named in ``new`` moved to its new
    ``(scale, zero_point)`` (keeping the quantization range)."""
    out = QConfig()
    for (op, t), q in cfg.inputs.items():
        out.inputs[(op, t)] = (*new[t], q[2], q[3]) if t in new else q
    for t, q in cfg.outputs.items():
        out.outputs[t] = (*new[t], q[2], q[3]) if t in new else q
    return out


def simple_config(
    model: onnx.ModelProto, x: tuple[float, int], y: tuple[float, int] | None
) -> QConfig:
    """Every activation input quantized at ``x``, the graph output at ``y``
    (``None``: not quantized, e.g. Greater -> Cast)."""
    cfg = QConfig()
    consts = {i.name for i in model.graph.initializer}
    if y is None:
        return cfg
    for n in model.graph.node:
        for t in n.input:
            if t and t not in consts:
                cfg.inputs[(n.name, t)] = (x[0], int(x[1]), 0, 255)
    for o in model.graph.output:
        cfg.outputs[o.name] = (y[0], int(y[1]), 0, 255)
    return cfg


# --------------------------------------------------------------------------
# device


def device_available() -> bool:
    """The card is reachable where ``run`` sends models: the ``AXCL_LXD_VM``
    guest (default ``axcl-vm``, as ``teng2_fault_injection`` routes it), or
    this host when that variable is set empty. Needs both the runner and the
    driver's ``/dev/axcl_host`` node (an installed runner without the driver
    fails with "Init failed")."""
    import subprocess

    cmd = ["test", "-x", "/usr/bin/axcl/axcl_run_model", "-a", "-e", "/dev/axcl_host"]
    vm = os.environ.get("AXCL_LXD_VM", "axcl-vm")
    if vm:
        cmd = ["lxc", "exec", vm, "--", *cmd]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def run(model: onnx.ModelProto, feeds: Mapping) -> dict:
    """One device run, holding the shared device lock for this run only."""
    from teng2_fault_injection import _run_on_device_with_inputs

    with tempfile.TemporaryDirectory(dir=os.environ.get("EMITTER_CHECK_TMP")) as td:
        path = os.path.join(td, "m.axmodel")
        onnx.save(model, path)
        import fcntl  # POSIX-only; imported here so this module (and its tests) import on Windows

        fd = os.open(_LOCK, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            return _run_on_device_with_inputs(
                path,
                {k: np.ascontiguousarray(v).tobytes() for k, v in feeds.items()},
                timeout=600,
            )
        finally:
            os.close(fd)


def lsb_error(dev: bytes, ref: np.ndarray, scale: float | None) -> dict:
    d = np.frombuffer(dev, np.float32).reshape(-1)
    r = np.asarray(ref, np.float32).reshape(-1)
    if d.size != r.size:
        return {"note": f"size {d.size} != reference {r.size}"}
    e = np.abs(d.astype(np.float64) - r.astype(np.float64))
    if scale is None:  # unquantized output (Greater -> Cast): exact match
        return {"max_abs": float(e.max()), "mismatches": int((e > 0).sum())}
    lsb = e / scale
    return {
        "max_lsb": float(lsb.max()),
        "p999_lsb": float(np.quantile(lsb, 0.999)),
        "frac_over_1lsb": float((lsb > 1.0 + 1e-3).mean()),
        "n": int(r.size),
    }


# --------------------------------------------------------------------------
# cases


@dataclass
class Case:
    name: str
    emitter: str
    target: str  # where the target calibration comes from
    native: onnx.ModelProto
    float_model: onnx.ModelProto
    native_cfg: QConfig
    emitted: Callable[[], onnx.ModelProto]
    target_cfg: QConfig
    note: str = ""


def _load(path: str) -> onnx.ModelProto:
    data = gzip.open(path).read() if path.endswith(".gz") else open(path, "rb").read()
    return onnx.load_model_from_string(data)


def _io(model: onnx.ModelProto):
    def dims(v):
        return [d.dim_value for d in v.type.tensor_type.shape.dim]

    ins = [(v.name, dims(v)) for v in model.graph.input]
    outs = [(v.name, dims(v)) for v in model.graph.output]
    return ins, outs


def _shape(dims) -> str:
    return ",".join(map(str, dims))


def _calib() -> dict:
    with gzip.open(STEP_CALIB, "rt") as f:
        return json.load(f)


def _step_records() -> list[dict]:
    with gzip.open(STEP_OPS, "rt") as f:
        return json.load(f)


def _q(calib: Mapping, name: str) -> tuple[float, int]:
    t = calib["tensors"][name]
    return float(t["scale"]), int(t["zero_point"])


def relu_case(shape, step_node: str) -> Case:
    calib = _calib()
    rec = next(r for r in _step_records() if r["name"] == step_node)
    s, zp = _q(calib, rec["inputs"][0])
    native, meta = ew.load_template("Relu", shape, {"x": 128, "y": 128})
    (xn, xd), (yn, _) = _io(native)[0][0], _io(native)[1][0]
    fm = parser.parse_model(
        f'<ir_version: 8, opset_import: ["": 17]> g (float[{_shape(xd)}] {xn}) '
        f"=> (float[{_shape(xd)}] {yn}) {{ {yn} = Relu ({xn}) }}"
    )
    fm.graph.node[0].name = "relu"
    s0 = meta["scales"]["x"]

    def emitted():
        m = onnx.ModelProto()
        m.CopyFrom(native)
        init = ew._mcode_initializer(m)
        init.raw_data = ew.retarget_relu_records(bytes(init.raw_data), s, zp)
        return m

    return Case(
        f"relu_{'x'.join(map(str, shape))}",
        "elementwise_scale_emit.retarget_relu_records",
        f"step {step_node}: s={s:.6g} zp={zp}",
        native,
        fm,
        simple_config(fm, (s0, 128), (s0, 128)),
        emitted,
        simple_config(fm, (s, zp), (s, zp)),
    )


def reshape_case(key: str, target: tuple[float, int], label: str, zp0=False) -> Case:
    man = rre.step_manifest()
    # a nonzero zero point is a signed input: the Reshape -> Identity template
    # (the Relu form clips it); zero point 0 keeps its Relu-form template, where
    # the input is nonnegative. The reference is the step's plain Reshape
    # either way, so a template that clips cannot pass.
    entry = (man["zero_point_0_templates"] if zp0 else man["identity_templates"])[key]
    native = rre.load_axmodel(os.path.join(rre.STEP_TEMPLATE_DIR, entry["axmodel"]))
    (xn, xd), (yn, yd) = _io(native)[0][0], _io(native)[1][0]
    fm = parser.parse_model(
        f'<ir_version: 8, opset_import: ["": 17]> g (float[{_shape(xd)}] {xn}) '
        f"=> (float[{_shape(yd)}] {yn}) "
        f"<int64[{len(yd)}] target = {{{_shape(yd)}}}> "
        f"{{ {yn} = Reshape ({xn}, target) }}"
    )
    fm.graph.node[0].name = "reshape"
    s0, z0 = entry["scale"], entry["zero_point"]
    s, zp = target
    in_shape, out_shape = key.split("->")

    def emitted():
        return rre.emit_step_reshape(
            [int(d) for d in in_shape.split("x")],
            [int(d) for d in out_shape.split("x")],
            s,
            zp,
        )

    return Case(
        f"reshape_{key.replace('->', '_to_')}{'_zp0' if zp0 else ''}",
        "reshape_record_emit.emit_step_reshape",
        label,
        native,
        fm,
        simple_config(fm, (s0, z0), (s0, z0)),
        emitted,
        simple_config(fm, (s, zp), (s, zp)),
    )


def _misc_float(meta: Mapping, native: onnx.ModelProto, key: str) -> onnx.ModelProto:
    (xn, xd), (yn, yd) = _io(native)[0][0], _io(native)[1][0]
    op = meta["op"]
    head = (
        f'<ir_version: 8, opset_import: ["": 17]> g (float[{_shape(xd)}] {xn}) '
        f"=> (float[{_shape(yd)}] {yn}) "
    )
    if op == "Softmax":
        body = f"{{ {yn} = Softmax <axis = -1> ({xn}) }}"
    elif op in ("Log", "Neg"):
        body = f"{{ {yn} = {op} ({xn}) }}"
    elif op == "MaxPool":
        body = (
            f"{{ {yn} = MaxPool <kernel_shape = [3, 3], pads = [1, 1, 1, 1], "
            f"strides = [2, 2]> ({xn}) }}"
        )
    elif op in ("ReduceSum", "ReduceMean"):
        part = key.split(":")
        axes = [int(a) for a in part[2][4:].split(",")]
        kd = int(part[3][1:])
        if op == "ReduceSum":
            head += f"<int64[{len(axes)}] axes = {{{_shape(axes)}}}> "
            body = f"{{ {yn} = ReduceSum <keepdims = {kd}> ({xn}, axes) }}"
        else:
            body = f"{{ {yn} = ReduceMean <keepdims = {kd}, axes = {axes}> ({xn}) }}"
    elif op in ("GreaterCast", "LessCast"):
        cmp = op[: -len("Cast")]
        head += "<float zero = {0.0}> "
        body = f"{{ b = {cmp} ({xn}, zero) {yn} = Cast <to = 1> (b) }}"
    else:
        raise ValueError(f"no float graph for {op}")
    fm = parser.parse_model(head + body)
    for i, n in enumerate(fm.graph.node):
        n.name = f"n{i}"
    return fm


def misc_case(
    key: str, scales: Mapping | None, zps: Mapping | None, label: str, suffix=""
) -> Case:
    native, meta = misc.load_template(key)
    fm = _misc_float(meta, native, key)
    if meta["op"] in misc.CALIBRATION_FREE:
        return Case(
            f"misc_{key}",
            "misc_op_record_emit.emit_model",
            label,
            native,
            fm,
            QConfig(),
            lambda: misc.emit_model(key),
            QConfig(),
            note="calibration-free: the emitted model is the template",
        )
    s0, z0 = meta["scales"], meta["zero_points"]
    new_key = key
    if meta["op"] == "Neg" and scales:
        new_key = key if misc.neg_program(scales["x"]) == meta["program"] else None
        if new_key is None:
            new_key = meta["programs"][misc.neg_program(scales["x"])]
            native, meta = misc.load_template(new_key)
            s0, z0 = meta["scales"], meta["zero_points"]
    return Case(
        f"misc_{new_key}{suffix}",
        "misc_op_record_emit.emit_model",
        label,
        native,
        fm,
        simple_config(fm, (s0["x"], z0["x"]), (s0["y"], z0["y"])),
        lambda: misc.emit_model(new_key, scales, zps),
        simple_config(fm, (scales["x"], zps["x"]), (scales["y"], zps["y"])),
    )


def _misc_step(key: str) -> tuple[dict, dict, str]:
    """The step node a misc template serves, at the predicted calibration."""
    calib = _calib()
    meta = misc.load_index()[key]
    keys = {key, meta.get("variant_of")} - {None}
    base = next(
        r
        for r in _step_records()
        if (r.get("attrs", {}).get("misc_key") in keys)
        or misc.equivalent_key(r.get("attrs", {}).get("misc_key") or "") in keys
    )
    sx, zx = _q(calib, base["inputs"][0])
    sy, zy = _q(calib, base["outputs"][0])
    return {"x": sx, "y": sy}, {"x": zx, "y": zy}, f"step {base['name']}"


def matmul_case(template: str, target: str = "step", seed: int = 0) -> Case:
    """``target='step'``: the template's own step node at the predicted
    calibration; ``'perturb'``: every scale times a random factor in
    [0.7, 1.4] (zero points kept), for templates the step calibration refuses."""
    man = mre.step_manifest()
    t = man["templates"][template]
    native = mre.load_model(os.path.join(mre.STEP_TEMPLATE_DIR, t["axmodel"]))
    old = mre.load_scales(os.path.join(mre.STEP_TEMPLATE_DIR, t["quant"]))
    with gzip.open(os.path.join(mre.STEP_TEMPLATE_DIR, t["quant"]), "rt") as f:
        quant = json.load(f)
    fm = _load(os.path.join(OWN, f"{template}.onnx.gz"))
    ncfg = _outputs_from(config_from_quant(quant), fm)
    if target == "step":
        calib = _calib()
        node = next(n for n, e in man["nodes"].items() if e["template"] == template)
        entry = mre.step_template(node, man)
        real = {}
        for step_name, name in entry["names"].items():
            q = calib["tensors"][step_name]
            if "consumer_int8_scale" in q and name in old and old[name][1] == 0:
                real[step_name] = (q["consumer_int8_scale"], 0.0)
            else:
                real[step_name] = (q["scale"], float(q["zero_point"]))
        new = mre.step_node_scales(entry, old, real)
        label = f"step {node} (predicted calibration)"
    else:
        # One factor per group of template calibrations whose scales are a
        # power of two apart (within float32 rounding): passive ops share a
        # calibration, and Pulsar2 sets a 3x3 Conv's Concat inputs to exactly
        # 2x (uint8 -> int8) or 1x their source's scale. A calibration that
        # breaks those ratios is not one Pulsar2 produces
        # (docs/axera-emitter-device-check.md, "Conv").
        rng = np.random.default_rng(seed)
        const = set(t.get("constants", []))
        roots: list[float] = []
        factor: dict[float, float] = {}
        for sc in sorted({v[0] for v in old.values()}):
            root = next(
                (
                    r
                    for r in roots
                    if abs(np.log2(sc / r) - round(np.log2(sc / r))) < 1e-6
                ),
                None,
            )
            if root is None:
                roots.append(sc)
                factor[sc] = rng.uniform(0.7, 1.4)
            else:
                factor[sc] = factor[root]
        if target == "perturb_free":  # every calibration on its own
            factor = {sc: rng.uniform(0.7, 1.4) for sc in factor}
        new = {
            k: v if k in const else (float(np.float32(v[0] * factor[v[0]])), v[1])
            for k, v in old.items()
        }
        label = f"perturbed template calibration (seed {seed})"
    tcfg = retarget_config(ncfg, {k: (v[0], int(round(v[1]))) for k, v in new.items()})

    def emitted():
        return mre.recalibrate(native, old, new)[0]

    return Case(
        f"matmul_{template}_{target}",
        "matmul_record_emit.recalibrate",
        label,
        native,
        fm,
        ncfg,
        emitted,
        tcfg,
    )


def cases() -> list[Callable[[], Case]]:
    """Representative cases, small first."""
    out: list[Callable[[], Case]] = []
    calib = _calib()
    recs = _step_records()

    # Relu zero-point retarget (#1897), every step Relu shape
    for shape, node in (
        ([16, 512, 7, 7], None),
        ([16, 256, 14, 14], None),
        ([16, 64, 56, 56], None),
    ):
        node = next(
            r["name"] for r in recs if r["op"] == "Relu" and r["shapes"][0] == shape
        )
        out.append(lambda shape=shape, node=node: relu_case(shape, node))

    # Reshape: untiled small, tiled, zero-point 0
    def reshape_step(key):
        a, b = key.split("->")
        rec = next(
            r
            for r in recs
            if r["op"] == "Reshape"
            and "x".join(map(str, r["shapes"][0])) == a
            and "x".join(map(str, r.get("attrs", {}).get("out", []))) == b
        )
        return _q(calib, rec["inputs"][0]), f"step {rec['name']}"

    for key in (
        "64x64x3x3->1x64x64x9",
        "16x512x7x7->16x1x512x49",
        "16x64x56x56->16x1x64x3136",
    ):
        out.append(lambda key=key: reshape_case(key, *reshape_step(key)))
    out.append(
        lambda: reshape_case(
            "1024x28224->1024x9x3136",
            *reshape_step("1024x28224->1024x9x3136"),
            zp0=True,
        )
    )

    # misc ops at the step's predicted calibration
    for key in (
        "Softmax:16x1000:axis1",
        "Log:16x1000",
        "MaxPool:16x64x112x112:k3x3:s2x2:p1,1,1,1:zp",
        "ReduceSum:16x1000:axes0,1:k1",
        "ReduceSum:16x1x512x49:axes0,3:k0",
        "ReduceSum:16x1x128x64:axes0:k0",
        "ReduceSum:1024x9x3136:axes1:k1:zp0",
        "Neg:1x1",
    ):
        out.append(lambda key=key: misc_case(key, *_misc_step(key)))
    # Neg's small program (the step's Neg is the large one) and a ReduceSum
    # moved to zp_y = 0 (Pulsar2 elides the zero-point writes: records removed)
    out.append(
        lambda: misc_case(
            "Neg:1x1", {"x": 0.012, "y": 0.012}, {"x": 200, "y": 55}, "synthetic s<1/64"
        )
    )
    out.append(
        lambda: misc_case(
            "ReduceSum:16x1x128x64:axes0:k0",
            {"x": 0.031, "y": 0.11},
            {"x": 121, "y": 0},
            "synthetic zp_y=0 (record removal)",
            ":zp_y0",
        )
    )
    # the step's ReduceMean output is int8 (it feeds only MatMuls), which no
    # template serves; move the uint8 template's scales instead
    out.append(
        lambda: misc_case(
            "ReduceMean:16x512x7x7:axes2,3:k1",
            {"x": 0.0213, "y": 0.0041},
            {"x": 0, "y": 0},
            "synthetic scales (zero points kept)",
        )
    )
    out.append(
        lambda: misc_case("GreaterCast:16x512x7x7", None, None, "no calibration")
    )

    # MatMul chains: Gather/Mul chain at the step calibration; fused bias and
    # 3x3 per-tap requant (Conv as MatMul) and the dense-head Gemm perturbed
    out.append(lambda: matmul_case("fc_dW_MatMul_38"))
    out.append(lambda: matmul_case("dX_MatMul_121"))
    out.append(lambda: matmul_case("dX_MatMul_240"))
    out.append(lambda: matmul_case("dX_MatMul_240", "perturb"))
    out.append(lambda: matmul_case("dW_MatMul_142"))
    out.append(lambda: matmul_case("conv_resnetv15_stage3_conv1_fwd"))
    # 3x3 Conv chains whose Concat header half is above 1 (k >= 1) and at
    # most 1 (k = 0), at the step's predicted calibration
    # (docs/axera-conv-concat-shift.md)
    out.append(lambda: matmul_case("conv_resnetv15_stage2_conv1_fwd"))
    out.append(lambda: matmul_case("conv_resnetv15_stage2_conv4_fwd"))
    out.append(lambda: matmul_case("conv_resnetv15_stage4_conv3_fwd"))
    out.append(lambda: matmul_case("fc_fwd_resnetv15_dense0_fwd", "perturb"))
    return out


# --------------------------------------------------------------------------


def feeds_for(case: Case, cfg: QConfig, seed: int = 7) -> tuple[dict, float]:
    """Inputs spanning each input's quantization range, narrowed until at
    most 5% of the reference output saturates (so a Reduce/MatMul output is
    exercised inside its range, not only at the clamp). Returns the feeds
    and the spread factor used."""
    rng = np.random.RandomState(seed)
    ins = _io(case.float_model)[0]
    ranges = {}
    for name, _ in ins:
        q = next((q for (_, t), q in cfg.inputs.items() if t == name), None)
        if q is None:
            ranges[name] = (-1.0, 1.0)
        else:
            s, zp, qmin, qmax = q
            # one step inside the range: Log's input never hits exactly 0
            ranges[name] = ((qmin - zp + 1) * s, (qmax - zp) * s)
    out_q = next(iter(cfg.outputs.values()), None)
    base = {n: rng.uniform(0, 1, d).astype(np.float32) for n, d in ins}
    for spread in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01):
        feeds = {}
        for n, _ in ins:
            lo, hi = ranges[n]
            c = 0.0 if lo < 0 < hi else lo
            a, b = c - (c - lo) * spread, c + (hi - c) * spread
            feeds[n] = (a + base[n] * (b - a)).astype(np.float32)
        if out_q is None:
            return feeds, spread
        ref = qdq_reference(case.float_model, cfg, feeds)[0]
        s, zp, qmin, qmax = out_q
        q = np.rint(ref / s) + zp
        # a clamp at qmin only counts when qmin stands for a negative value
        sat = (q >= qmax) | ((q <= qmin) & (zp > qmin))
        if sat.mean() <= 0.05:
            return feeds, spread
    return feeds, spread


def _out_scale(cfg: QConfig) -> float | None:
    q = next(iter(cfg.outputs.values()), None)
    return None if q is None else q[0]


def check_case(case: Case) -> dict:
    res: dict = {"case": case.name, "emitter": case.emitter, "target": case.target}
    if case.note:
        res["note"] = case.note
    # control: native template at its own calibration
    feeds, spread = feeds_for(case, case.native_cfg)
    ref = qdq_reference(case.float_model, case.native_cfg, feeds)
    ctl = run(case.native, feeds)
    if ctl["outputs"] is None:
        res["error"] = f"control run failed: {ctl['error']}"
        return res
    res["control"] = {
        "spread": spread,
        **lsb_error(ctl["outputs"][0], ref[0], _out_scale(case.native_cfg)),
    }
    # emitted at the target calibration
    try:
        model = case.emitted()
    except Exception as exc:  # the emitter refused: report, no device run
        res["error"] = f"emitter refused: {exc}"
        return res
    tfeeds, tspread = feeds_for(case, case.target_cfg)
    tref = qdq_reference(case.float_model, case.target_cfg, tfeeds)
    out = run(model, tfeeds)
    if out["outputs"] is None:
        res["emitted"] = {"error": out["error"]}
    else:
        res["emitted"] = {
            "spread": tspread,
            **lsb_error(out["outputs"][0], tref[0], _out_scale(case.target_cfg)),
        }
    health = run(case.native, feeds)
    res["health_after"] = health["outputs"] == ctl["outputs"]
    return res


def device(out_path: str, only: list[str]) -> None:
    results = []
    for make in cases():
        case = make()
        if only and case.name not in only:
            continue
        print("running", case.name, flush=True)
        r = check_case(case)
        print(json.dumps(r), flush=True)
        results.append(r)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=1)
        if r.get("health_after") is False:
            print("health run differs from control: stopping", flush=True)
            break


def main(argv: list[str]) -> int:
    if argv[:1] == ["list"]:
        for make in cases():
            c = make()
            print(c.name, "|", c.emitter, "|", c.target)
        return 0
    if argv[:1] == ["device"] and len(argv) >= 2:
        device(argv[1], argv[2:])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
