"""The ResNet18 training-step runner (``scripts/axera/step_runner.py``), the
persistent AXCL session (``axcl_session.py``) and the tinygrad ``AX`` device.

Offline tests need only the committed fixtures. The step graph itself
(``step.onnx``) and the device are local to the Axera box: those tests skip
elsewhere. Device tests run through ``AXCL_LXD_VM`` like every other device
test here, and hold ``/tmp/axcl-device.lock`` while they do.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnx
import pytest
from onnx import parser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

import misc_op_record_emit as misc  # noqa: E402
import step_runner as sr  # noqa: E402

_HAVE_STEP = os.path.exists(sr.STEP_ONNX) and os.path.exists(sr.STEP_REF)
needs_step = pytest.mark.skipif(
    not _HAVE_STEP, reason="step.onnx is local to the Axera box"
)


def _device_ok() -> bool:
    # pulsar2_docker.axcl_available() does the same, but importing it needs the
    # built onnxsim extension
    import subprocess

    vm = os.environ.get("AXCL_LXD_VM")
    if not vm:
        return False
    try:
        return (
            subprocess.run(
                ["lxc", "exec", vm, "--", "test", "-x", "/usr/bin/axcl/axcl_run_model"],
                capture_output=True,
                timeout=30,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


needs_device = pytest.mark.skipif(
    not _device_ok(), reason="needs the AX650 (AXCL_LXD_VM)"
)


def test_fake_quant_rounds_and_clips():
    x = np.array([-1.0, 0.0, 0.004, 0.006, 10.0], np.float32)
    got = sr.fake_quant(x, 0.01, 100, False)
    np.testing.assert_allclose(got, [-1.0, 0.0, 0.0, 0.01, 1.55], atol=1e-6)
    got = sr.fake_quant(x, 0.01, 0, True)
    np.testing.assert_allclose(got, [-1.0, 0.0, 0.0, 0.01, 1.27], atol=1e-6)


def _adam_graph() -> onnx.ModelProto:
    # w' = w - lr * m'; m' = 0.9 m + 0.1 g; g = x * w (a stand-in backward pass)
    return parser.parse_model(
        """
        <ir_version: 8, opset_import: ["" : 17]>
        g (float[4] x, float[4] w, float[4] w__m) => (float[4] w_new, float[4] m_new)
          <float b1 = {0.9}, float ob1 = {0.1}, float lr = {0.01}> {
            grad = Mul(x, w)
            a = Mul(b1, w__m)
            b = Mul(ob1, grad)
            m_new = Add(a, b)
            step = Mul(lr, m_new)
            w_new = Sub(w, step)
        }
        """
    )


def test_gradient_tensors_follow_the_first_moment_update():
    m = _adam_graph()
    for i, n in enumerate(m.graph.node):
        n.name = f"n{i}"
    state_map = {"w": "w_new", "w__m": "m_new"}
    assert sr.gradient_tensors(m, state_map) == {"w": "grad"}
    # the Adam math (not the gradient's own producer) is the optimizer update
    opt = sr.optimizer_nodes(m, state_map)
    by_out = {n.output[0]: n.name for n in m.graph.node}
    assert by_out["grad"] not in opt
    assert {by_out[t] for t in ("a", "b", "m_new", "step", "w_new")} <= opt


def test_misc_emit_keeps_the_mcode_dims_in_step():
    """A zero-point move re-encodes the MCode to a different length. The
    runtime reads the length from the initializer's dims: a stale one fails
    to load (0x80300709) or, inside a long session, wedged the card. This is
    the ReduceSum_62 of the first whole-step run."""
    key = "ReduceSum:16x1x512x4608:axes0:k0"
    tmpl, meta = misc.load_template(key)
    out = misc.emit_model(
        key,
        {"x": 2.1986086721881293e-05, "y": 0.0001273591333301738},
        {"x": 163, "y": 68},
    )
    init = misc.mcode_initializer(out)
    assert len(init.raw_data) != len(misc.mcode_initializer(tmpl).raw_data)
    assert list(init.dims) == [len(init.raw_data)]


@needs_step
def test_plan_covers_the_validated_nodes_and_marks_relu_reshapes_unsafe():
    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    records = sr.load_records()
    segs, host = sr.build_plan(model, records, calib)
    everything, _ = sr.build_plan(model, records, calib, include_unsafe=True)
    covered = sum(len(s.nodes) for s in everything)
    assert (
        covered
        == sr.axb.coverage_report(records, calibration=calib)["totals"]["covered"]
    )
    unsafe = [s for s in everything if s.unsafe]
    assert unsafe and all(s.kind == "reshape" for s in unsafe)
    assert sum(len(s.nodes) for s in segs) == covered - len(unsafe)


@needs_step
def test_float_mode_reproduces_the_reference_step():
    model = sr.load_step()
    ref = sr.load_reference()
    outs, _ = sr.StepRunner(model, []).run(ref["feeds"], "float")
    loss = float(np.ravel(outs["distill__add_27"])[0])
    assert loss == pytest.approx(
        float(np.ravel(ref["ref"]["distill__add_27"])[0]), rel=1e-5
    )


@needs_device
def test_session_health_check_on_device():
    import axcl_session

    with axcl_session.AXSession() as s:
        assert axcl_session.health_check(s) <= 1.01


@needs_device
@needs_step
def test_one_segment_of_each_kind_matches_its_simulation_on_device():
    import axcl_session

    model = sr.load_step()
    calib = sr.axb.load_calibration(sr.STEP_CALIB)
    segs, _ = sr.build_plan(model, sr.load_records(), calib)
    first: dict[str, sr.Segment] = {}
    for s in segs:
        first.setdefault(s.kind, s)
    ref = sr.load_reference()
    with axcl_session.AXSession() as sess:
        runner = sr.StepRunner(model, list(first.values()), sess, health_every=1)
        _, stats = runner.run(ref["feeds"], "npu")
    assert {st.kind for st in stats} == set(first)
    for st in stats:
        assert sr.segment_passed(vars(st)), st


@needs_device
def test_tinygrad_ax_device_runs_a_relu_and_a_matmul_chain():
    tinygrad = pytest.importorskip("tinygrad")
    import tinygrad_ax_backend as axb
    from tinygrad.device import Buffer, Device

    axb.register_ax_device()
    dev = Device["AX"]
    classes = axb.tinygrad_classes()
    try:
        # Relu: an AXCompiler request (template + ElementwiseScaleEdit)
        s = 0.01
        key = axb.TemplateKey("Relu", ((16, 512, 7, 7),), calibration_class="x0,y0")
        src = axb.build_request(key, [axb.ElementwiseScaleEdit({"x": s, "y": s})])
        prog = classes["AXProgram"](dev, classes["AXCompiler"]().compile(src))
        x = np.random.default_rng(0).uniform(0, 2, (16, 512, 7, 7)).astype(np.float32)
        xb = Buffer("AX", x.size, tinygrad.dtypes.float32, initial_value=x.tobytes())
        yb = Buffer("AX", x.size, tinygrad.dtypes.float32).allocate()
        prog(yb._buf, xb._buf, wait=True)
        want = np.clip(np.rint(x / np.float32(s)), 0, 255) * np.float32(s)
        assert np.abs(yb.numpy() - want.ravel()).max() <= s * 1.01

        # a live-operand MatMul chain: the step's fc dX (matmul_record_emit)
        if _HAVE_STEP:
            model = sr.load_step()
            calib = sr.axb.load_calibration(sr.STEP_CALIB)
            segs, _ = sr.build_plan(
                model, sr.load_records(), calib, kinds={"matmul_chain"}
            )
            seg = next(g for g in segs if g.name == "MatMul_36")
            prog = classes["AXProgram"](dev, seg.emit().SerializeToString())
            rng = np.random.default_rng(1)
            ins = [
                rng.uniform(-0.02, 0.02, sp.shape).astype(np.float32)
                for sp in prog.model.inputs
            ]
            bufs = [
                Buffer("AX", a.size, tinygrad.dtypes.float32, initial_value=a.tobytes())
                for a in ins
            ]
            out = prog.model.outputs[0]
            ob = Buffer(
                "AX", int(np.prod(out.shape)), tinygrad.dtypes.float32
            ).allocate()
            prog(ob._buf, *[b._buf for b in bufs], wait=True)
            env = dict(zip(seg.inputs, ins))
            sim = sr.StepRunner(model, [seg])._sim(seg, env)[0]
            lsb = np.abs(ob.numpy() - sim.ravel()).max() / seg.out_q[0][0]
            assert lsb <= 2.01
    finally:
        axb.close_ax_session()
