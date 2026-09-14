"""Unit tests for scripts/apple/train_mlp_step_coreml.py.

The step builder is plain ONNX construction on top of onnxsim's training
graph tooling, so graph structure and the Core ML scalar workaround are
tested directly. End-to-end training (loss decreases) runs on ONNX Runtime,
and the Core ML export check is gated on coremltools like the other export
tests -- neither needs macOS.
"""

import os
import sys

import numpy as np
import pytest

_APPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "apple"
)
if _APPLE_DIR not in sys.path:
    sys.path.insert(0, _APPLE_DIR)

from train_mlp_step_coreml import (  # noqa: E402
    build_step,
    calibrate_scales,
    initial_state,
    make_data,
)

import onnxsim  # noqa: E402
from onnxsim import qat_graph  # noqa: E402

ort = pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")


def test_widen_rank0_inputs_leaves_other_shapes_alone():
    step = build_step(batch=8, dim=16, hidden=32, out=16)
    scalar_names = {"lr", "m_correction", "v_correction"}
    for vi in step.model.graph.input:
        rank = len(vi.type.tensor_type.shape.dim)
        if vi.name in scalar_names:
            assert rank == 1, (vi.name, rank)
        else:
            assert rank >= 1, (vi.name, rank)


def test_step_graph_trains_on_cpu():
    batch, dim, hidden, out = 8, 16, 32, 16
    step = build_step(batch, dim, hidden, out)
    x, y = make_data(batch, dim, out)
    shapes = {
        "w1": (dim, hidden),
        "b1": (hidden,),
        "w2": (hidden, out),
        "b2": (out,),
    }

    def scalars(t):
        corr = qat_graph.adam_bias_corrections(t)
        # The model declares its scalars shape-[1] (see
        # _widen_rank0_inputs), so feed them that way too.
        return {k: np.array([v], np.float32) for k, v in dict(lr=1e-3, **corr).items()}

    losses = []
    qat_graph.run_step_graph(
        step,
        constants={"x": x, "y": y},
        state=initial_state(shapes),
        num_steps=10,
        scalars=scalars,
        losses=losses,
    )
    assert losses[-1] < losses[0], losses


ct = pytest.importorskip("coremltools", reason="coremltools is not installed")


def test_step_graph_exports_to_coreml(tmp_path):
    step = build_step(batch=8, dim=16, hidden=32, out=16)
    out_path = str(tmp_path / "mlp_step.mlpackage")
    onnxsim.export_coreml(step.model, out_path, skip_model_load=True)
    assert os.path.isdir(out_path)


def test_qat_int8_step_graph_trains_on_cpu():
    # Fake-quantized weights (straight-through gradients) still optimize:
    # the int8 forward perturbs each step, but Adam on the fp32 master
    # weights converges all the same.
    batch, dim, hidden, out = 8, 16, 32, 16
    shapes = {
        "w1": (dim, hidden),
        "b1": (hidden,),
        "w2": (hidden, out),
        "b2": (out,),
    }
    scales = calibrate_scales(initial_state(shapes))
    assert set(scales) == {"w1", "w2"}
    assert all(v > 0 for v in scales.values())
    step = build_step(batch, dim, hidden, out, scales=scales, qat_int8=True)
    x, y = make_data(batch, dim, out)

    def scalars(t):
        corr = qat_graph.adam_bias_corrections(t)
        return {k: np.array([v], np.float32) for k, v in dict(lr=1e-3, **corr).items()}

    losses = []
    qat_graph.run_step_graph(
        step,
        constants={"x": x, "y": y},
        state=initial_state(shapes),
        num_steps=10,
        scalars=scalars,
        losses=losses,
    )
    assert losses[-1] < losses[0], losses


def test_qat_int8_step_graph_exports_to_coreml(tmp_path):
    batch, dim, hidden, out = 8, 16, 32, 16
    shapes = {
        "w1": (dim, hidden),
        "b1": (hidden,),
        "w2": (hidden, out),
        "b2": (out,),
    }
    step = build_step(
        batch,
        dim,
        hidden,
        out,
        scales=calibrate_scales(initial_state(shapes)),
        qat_int8=True,
    )
    out_path = str(tmp_path / "mlp_step_qat.mlpackage")
    onnxsim.export_coreml(step.model, out_path, skip_model_load=True)
    assert os.path.isdir(out_path)


def test_resident_export_holds_state_and_returns_only_loss(tmp_path):
    # The resident form drops next-state outputs (they persist in MLState)
    # and only the loss crosses the boundary; needs no Apple hardware to
    # check, only the MIL program.
    from onnxsim import coreml_export

    step = build_step(batch=8, dim=16, hidden=32, out=16)
    out_path = str(tmp_path / "mlp_step_resident.mlpackage")
    onnxsim.export_coreml(
        step.model, out_path, skip_model_load=True, state=dict(step.state)
    )
    assert os.path.isdir(out_path)
    mb, types, Function, Program, RangeDim, TensorType = coreml_export._import_mil()
    prog, _ = coreml_export._build_mil_program(
        step.model,
        mb,
        types,
        Function,
        Program,
        RangeDim,
        TensorType,
        opset_version=ct.target.iOS18,
        state=dict(step.state),
    )
    func = prog.functions["main"]
    op_types = [op.op_type for op in func.operations]
    assert "read_state" in op_types
    assert "coreml_update_state" in op_types
    assert [o.name for o in func.outputs] == [step.loss_name]
