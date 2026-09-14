"""Unit tests for scripts/apple/train_block_step_coreml.py.

A transformer-block Adam step built with onnxsim's training-graph tooling:
graph structure, ORT-CPU convergence, and the Core ML export (gated on
coremltools like the other export tests) -- none needing macOS.
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

from train_block_step_coreml import (  # noqa: E402
    PARAMS,
    build_block_step,
    initial_state,
    make_data,
)

import onnxsim  # noqa: E402
from onnxsim import qat_graph  # noqa: E402

ort = pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")


def _tiny():
    return dict(batch=4, seq=8, dim=32, heads=4, head_dim=8, ffn=64, out=32)


def test_block_step_has_expected_params_and_state():
    step, pshapes = build_block_step(**_tiny())
    assert set(pshapes) == set(PARAMS)
    assert set(step.state) == {
        *(PARAMS),
        *(f"m_{p}" for p in PARAMS),
        *(f"v_{p}" for p in PARAMS),
    }
    assert step.loss_name is not None


def test_block_step_trains_on_cpu():
    kw = _tiny()
    step, pshapes = build_block_step(**kw)
    x, y = make_data(kw["batch"], kw["seq"], kw["dim"], kw["out"])

    def scalars(t):
        corr = qat_graph.adam_bias_corrections(t)
        return {k: np.array([v], np.float32) for k, v in dict(lr=1e-3, **corr).items()}

    losses = []
    qat_graph.run_step_graph(
        step,
        constants={"x": x, "y": y},
        state=initial_state(pshapes),
        num_steps=6,
        scalars=scalars,
        losses=losses,
    )
    assert losses[-1] < losses[0], losses


ct = pytest.importorskip("coremltools", reason="coremltools is not installed")


def test_block_step_exports_to_coreml(tmp_path):
    step, _ = build_block_step(**_tiny())
    out_path = str(tmp_path / "block_step.mlpackage")
    onnxsim.export_coreml(step.model, out_path, skip_model_load=True)
    assert os.path.isdir(out_path)
