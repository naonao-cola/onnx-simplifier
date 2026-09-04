"""Tests for onnxsim.ppq_integration.

PPQ itself (``pip install ppq``) is confirmed, not speculatively, unusable
in any environment with the modern ``onnx``/``protobuf`` versions onnxsim
requires -- see onnxsim/ppq_integration.py's module docstring for the two
independent import failures found by actually attempting the import in
this repo's own dev environment. That makes ``PPQ_AVAILABLE`` false here,
so these tests exercise the graceful-degradation contract (the same
contract ``scripts/axera/pulsar2_quantizer.py`` follows for its own
optional onnxruntime dependency) rather than PPQ's actual quantization
path, which cannot be run in this environment.
"""

import numpy as np
import onnx
import pytest

from onnxsim import ppq_integration


def test_ppq_unavailable_in_this_environment():
    """Documents the confirmed state of this dev environment: PPQ 0.6.6
    cannot be imported alongside a current onnx/protobuf, so the bridge
    reports itself unavailable rather than silently no-op'ing."""
    assert ppq_integration.PPQ_AVAILABLE is False
    reason = ppq_integration.unavailable_reason()
    assert reason is not None
    assert isinstance(reason, str) and reason


def test_quantize_with_ppq_raises_clear_error_when_unavailable():
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["x"], ["y"])],
        "g",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )

    with pytest.raises(RuntimeError, match="PPQ unavailable"):
        ppq_integration.quantize_with_ppq(
            model, [{"x": np.zeros((1,), dtype=np.float32)}]
        )


def test_quantize_with_ppq_reports_unavailable_reason_in_error():
    """The raised error should surface *why* PPQ is unusable (one of the
    two confirmed import failures), not just that it is."""
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["x"], ["y"])],
        "g",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )

    with pytest.raises(RuntimeError) as exc_info:
        ppq_integration.quantize_with_ppq(
            model, [{"x": np.zeros((1,), dtype=np.float32)}]
        )
    assert ppq_integration.unavailable_reason() in str(exc_info.value)
