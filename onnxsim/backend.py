"""Inference backend used for constant folding and correctness checking.

onnxruntime is preferred when it is available. If it is not installed,
onnxsim falls back to onnx's built-in reference evaluator so that
onnxruntime becomes an optional dependency (installing onnxruntime is
sometimes harmful, see https://github.com/onnxsim/onnxsim/issues/441).
"""

import os
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Union

import numpy as np
import onnx

try:
    import onnxruntime as rt  # type: ignore

    _HAS_ONNXRUNTIME = True
except ImportError:
    rt = None  # type: ignore
    _HAS_ONNXRUNTIME = False


def has_onnxruntime() -> bool:
    """Whether onnxruntime is available as the inference backend."""
    return _HAS_ONNXRUNTIME


def _run_with_onnxruntime(
    model: Union[str, bytes, onnx.ModelProto],
    inputs: Dict[str, np.ndarray],
    output_names: Optional[Sequence[str]],
    custom_lib: Optional[str],
) -> "OrderedDict[str, np.ndarray]":
    sess_options = rt.SessionOptions()
    if custom_lib is not None:
        if os.path.exists(custom_lib):
            sess_options.register_custom_ops_library(custom_lib)
        else:
            raise ValueError("No such file '{}'".format(custom_lib))
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel(0)
    sess_options.log_severity_level = 3
    if isinstance(model, onnx.ModelProto):
        model = model.SerializeToString()
    sess = rt.InferenceSession(
        model,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    if output_names is None:
        output_names = [x.name for x in sess.get_outputs()]
    run_options = rt.RunOptions()
    run_options.log_severity_level = 3
    outputs = sess.run(list(output_names), inputs, run_options=run_options)
    return OrderedDict(zip(output_names, outputs))


def _run_with_reference(
    model: Union[str, bytes, onnx.ModelProto],
    inputs: Dict[str, np.ndarray],
    output_names: Optional[Sequence[str]],
    custom_lib: Optional[str],
) -> "OrderedDict[str, np.ndarray]":
    if custom_lib is not None:
        raise ValueError("custom_lib is only supported when onnxruntime is installed")
    from onnx.reference import ReferenceEvaluator

    if isinstance(model, str):
        model = onnx.load(model)
    elif isinstance(model, bytes):
        model = onnx.load_from_string(model)
    sess = ReferenceEvaluator(model)
    if output_names is None:
        output_names = list(sess.output_names)
    outputs = sess.run(list(output_names), inputs)
    return OrderedDict(zip(output_names, outputs))


def run_model(
    model: Union[str, bytes, onnx.ModelProto],
    inputs: Dict[str, np.ndarray],
    output_names: Optional[Sequence[str]] = None,
    custom_lib: Optional[str] = None,
) -> "OrderedDict[str, np.ndarray]":
    """Run ``model`` on ``inputs`` and return an ordered ``{name: array}`` map.

    :param model: onnx ModelProto, serialized bytes, or a file path
    :param inputs: mapping from input name to numpy array
    :param output_names: outputs to fetch, ``None`` means all model outputs
    :param custom_lib: onnxruntime custom ops's shared library (onnxruntime only)
    """
    if _HAS_ONNXRUNTIME:
        return _run_with_onnxruntime(model, inputs, output_names, custom_lib)
    return _run_with_reference(model, inputs, output_names, custom_lib)
