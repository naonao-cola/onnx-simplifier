"""Inference backend used for constant folding and correctness checking.

onnxruntime is preferred when it is available. If it is not installed,
onnxsim falls back to onnx's built-in reference evaluator so that
onnxruntime becomes an optional dependency (installing onnxruntime is
sometimes harmful, see https://github.com/onnxsim/onnxsim/issues/441).
"""

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx

try:
    import onnxruntime as rt  # type: ignore

    _HAS_ONNXRUNTIME = True
except ImportError:
    rt = None  # type: ignore
    _HAS_ONNXRUNTIME = False


# An execution provider is either the provider name (e.g.
# ``"CUDAExecutionProvider"``) or a ``(name, options_dict)`` tuple as accepted
# by ``onnxruntime.InferenceSession``. onnxruntime tries the providers in order
# and falls back to the next one for operators a provider cannot run, so the CPU
# provider is normally kept last as a catch-all.
Provider = Union[str, Tuple[str, Dict[str, object]]]

# Constant folding runs on CPU unless the caller asks otherwise. CPU is always
# available and deterministic, which keeps folding results stable regardless of
# the machine onnxsim happens to run on.
DEFAULT_PROVIDERS: List[str] = ["CPUExecutionProvider"]


def has_onnxruntime() -> bool:
    """Whether onnxruntime is available as the inference backend."""
    return _HAS_ONNXRUNTIME


def _provider_name(provider: Provider) -> str:
    """The provider name whether ``provider`` is a bare string or a
    ``(name, options)`` tuple."""
    return provider[0] if isinstance(provider, (tuple, list)) else provider


def _check_providers_available(providers: Sequence[Provider]) -> None:
    """Raise a helpful error if any requested provider is not built into the
    installed onnxruntime.

    onnxruntime otherwise only logs a warning and silently drops an unavailable
    provider (e.g. ``CUDAExecutionProvider`` when the CPU-only wheel is
    installed), so a user who asked to fold on the GPU would quietly get CPU
    execution instead. Failing loudly makes that misconfiguration obvious.
    """
    available = set(rt.get_available_providers())
    missing = [
        _provider_name(p) for p in providers if _provider_name(p) not in available
    ]
    if missing:
        raise ValueError(
            "The following execution provider(s) are not available in the "
            f"installed onnxruntime: {missing}. Available providers: "
            f"{sorted(available)}. For CUDA, install the GPU build with "
            "`pip install onnxruntime-gpu`."
        )


def _run_with_onnxruntime(
    model: Union[str, bytes, onnx.ModelProto],
    inputs: Dict[str, np.ndarray],
    output_names: Optional[Sequence[str]],
    custom_lib: Optional[str],
    providers: Optional[Sequence[Provider]] = None,
) -> "OrderedDict[str, np.ndarray]":
    if providers is None:
        providers = DEFAULT_PROVIDERS
    _check_providers_available(providers)
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
        providers=list(providers),
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
    providers: Optional[Sequence[Provider]] = None,
) -> "OrderedDict[str, np.ndarray]":
    if custom_lib is not None:
        raise ValueError("custom_lib is only supported when onnxruntime is installed")
    # The reference evaluator runs in pure Python on the CPU and has no notion of
    # execution providers. Asking for a non-CPU provider (e.g. CUDA) without
    # onnxruntime installed cannot be honoured, so surface that instead of
    # silently ignoring the request.
    if providers is not None and any(
        _provider_name(p) != "CPUExecutionProvider" for p in providers
    ):
        raise ValueError(
            "Execution providers other than CPUExecutionProvider require "
            "onnxruntime. Please install it (e.g. `pip install onnxruntime-gpu` "
            f"for CUDA). Requested providers: {[_provider_name(p) for p in providers]}."
        )
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
    providers: Optional[Sequence[Provider]] = None,
) -> "OrderedDict[str, np.ndarray]":
    """Run ``model`` on ``inputs`` and return an ordered ``{name: array}`` map.

    :param model: onnx ModelProto, serialized bytes, or a file path
    :param inputs: mapping from input name to numpy array
    :param output_names: outputs to fetch, ``None`` means all model outputs
    :param custom_lib: onnxruntime custom ops's shared library (onnxruntime only)
    :param providers: onnxruntime execution providers to run with, in priority
            order (e.g. ``["CUDAExecutionProvider", "CPUExecutionProvider"]``).
            ``None`` means CPU only. Non-CPU providers require onnxruntime.
    """
    if _HAS_ONNXRUNTIME:
        return _run_with_onnxruntime(model, inputs, output_names, custom_lib, providers)
    return _run_with_reference(model, inputs, output_names, custom_lib, providers)
