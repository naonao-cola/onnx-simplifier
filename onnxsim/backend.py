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


def _ort_profile_prefix() -> Optional[str]:
    """The file prefix for onnxruntime's built-in session profiler, or ``None``
    when it is disabled.

    Constant folding runs each fold group through an ``onnxruntime`` session;
    setting ``ONNXSIM_ORT_PROFILE`` turns on onnxruntime's own per-operator
    profiler (``SessionOptions.enable_profiling``) for those sessions, which is
    finer-grained than onnxsim's ``OrtSession`` span. The variable names a file
    *prefix* -- onnxruntime writes one ``<prefix>_<timestamp>.json`` Chrome trace
    per session -- and mirrors ``ONNXSIM_PROFILE``: the truthy shorthands
    ``1``/``true``/``on``/``yes`` (and the empty string, as set by the
    ``ort_profile=""`` API default) select the default prefix.
    """
    value = os.environ.get("ONNXSIM_ORT_PROFILE")
    if value is None:
        return None
    if value.lower() in ("", "1", "true", "on", "yes"):
        return "onnxsim_ort_profile"
    return value


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


def validate_providers(providers: Optional[Sequence[Provider]]) -> None:
    """Validate a requested execution-provider list, raising if it cannot be
    honoured by the current backend.

    Callers use this to fail fast *before* constant folding starts. onnxsim's
    folding loop catches per-op executor errors and simply leaves the op
    unfolded, so an unavailable provider raised deep inside a fold would be
    swallowed and silently degrade to no folding rather than surfacing to the
    user. Checking here instead turns a misconfigured provider into an
    immediate, actionable error.

    ``None`` (fold on CPU) is always valid.
    """
    if providers is None:
        return
    if _HAS_ONNXRUNTIME:
        _check_providers_available(providers)
        return
    # Without onnxruntime only the pure-Python reference evaluator is available,
    # which runs on the CPU and cannot honour any other provider.
    non_cpu = [
        _provider_name(p)
        for p in providers
        if _provider_name(p) != "CPUExecutionProvider"
    ]
    if non_cpu:
        raise ValueError(
            "Execution providers other than CPUExecutionProvider require "
            "onnxruntime. Please install it (e.g. `pip install onnxruntime-gpu` "
            f"for CUDA). Requested providers: {non_cpu}."
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
    validate_providers(providers)
    sess_options = rt.SessionOptions()
    if custom_lib is not None:
        if os.path.exists(custom_lib):
            sess_options.register_custom_ops_library(custom_lib)
        else:
            raise ValueError("No such file '{}'".format(custom_lib))
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel(0)
    sess_options.log_severity_level = 3
    # Optionally turn on onnxruntime's own per-operator session profiler for this
    # folding session (separate from onnxsim's span profiler; see
    # ``_ort_profile_prefix``). onnxruntime writes one Chrome trace JSON per
    # session when the session ends.
    ort_profile_prefix = _ort_profile_prefix()
    if ort_profile_prefix is not None:
        sess_options.enable_profiling = True
        # onnxruntime appends "_<timestamp>.json" to the prefix. Guard the
        # attribute: it was added in newer onnxruntime, and without it the
        # default prefix ("onnxruntime_profile_") is used instead.
        if hasattr(sess_options, "profile_file_prefix"):
            sess_options.profile_file_prefix = ort_profile_prefix
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
    if ort_profile_prefix is not None:
        # Flush the per-operator trace to disk and stop profiling for this
        # session (otherwise the file is only written when the session is later
        # garbage-collected).
        sess.end_profiling()
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
    validate_providers(providers)
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
