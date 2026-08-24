"""A unified, typed entry point over onnxsim's quantization schemes
(:class:`QuantizationConfig` / :func:`quantize`), and a data-driven
measurement of a specific quantized model's actual accuracy drop
(:func:`measure_accuracy_drop`).

onnxsim ships more than a dozen ``quantize_*`` functions
(:func:`onnxsim.quantize_dynamic`, :func:`onnxsim.quantize_static`,
:func:`onnxsim.quantize_weight_only_int4`, ...), each its own scheme with its
own parameter surface, documented and callable directly as always. This
module adds a second, unified way to reach all of them: describe *what*
quantization you want (scheme, dtype, granularity, calibration settings) as
one :class:`QuantizationConfig`, and let :func:`quantize` dispatch to the
right underlying function -- useful for code that picks a scheme
programmatically (a sweep over configs, a config file, a CLI flag) rather
than calling a specific ``quantize_*`` function by name.

For how much accuracy a given quantization actually costs, two tools, at two
different price points:

- :func:`onnxsim.estimate_model_quantization_drop` (in
  ``precision_estimator.py``) -- static, no execution or data needed, an
  *estimate* from the model's weights and shapes alone. Fast pre-check.
- :func:`measure_accuracy_drop`, here -- runs the float and quantized models
  through ONNX Runtime (or the reference evaluator, see ``backend.py``) on
  the same input data and reports actual output differences. Slower (needs
  data and two full model runs per sample), but it's a measurement, not an
  estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx

from onnxsim import backend
from onnxsim.calibration import (
    _ELEM_TYPE_TO_NP,
    Tensors,
    generate_random_calibration_data,
    quantize_qoperator,
    quantize_static,
    quantize_static_int16,
)
from onnxsim.onnx_simplifier import (
    quantize_bf16,
    quantize_dynamic,
    quantize_dynamic_matmul_integer_to_float,
    quantize_fp8,
    quantize_fp16,
    quantize_ternary,
    quantize_weight_only,
    quantize_weight_only_int4,
    quantize_weight_only_int8_block,
    quantize_weight_only_int16,
)

# scheme -> the dtypes it accepts for QuantizationConfig.dtype.
_SCHEME_DTYPES = {
    "dynamic": {"int8"},
    "dynamic_fused": {"int8"},
    "ternary": {"int8"},
    "weight_only": {"int8", "int16", "int4"},
    "static": {"int8"},
    "static_int16": {"int16"},
    "qoperator": {"int8"},
    "float": {"float16", "bfloat16", "float8_e4m3", "float8_e5m2"},
}
_CALIBRATION_SCHEMES = {"static", "static_int16", "qoperator"}


@dataclass
class QuantizationConfig:
    """Describes one onnxsim quantization scheme and its parameters, for
    :func:`quantize` to dispatch on. Every field not relevant to ``scheme``
    is simply ignored (e.g. ``calibration_data`` for ``scheme="dynamic"``,
    which needs none) -- see :func:`quantize`'s docstring for the full
    scheme/dtype/granularity matrix and which onnxsim function each maps to.

    :param scheme: one of ``"dynamic"``, ``"dynamic_fused"``, ``"ternary"``,
            ``"weight_only"``, ``"static"``, ``"static_int16"``,
            ``"qoperator"``, ``"float"``.
    :param dtype: the quantized representation. Meaning depends on
            ``scheme`` -- see :func:`quantize`.
    :param granularity: ``"per_channel"`` (default) or ``"per_block"``.
            Only ``scheme="weight_only"`` currently offers a choice (its
            ``dtype="int8"`` case); every other scheme/dtype combination has
            exactly one granularity onnxsim implements today, and this field
            is ignored for them.
    :param calibration_data: representative input batches for the
            calibration-based schemes (``"static"``, ``"static_int16"``,
            ``"qoperator"``) -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data`.
    :param num_calibration_samples: random calibration batches to generate
            when ``calibration_data`` is omitted (calibration-based schemes
            only).
    :param seed: seed for the random calibration data (calibration-based
            schemes only; ignored if ``calibration_data`` is supplied).
    :param providers: onnxruntime execution providers to calibrate on
            (calibration-based schemes only).
    :param calibration_method: ``"minmax"`` (default) or ``"entropy"``,
            passed through to :func:`onnxsim.calibrate` (calibration-based
            schemes only).
    :param keep_io_types: for ``scheme="float"`` only -- keep the model's
            external input/output types at float32 (inserting boundary
            ``Cast`` nodes) instead of redeclaring them in the target
            format. Default ``True``.
    """

    scheme: str
    dtype: str = "int8"
    granularity: str = "per_channel"
    calibration_data: Optional[Sequence[Tensors]] = None
    num_calibration_samples: int = 8
    seed: int = 0
    providers: Optional[Sequence[str]] = None
    calibration_method: str = "minmax"
    keep_io_types: bool = True


def quantize(
    model: Union[str, onnx.ModelProto], config: QuantizationConfig
) -> onnx.ModelProto:
    """Quantizes ``model`` according to ``config``, dispatching to the
    matching onnxsim ``quantize_*`` function. The scheme/dtype/granularity
    matrix:

    ================  ===============  ==============  ===========================================
    scheme            dtype            granularity     onnxsim function
    ================  ===============  ==============  ===========================================
    dynamic           int8             per_channel      :func:`onnxsim.quantize_dynamic`
    dynamic_fused     int8             per_channel      :func:`onnxsim.quantize_dynamic_matmul_integer_to_float`
    ternary           int8             per_channel      :func:`onnxsim.quantize_ternary`
    weight_only       int8             per_channel      :func:`onnxsim.quantize_weight_only`
    weight_only       int8             per_block        :func:`onnxsim.quantize_weight_only_int8_block`
    weight_only       int16            per_channel      :func:`onnxsim.quantize_weight_only_int16`
    weight_only       int4             per_block        :func:`onnxsim.quantize_weight_only_int4`
    static            int8             per_channel      :func:`onnxsim.quantize_static`
    static_int16      int16            per_channel      :func:`onnxsim.quantize_static_int16`
    qoperator         int8             per_channel      :func:`onnxsim.quantize_qoperator`
    float              float16          n/a             :func:`onnxsim.quantize_fp16`
    float              bfloat16         n/a             :func:`onnxsim.quantize_bf16`
    float              float8_e4m3      n/a             :func:`onnxsim.quantize_fp8` (format="e4m3")
    float              float8_e5m2      n/a             :func:`onnxsim.quantize_fp8` (format="e5m2")
    ================  ===============  ==============  ===========================================

    ``weight_only``/``int4`` is always block-wise (block_size=32, the only
    granularity :func:`onnxsim.quantize_weight_only_int4` implements) --
    ``granularity`` is not consulted for it, and likewise for every other
    row with exactly one implemented granularity.

    Raises :class:`ValueError` for an unknown ``scheme``, a ``dtype`` not
    valid for that ``scheme``, or (``scheme="weight_only"``, ``dtype="int8"``)
    with a ``granularity`` other than ``"per_channel"``/``"per_block"``.

    :param model: onnx ModelProto object or file path
    :param config: the quantization scheme and its parameters
    :returns: the quantized onnx ModelProto
    """
    scheme = config.scheme
    valid_dtypes = _SCHEME_DTYPES.get(scheme)
    if valid_dtypes is None:
        raise ValueError(
            f"unknown QuantizationConfig.scheme {scheme!r}; expected one of "
            f"{sorted(_SCHEME_DTYPES)}"
        )
    if config.dtype not in valid_dtypes:
        raise ValueError(
            f"scheme={scheme!r} does not support dtype={config.dtype!r}; "
            f"expected one of {sorted(valid_dtypes)}"
        )

    if scheme == "dynamic":
        return quantize_dynamic(model)
    if scheme == "dynamic_fused":
        return quantize_dynamic_matmul_integer_to_float(model)
    if scheme == "ternary":
        return quantize_ternary(model)

    if scheme == "weight_only":
        if config.dtype == "int16":
            return quantize_weight_only_int16(model)
        if config.dtype == "int4":
            return quantize_weight_only_int4(model)
        # dtype == "int8": the one scheme/dtype pair with a granularity choice.
        if config.granularity == "per_channel":
            return quantize_weight_only(model)
        if config.granularity == "per_block":
            return quantize_weight_only_int8_block(model)
        raise ValueError(
            "scheme='weight_only', dtype='int8' supports granularity "
            f"'per_channel' or 'per_block', got {config.granularity!r}"
        )

    if scheme in _CALIBRATION_SCHEMES:
        fn = {
            "static": quantize_static,
            "static_int16": quantize_static_int16,
            "qoperator": quantize_qoperator,
        }[scheme]
        return fn(
            model,
            calibration_data=config.calibration_data,
            num_calibration_samples=config.num_calibration_samples,
            seed=config.seed,
            providers=config.providers,
            method=config.calibration_method,
        )

    # scheme == "float"
    if config.dtype == "float16":
        return quantize_fp16(model, keep_io_types=config.keep_io_types)
    if config.dtype == "bfloat16":
        return quantize_bf16(model, keep_io_types=config.keep_io_types)
    fp8_format = "e4m3" if config.dtype == "float8_e4m3" else "e5m2"
    return quantize_fp8(model, format=fp8_format, keep_io_types=config.keep_io_types)


def _cast_batch_to_model_inputs(model: onnx.ModelProto, batch: Tensors) -> Tensors:
    """Casts each array in `batch` to the graph input's own declared dtype,
    for a floating-point mismatch only -- e.g. a ``keep_io_types=False``
    float16 quantization (see :func:`onnxsim.quantize_fp16`) redeclares
    graph inputs in the narrow format directly, so the same float32
    calibration data used against the original model needs casting before
    it can feed the quantized one. Integer/bool inputs, and bfloat16/float8
    targets (no native numpy dtype in ``_ELEM_TYPE_TO_NP``; pre-cast
    ``calibration_data`` yourself with ``ml_dtypes`` for those), are left
    untouched.
    """
    elem_type_by_name = {
        i.name: i.type.tensor_type.elem_type for i in model.graph.input
    }
    out = {}
    for name, arr in batch.items():
        elem_type = elem_type_by_name.get(name)
        np_dtype = _ELEM_TYPE_TO_NP.get(elem_type) if elem_type is not None else None
        if (
            np_dtype is not None
            and np.issubdtype(np_dtype, np.floating)
            and arr.dtype != np_dtype
        ):
            arr = arr.astype(np_dtype)
        out[name] = arr
    return out


@dataclass
class OutputAccuracyStats:
    """Worst-case-over-samples accuracy stats for one model output -- see
    :func:`measure_accuracy_drop`."""

    output_name: str
    relative_l2: float  # ||float - quantized|| / ||float||, worst over samples
    max_abs_error: float  # max(|float - quantized|), worst over samples
    cosine_similarity: (
        float  # dot(float, quantized) / (||float|| * ||quantized||), worst over samples
    )


@dataclass
class AccuracyDropReport:
    """Measured (not estimated) accuracy drop between a float model and a
    quantized version of it -- see :func:`measure_accuracy_drop`."""

    num_samples: int
    per_output: Dict[str, OutputAccuracyStats] = field(default_factory=dict)
    worst_relative_l2: float = float("nan")  # max over every output/sample
    worst_cosine_distance: float = float("nan")  # 1 - min cosine_similarity
    all_finite: bool = True  # False if the quantized model ever produced NaN/Inf


def measure_accuracy_drop(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
) -> AccuracyDropReport:
    """Runs ``float_model`` and ``quantized_model`` on the same input data
    (through :func:`onnxsim.backend.run_model` -- onnxruntime when
    installed, the pure-Python reference evaluator otherwise) and reports
    how far each of the quantized model's outputs actually drifts from the
    float model's -- an empirical measurement, not
    :func:`onnxsim.estimate_model_quantization_drop`'s static estimate.

    Assumes ``float_model`` and ``quantized_model`` declare the same output
    *names* and count -- true of every onnxsim ``quantize_*``/:func:`quantize`
    call, which never renames or adds/removes graph outputs. Per-output
    stats are the **worst case across samples**, not an average: the point
    of measuring accuracy drop is to know the worst a deployment might see,
    not to average it away.

    :param float_model: the original (unquantized) onnx ModelProto or file path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), e.g. from :func:`quantize` or any
            ``quantize_*`` function
    :param calibration_data: representative input batches to measure on.
            Each batch is a ``{input_name: np.ndarray}`` dict matching
            ``float_model``'s graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and :func:`onnxsim.load_huggingface_calibration_data`
            (real data, a much more representative measurement than random
            input for a real deployment).
    :param num_samples: number of random batches to generate when
            ``calibration_data`` is not supplied
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run both models on
    :returns: the measured accuracy-drop report
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )

    output_names = [o.name for o in float_model.graph.output]
    per_output_l2: Dict[str, List[float]] = {name: [] for name in output_names}
    per_output_abs: Dict[str, List[float]] = {name: [] for name in output_names}
    per_output_cos: Dict[str, List[float]] = {name: [] for name in output_names}
    all_finite = True

    for batch in calibration_data:
        float_out = backend.run_model(float_model, batch, providers=providers)
        quantized_out = backend.run_model(
            quantized_model,
            _cast_batch_to_model_inputs(quantized_model, batch),
            providers=providers,
        )
        for name in output_names:
            f = np.asarray(float_out[name], dtype=np.float64).ravel()
            q = np.asarray(quantized_out[name], dtype=np.float64).ravel()
            if not np.all(np.isfinite(q)):
                all_finite = False
            f_norm = float(np.linalg.norm(f))
            q_norm = float(np.linalg.norm(q))
            rel_l2 = float(np.linalg.norm(f - q)) / max(f_norm, 1e-12)
            max_abs = float(np.max(np.abs(f - q))) if f.size else 0.0
            denom = f_norm * q_norm
            cos_sim = float(np.dot(f, q) / denom) if denom > 0 else float("nan")
            per_output_l2[name].append(rel_l2)
            per_output_abs[name].append(max_abs)
            per_output_cos[name].append(cos_sim)

    per_output: Dict[str, OutputAccuracyStats] = {}
    for name in output_names:
        per_output[name] = OutputAccuracyStats(
            output_name=name,
            relative_l2=max(per_output_l2[name]),
            max_abs_error=max(per_output_abs[name]),
            cosine_similarity=min(per_output_cos[name]),
        )

    worst_relative_l2 = max(
        (s.relative_l2 for s in per_output.values()), default=float("nan")
    )
    worst_cosine_similarity = min(
        (s.cosine_similarity for s in per_output.values()), default=1.0
    )
    return AccuracyDropReport(
        num_samples=len(calibration_data),
        per_output=per_output,
        worst_relative_l2=worst_relative_l2,
        worst_cosine_distance=1.0 - worst_cosine_similarity,
        all_finite=all_finite,
    )
