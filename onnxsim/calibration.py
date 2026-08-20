"""Calibration data generation and static (calibration-based) quantization.

``onnxsim.quantize_dynamic`` needs no calibration data: it defers the
activation's quantization range to a ``DynamicQuantizeLinear`` computed fresh
on every inference. Static quantization instead *fixes* that range ahead of
time, from representative data -- usually a better trade (no per-inference
range computation) but only as good as the calibration data it is given.

This module provides two calibration data sources, meant to be used in order
as a model moves from a quick smoke test towards real deployment:

- :func:`generate_random_calibration_data` -- synthetic random data. Works
  out of the box with no external dependency or dataset to find, so
  :func:`quantize_static` falls back to it automatically. Good for checking
  the quantization pipeline itself works; a poor proxy for a real model's
  actual activation statistics.
- :func:`load_huggingface_calibration_data` -- real examples pulled from a
  Hugging Face Hub dataset (needs the optional ``datasets`` package). Gives
  calibration ranges that actually reflect deployment-time data, at the cost
  of needing a dataset whose columns can be matched to the model's inputs.

:func:`calibrate` runs the float model over either data source through
ONNX Runtime to produce the ``{tensor_name: (min, max)}`` ranges
:func:`onnxsim.quantize_static` (this module's main entry point) needs.
"""

import itertools
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx

import onnxsim.onnxsim_cpp2py_export as C

Tensors = Dict[str, np.ndarray]

_ELEM_TYPE_TO_NP = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.DOUBLE: np.float64,
    onnx.TensorProto.FLOAT16: np.float16,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT16: np.int16,
    onnx.TensorProto.INT8: np.int8,
    onnx.TensorProto.UINT8: np.uint8,
    onnx.TensorProto.BOOL: np.bool_,
}


def _input_specs(model: onnx.ModelProto) -> List[Tuple[str, List[int], type]]:
    """(name, shape, np_dtype) for every graph input that is not an initializer.

    Dynamic dimensions (including an unset/symbolic batch dimension) are
    fixed to 1, so the returned shapes are always fully concrete.
    """
    initializer_names = {i.name for i in model.graph.initializer}
    specs = []
    for ipt in model.graph.input:
        if ipt.name in initializer_names:
            continue
        shape = [
            dim.dim_value if dim.dim_value > 0 else 1
            for dim in ipt.type.tensor_type.shape.dim
        ]
        np_dtype = _ELEM_TYPE_TO_NP.get(ipt.type.tensor_type.elem_type, np.float32)
        specs.append((ipt.name, shape, np_dtype))
    return specs


def generate_random_calibration_data(
    model: Union[str, onnx.ModelProto],
    num_samples: int = 8,
    seed: int = 0,
) -> List[Tensors]:
    """
    Generate ``num_samples`` batches of random input data matching ``model``'s
    input shapes/dtypes: a calibration data source that works with no
    external dependency, for a first pass through the static quantization
    pipeline before wiring up real representative data (e.g.
    :func:`load_huggingface_calibration_data`).

    Floating-point inputs are drawn from a standard normal distribution (a
    closer proxy for typical activation/feature statistics than a uniform
    [0, 1) draw); integer and boolean inputs are filled with zeros, a safe
    default when random values are unlikely to be valid indices (e.g. token
    ids) -- pass your own calibration data for a model whose behavior
    actually depends on integer input values.

    :param model: onnx ModelProto object or file path
    :param num_samples: number of calibration batches to generate
    :param seed: seed for reproducibility
    :returns: a list of ``{input_name: np.ndarray}`` dicts, one per batch
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    specs = _input_specs(model)
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(num_samples):
        batch = {}
        for name, shape, np_dtype in specs:
            if np.issubdtype(np_dtype, np.floating):
                batch[name] = rng.standard_normal(shape).astype(np_dtype)
            else:
                batch[name] = np.zeros(shape, dtype=np_dtype)
        batches.append(batch)
    return batches


# A short list of common Hugging Face dataset column names for each of a few
# common ONNX input names, tried (after `field_map` and an exact name match)
# by load_huggingface_calibration_data before giving up on an input.
_COMMON_COLUMN_ALIASES = {
    "pixel_values": ["pixel_values", "image", "img"],
    "input_ids": ["input_ids", "input_id", "tokens", "text"],
    "attention_mask": ["attention_mask", "mask"],
    "input_values": ["input_values", "audio", "speech"],
}


def load_huggingface_calibration_data(
    dataset: str,
    model: Union[str, onnx.ModelProto],
    num_samples: int = 8,
    split: str = "train",
    field_map: Optional[Dict[str, str]] = None,
    seed: int = 0,
) -> List[Tensors]:
    """
    Best-effort loader that pulls ``num_samples`` real examples from a
    Hugging Face ``datasets`` dataset and adapts them into calibration
    batches for ``model``. Requires the optional ``datasets`` package
    (``pip install datasets``).

    Matching a dataset's columns to a model's ONNX input names is inherently
    ambiguous -- there is no schema linking the two -- so this only handles
    the common case. For each input, in order: an explicit ``field_map``
    entry, a same-named column, then a short list of common aliases (e.g.
    ONNX input "pixel_values" matches a dataset column named "image"). If an
    input still cannot be matched, this raises ``ValueError`` naming it
    rather than silently feeding it random or zero data -- pass ``field_map``
    to resolve the mismatch, or fall back to
    :func:`generate_random_calibration_data` (or your own preprocessing
    pipeline, feeding its output directly to :func:`quantize_static` /
    :func:`calibrate` as ``calibration_data``) instead of this loader.

    A matched column's values are converted to a numpy array and cast to the
    target input's dtype; this only handles a column that is already a plain
    (optionally ragged, single-example) tensor, so a variable-length sequence
    column (e.g. un-padded tokenized text) needs its own tokenizer/padding
    step before it can be used here.

    :param dataset: a Hugging Face Hub dataset id, e.g. "mnist" or "cifar10"
    :param model: onnx ModelProto object or file path
    :param num_samples: number of examples to pull from the dataset
    :param split: dataset split to sample from
    :param field_map: optional explicit ``{onnx_input_name: dataset_column_name}``
            overrides, for a dataset whose columns don't already match by
            name or common alias
    :param seed: seed used to shuffle the dataset before sampling
    :returns: a list of ``{input_name: np.ndarray}`` dicts, one per example
    """
    try:
        import datasets as hf_datasets
    except ImportError as e:
        raise ImportError(
            "load_huggingface_calibration_data needs the optional 'datasets' "
            "package: pip install datasets"
        ) from e

    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    specs = _input_specs(model)

    ds = hf_datasets.load_dataset(dataset, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=max(num_samples * 4, 64))
    examples = list(itertools.islice(ds, num_samples))
    if not examples:
        raise ValueError(f'Dataset "{dataset}" split "{split}" yielded no examples.')
    columns = set(examples[0].keys())

    resolved: Dict[str, str] = {}
    unmatched: List[str] = []
    for name, _shape, _np_dtype in specs:
        if field_map and name in field_map:
            resolved[name] = field_map[name]
            continue
        candidates = [name] + _COMMON_COLUMN_ALIASES.get(name, [])
        match = next((c for c in candidates if c in columns), None)
        if match is None:
            match = next((c for c in columns if c.lower() == name.lower()), None)
        if match is None:
            unmatched.append(name)
        else:
            resolved[name] = match

    if unmatched:
        raise ValueError(
            'Could not match onnx input(s) {} to any column of dataset "{}" '
            "(columns: {}). Pass field_map={{onnx_input_name: "
            "dataset_column_name}} to resolve manually.".format(
                unmatched, dataset, sorted(columns)
            )
        )

    batches = []
    for example in examples:
        batch = {}
        for name, shape, np_dtype in specs:
            arr = np.asarray(example[resolved[name]], dtype=np_dtype)
            if arr.shape != tuple(shape):
                if arr.size != int(np.prod(shape)):
                    raise ValueError(
                        f'Dataset "{dataset}" column "{resolved[name]}" has '
                        f"{arr.size} elements per example, which does not "
                        f'match onnx input "{name}"\'s shape {shape} '
                        f"({int(np.prod(shape))} elements). Pass your own "
                        "preprocessing pipeline's output as calibration_data "
                        "instead."
                    )
                arr = arr.reshape(shape)
            batch[name] = arr
        batches.append(batch)
    return batches


def calibrate(
    model: Union[str, onnx.ModelProto],
    calibration_data: Sequence[Tensors],
    providers: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Run the float ``model`` over every batch in ``calibration_data`` through
    ONNX Runtime, recording each quantizable activation's observed (min, max)
    range across all batches -- the calibration ranges
    :func:`onnxsim.quantize_static` needs.

    :param model: onnx ModelProto object or file path
    :param calibration_data: representative input batches, e.g. from
            :func:`generate_random_calibration_data` or
            :func:`load_huggingface_calibration_data`
    :param providers: onnxruntime execution providers to run calibration on
            (defaults to onnxruntime's own default provider selection)
    :returns: ``{tensor_name: (min, max)}`` for every tensor
            ``onnxsim_cpp2py_export.list_quantizable_activations`` reports
    """
    import onnxruntime as ort

    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    model_bytes = model.SerializeToString()
    tensor_names = set(C.list_quantizable_activations(model_bytes))
    if not tensor_names:
        return {}

    # Expose every candidate tensor as an extra graph output, so onnxruntime
    # computes (and returns) it without the graph itself needing to change.
    calib_model = onnx.ModelProto()
    calib_model.CopyFrom(model)
    existing_outputs = {o.name for o in calib_model.graph.output}
    for name in tensor_names:
        if name not in existing_outputs:
            calib_model.graph.output.append(onnx.ValueInfoProto(name=name))

    sess = ort.InferenceSession(
        calib_model.SerializeToString(),
        providers=list(providers) if providers else None,
    )
    output_names = [o.name for o in sess.get_outputs()]

    ranges: Dict[str, Tuple[float, float]] = {}
    for batch in calibration_data:
        outputs = sess.run(output_names, batch)
        for name, value in zip(output_names, outputs):
            if name not in tensor_names:
                continue
            arr = np.asarray(value)
            if arr.size == 0:
                continue
            batch_min = float(arr.min())
            batch_max = float(arr.max())
            if name in ranges:
                prev_min, prev_max = ranges[name]
                ranges[name] = (min(prev_min, batch_min), max(prev_max, batch_max))
            else:
                ranges[name] = (batch_min, batch_max)
    return ranges


def quantize_static(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """
    Statically (calibration-based) quantize every MatMul, and every "vanilla"
    Gemm (transA=0, alpha=1, beta=1), whose weight is a constant 2-D float32
    tensor.

    Unlike :func:`onnxsim.quantize_dynamic`, the activation's quantization
    range is *calibrated*: fixed ahead of time from ``calibration_data``
    (falling back to :func:`generate_random_calibration_data` when omitted)
    rather than recomputed on every inference. A
    QuantizeLinear/DequantizeLinear pair is inserted around each quantized
    tensor (the "QDQ" format): the graph still computes in float32, ready for
    a QDQ-aware runtime to fuse the pattern into a true integer kernel at
    load time.

    :param model: onnx ModelProto object or file path
    :param calibration_data: representative input batches to calibrate
            activation ranges from. Each batch is a ``{input_name: np.ndarray}``
            dict matching the model's graph inputs -- see
            :func:`generate_random_calibration_data` (the default, a quick
            smoke test) and :func:`load_huggingface_calibration_data` (real
            data, a much better calibration source for real deployment).
    :param num_calibration_samples: number of random batches to generate when
            ``calibration_data`` is not supplied
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run calibration on
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_calibration_samples, seed=seed
        )
    ranges = calibrate(model, calibration_data, providers=providers)
    return onnx.load_from_string(C.quantize_static(model.SerializeToString(), ranges))
