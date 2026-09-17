"""Embedding-output quantization for retrieval encoder models -- see
Mixedbread's "Asymmetric Quantization: Near-Lossless Late-Interaction
Retrieval with 97% Storage Reduction" (mixedbread.com/blog/asymmetric-quant)
and its predecessor "Binary and Scalar Embedding Quantization for
Significantly Faster & Cheaper Retrieval" (huggingface.co/blog/embedding-
quantization, Mixedbread + Hugging Face).

Every other quantizer in onnxsim rewrites how a model *computes* --
weights, KV cache, attention. This one is a different kind of quantizer:
it compresses what the model's own graph *emits*, for downstream storage
in a vector index rather than for cheaper on-device math. Retrieval
systems that embed both a query and a very large number of documents
naturally want **asymmetric** precision: keep the (single, per-query) query
vector at higher precision for accurate scoring, and compress the (many,
stored-forever) document vectors as hard as the accuracy budget allows --
Mixedbread's own numbers: int8 query against binary documents keeps 89.65
NDCG@10 versus 90.26 for full float32, at roughly 32x less per-document
storage. Call :func:`quantize_embedding_int8` on a query-encoder export and
:func:`quantize_embedding_binary` on a document-encoder export (whether
that's two literally separate ONNX exports, or the same encoder exported
twice under each name) to reproduce that asymmetry.

Both functions append the quantization step directly to the graph's own
declared output -- rebinding the output name to a freshly quantized tensor
computed with ordinary ONNX ops, changing that output's own dtype/shape --
so a deployed encoder model emits already-compressed vectors at inference
time, with no separate post-processing step downstream.
"""

from __future__ import annotations

from typing import Optional, Sequence, Set, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.onnx_simplifier import quantize_embedding_binary_cpp


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


def _resolve_output(
    graph: onnx.GraphProto, output_name: Optional[str]
) -> Optional[int]:
    """Returns the index into ``graph.output`` of the float32 tensor to
    quantize, or ``None`` if it can't be resolved unambiguously: an explicit
    ``output_name`` must name an existing float32 output; omitting it
    requires the graph to have *exactly one* float32 output (declining
    rather than guessing which one is the embedding, if there's more than
    one).
    """
    if output_name is not None:
        for i, o in enumerate(graph.output):
            if o.name == output_name:
                if o.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
                    return None
                return i
        return None

    float_outputs = [
        i
        for i, o in enumerate(graph.output)
        if o.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    ]
    return float_outputs[0] if len(float_outputs) == 1 else None


def quantize_embedding_binary(
    model: Union[str, onnx.ModelProto],
    output_name: Optional[str] = None,
) -> onnx.ModelProto:
    """Binarizes a model's own embedding output at inference time: each
    element is thresholded at zero (``1`` if greater than zero, else
    ``0``, matching Mixedbread/Hugging Face's own scheme -- see this
    module's docstring), then 8 consecutive elements along the last axis
    are packed into one ``uint8`` byte MSB-first, exactly
    ``numpy.packbits(x > 0, axis=-1, bitorder="big")`` -- a 32x reduction
    in output size, and needs no calibration data at all.

    Delegates to :func:`onnxsim.quantize_embedding_binary_cpp` (the
    verified C++ port); this pure-Python name is kept only for backward
    compatibility with existing callers.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param output_name: which graph output to binarize; if omitted, the
            graph must have exactly one float32 output (declining
            otherwise, rather than guessing)
    :returns: ``model`` with the resolved output's dtype changed to
            ``uint8`` and its last dimension divided by 8; a model whose
            output can't be resolved, whose last dimension isn't known
            statically, or isn't a multiple of 8, is returned unchanged
    """
    return quantize_embedding_binary_cpp(model, output_name)


def quantize_embedding_int8(
    model: Union[str, onnx.ModelProto],
    output_name: Optional[str] = None,
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Quantizes a model's own embedding output to INT8 at inference time:
    symmetric, one scale for the whole tensor (matching Mixedbread/Hugging
    Face's own "scalar quantization" -- see this module's docstring),
    calibrated once from representative data. A 4x reduction in output
    size -- the precision this module's own docstring recommends for a
    retrieval query vector, paired with :func:`quantize_embedding_binary`
    on the (far more numerous, storage-dominant) document side.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param output_name: which graph output to quantize; if omitted, the
            graph must have exactly one float32 output (declining
            otherwise, rather than guessing)
    :param calibration_data: representative input batches (each a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs) to calibrate the scale on -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to calibrate on
    :returns: ``model`` with the resolved output's dtype changed to
            ``int8`` (same shape); a model whose output can't be resolved
            is returned unchanged
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if not _has_min_opset(model, 13):
        return model

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    idx = _resolve_output(graph, output_name)
    if idx is None:
        return out
    target = graph.output[idx]
    x = target.name

    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    probe = _add_probe_outputs(model, [x])
    absmax = 0.0
    for batch in calibration_data:
        outputs = backend.run_model(probe, batch, providers=providers)
        arr = np.asarray(outputs[x], dtype=np.float64)
        if arr.size == 0:
            continue
        absmax = max(absmax, float(np.abs(arr).max()))
    scale = np.array(max(absmax, 1e-12) / 127.0, dtype=np.float32)

    taken_names: Set[str] = _all_names(graph)
    prefix = f"{x}_i8"
    scale_name = _unique_name(f"{prefix}_scale", taken_names)
    graph.initializer.append(onnx.numpy_helper.from_array(scale, name=scale_name))
    zp_name = _unique_name(f"{prefix}_zero_point", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(np.array(0, dtype=np.int8), name=zp_name)
    )
    quantized_name = _unique_name(f"{prefix}_quantized", taken_names)
    graph.node.append(
        onnx.helper.make_node(
            "QuantizeLinear",
            [x, scale_name, zp_name],
            [quantized_name],
            name=_unique_name(f"{prefix}_quantize_node", taken_names),
        )
    )

    target.name = quantized_name
    target.type.tensor_type.elem_type = onnx.TensorProto.INT8

    return out
