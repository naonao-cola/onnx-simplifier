"""Double quantization (Dettmers et al., 2023, "QLoRA: Efficient Finetuning
of Quantized LLMs", https://arxiv.org/abs/2305.14314, Section 3.2). onnxsim
ports the idea, not any framework's code, per the same rationale as the
rest of this package's PTQ modules -- but unlike every other module here,
this one has no "live weights" to quantize at all: it operates purely on
an *already-quantized* onnx model, as a second pass.

Every block-wise INT4/INT4-like scheme in onnxsim (`quantize_weight_only_int4`,
:mod:`onnxsim.nf4`, :mod:`onnxsim.spqr`, :mod:`onnxsim.spinquant`,
:mod:`onnxsim.quarot`, :mod:`onnxsim.quip_sharp`) stores one float32 scale
per quantization block, alongside the packed low-bit codes -- e.g. a
32-element block of INT4 codes needs 16 bytes of codes plus 4 bytes of
scale, a 25% overhead on top of the codes themselves. QLoRA's own
observation: those scale factors are themselves just numbers with their
own, much milder, dynamic range (they're already absmax-normalized
magnitudes, not raw weights) -- so quantizing *them* too, one more level
down, trades a little more precision for real additional memory savings,
without touching the original codes at all.

This module implements that second level directly: for every
``DequantizeLinear`` node already in the graph whose scale input is a
constant float32 initializer with at least ``min_elements`` values (a
per-block or per-channel scale tensor -- a single scalar per-tensor scale
isn't worth the overhead of a second quantizer around it), the scale
tensor itself is quantized to UINT8 with a single per-tensor meta-scale
(``codes = round(scale / meta_scale)``, ``meta_scale = max(scale) / 255``
-- scale values are always non-negative absmax-derived magnitudes, so a
plain unsigned 0..255 range needs no zero-point offset) and reconstructed
in-graph via a second, nested ``DequantizeLinear(codes_uint8, meta_scale)``
feeding into the original node's own scale input:

    Before:
      Whatever_hat = DequantizeLinear(Codes, Scale, ...)  -- Scale: float32, [blocks, N]

    After:
      ScaleCodes: initializer, uint8, [blocks, N]
      MetaScale: initializer, float32 scalar
      ScaleHat = DequantizeLinear(ScaleCodes, MetaScale)
      Whatever_hat = DequantizeLinear(Codes, ScaleHat, ...)  -- same attributes as before

This is technique-agnostic: it doesn't know or care which module produced
the outer ``DequantizeLinear`` (or its ``axis``/``block_size`` attributes,
left untouched), only that its scale input is a sufficiently large constant
tensor -- so it composes with every block-wise scheme above unchanged, and
with future ones with no new code. The original float32 scale initializer
is left in the graph, unreferenced, exactly like every other onnxsim
rewrite leaves its own now-unused original weight initializer in place
(e.g. :mod:`onnxsim.spinquant`'s original float32 ``W``) -- pair this with
:func:`onnxsim.simplify` (or any generic dead-initializer elimination)
afterward to actually reclaim the storage the original float32 scale used.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name


def apply_double_quantization(
    model: Union[str, onnx.ModelProto],
    min_elements: int = 64,
) -> onnx.ModelProto:
    """Applies QLoRA-style double quantization (see this module's own
    docstring) to every ``DequantizeLinear`` node already present in
    ``model`` whose scale input is a constant float32 tensor with at
    least ``min_elements`` values.

    :param model: an already-quantized onnx ModelProto or file path (the
            output of any onnxsim block-wise/per-channel quantizer, or
            any other model containing ``DequantizeLinear`` nodes)
    :param min_elements: minimum element count a ``DequantizeLinear``
            node's scale tensor must have to be worth double-quantizing;
            a smaller scale tensor (e.g. a single per-tensor scalar) is
            left untouched, since a second quantizer's own overhead
            (a meta-scale initializer plus a new node) would cost more
            than it saves
    :returns: ``model`` with every matched ``DequantizeLinear`` node's
            scale input replaced by a UINT8-quantized-and-dequantized
            reconstruction of the original scale values (see the module
            docstring's diagram); all other node attributes (``axis``,
            ``block_size``, etc.) unchanged. A scale that is not a
            constant initializer (e.g. a dynamically-computed scale, like
            :func:`onnxsim.quantize_kv_cache`'s Value-style per-token
            scale), not float32, or too small, is left untouched; a model
            with no matching node is returned unchanged
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)

    candidates = []
    for node in graph.node:
        if node.op_type != "DequantizeLinear" or len(node.input) < 2:
            continue
        scale_init = initializer_map.get(node.input[1])
        if scale_init is None or scale_init.data_type != onnx.TensorProto.FLOAT:
            continue
        scale = onnx.numpy_helper.to_array(scale_init)
        if scale.size < min_elements:
            continue
        candidates.append((node, scale_init, scale))

    if not candidates:
        return out

    for node, scale_init, scale in candidates:
        scale64 = scale.astype(np.float64)
        meta_scale = max(float(np.abs(scale64).max()), 1e-12) / 255.0
        codes = np.clip(np.round(scale64 / meta_scale), 0, 255).astype(np.uint8)

        prefix = f"{scale_init.name}_dblq"
        codes_name = _unique_name(f"{prefix}_codes", taken_names)
        graph.initializer.append(onnx.numpy_helper.from_array(codes, name=codes_name))
        meta_scale_name = _unique_name(f"{prefix}_meta_scale", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array(meta_scale, dtype=np.float32), name=meta_scale_name
            )
        )

        scale_hat_name = _unique_name(f"{prefix}_scale_hat", taken_names)
        dequant_node = onnx.helper.make_node(
            "DequantizeLinear",
            [codes_name, meta_scale_name],
            [scale_hat_name],
            name=_unique_name(f"{prefix}_scale_hat_node", taken_names),
        )
        node_idx = next(i for i, n in enumerate(graph.node) if n is node)
        graph.node.insert(node_idx, dequant_node)
        node.input[1] = scale_hat_name

    return out
