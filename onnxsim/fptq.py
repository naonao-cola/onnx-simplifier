"""FPTQ (Li, Zhang, Li, Yao, Zhang, Chu, Sun, Du and Xie, 2023, "FPTQ:
Fine-grained Post-Training Quantization for Large Language Models",
https://arxiv.org/abs/2308.15987). onnxsim ports the *algorithm*, not any
framework's code, per the same rationale as :mod:`onnxsim.smoothquant`/
:mod:`onnxsim.awq`/:mod:`onnxsim.gptq` (FPTQ's own reference implementation
quantizes live PyTorch models with no ONNX export path).

FPTQ targets the same W4A8 recipe :mod:`onnxsim.smoothquant` is meant to
feed (weight-only INT4 for I/O bandwidth, INT8 activations for fast integer
matmul), and shares that module's core mechanism -- for a MatMul/Gemm
``Y = X @ W``, dividing one activation channel by a per-channel scale ``s_j``
and multiplying the matching weight row by the same ``s_j`` leaves ``Y``
exactly unchanged (``(X / s) @ (W * s) == X @ W``) while moving quantization
difficulty from the activation into the weight. Where FPTQ differs is *how*
``s`` is chosen on the layers where :mod:`onnxsim.smoothquant`'s own
power-law scale (``s_j = max(|X_j|) ** alpha / max(|W_j|) ** (1 - alpha)``)
falls short: the paper reports that a handful of layers have activation
channels so much larger than the rest that fully equalizing them the
power-law way forces a correspondingly enormous compensating blow-up on the
matching weight row, which then itself becomes hard to quantize -- moving
the difficulty rather than actually reducing it. FPTQ's fix for exactly
these "intractable" layers is what its own abstract calls "a novel
logarithmic equalization": instead of scaling a channel down in proportion
to its own raw magnitude (SmoothQuant's ``alpha=1`` case), scale it down in
proportion to the *logarithm* of how far above the layer's typical channel
magnitude it sits --

    ref_j = geometric mean of max(|X_j'|) over every channel j' in the layer
    ratio_j = max(|X_j|) / ref_j
    s_j = ref_j * log2(1 + ratio_j)

-- so an ordinary channel (``ratio_j`` close to 1) is left close to
untouched (``log2(2) = 1``), while a 1000x outlier channel is pulled down to
roughly ``log2(1001) ~= 10x`` its typical neighbor rather than the full
1000x :mod:`onnxsim.smoothquant`'s own ``alpha=1`` would demand -- most of
the outlier's difficulty is genuinely absorbed rather than only relocated,
at the cost of leaving the migrated activation channel itself less
perfectly uniform than a full linear equalization would. This module
applies that logarithmic scale only to layers it detects as "intractable"
this way (``max_j(max(|X_j|)) / ref`` exceeding ``outlier_ratio_threshold``,
i.e. some channel sits at least that many times above the layer's typical
channel scale); every other ("tractable") layer keeps
:mod:`onnxsim.smoothquant`'s own power-law scale unchanged, matching the
paper's own description of combining both strategies layer-by-layer rather
than picking one globally.

Like :mod:`onnxsim.smoothquant`/:mod:`onnxsim.outlier_suppression`, this
only performs the *migration*: :func:`apply_fptq` returns a float model,
provably equivalent to the input up to floating-point rounding -- no
quantization happens here. The result is meant to be fed to onnxsim's own
W4 weight-only quantizer (e.g. :func:`onnxsim.quantize_weight_only_int4`)
and an A8 activation quantizer (e.g. :func:`onnxsim.quantize_static`)
afterwards to realize the paper's actual W4A8 recipe -- this module does not
wire those up itself, the same way :mod:`onnxsim.smoothquant` doesn't.
Deliberately not ported: FPTQ's other contribution, layerwise knowledge
distillation to recover LayerNorm/activation-function accuracy lost to
quantization -- a full fine-tuning loop, out of scope for this repo's
graph-rewrite-only quantization passes (see :mod:`onnxsim.adaround`/
:mod:`onnxsim.autoround`/:mod:`onnxsim.finetune` for onnxsim's existing,
separate fine-tuning-based passes, none of which currently target this
specific LayerNorm/activation recovery).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.smoothquant import _match_matmul_like


def apply_fptq(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    alpha: float = 0.5,
    outlier_ratio_threshold: float = 10.0,
    epsilon: float = 1e-5,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Migrates activation quantization difficulty into the weight for
    every MatMul/vanilla-Gemm layer with a constant 2-D float32 weight and a
    plain 2-D activation input, using real calibration activations --
    :mod:`onnxsim.smoothquant`'s own power-law scale for most layers, and
    FPTQ's logarithmic equalization for layers whose activation has a
    channel at least ``outlier_ratio_threshold`` times the layer's typical
    channel scale. See this module's own docstring for the technique.
    Returns a float model -- pass the result to onnxsim's own weight-only
    INT4 quantizer and a W8A8 activation quantizer (e.g.
    :func:`onnxsim.quantize_weight_only_int4` then
    :func:`onnxsim.quantize_static`) to realize the paper's actual W4A8
    recipe.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            input channel's activation range on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative migration than random input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param alpha: the migration strength used on "tractable" layers --
            identical in meaning to :func:`onnxsim.apply_smoothquant`'s own
            ``alpha``
    :param outlier_ratio_threshold: a layer is classified "intractable"
            (and gets the logarithmic scale instead of the power-law one)
            when its largest per-channel activation max is at least this
            many times its per-channel geometric-mean activation max
    :param epsilon: floor applied to every per-channel activation/weight
            max-abs value before computing a scale, avoiding a divide-by-
            zero on an all-zero channel
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight columns rescaled
            in place and a new ``Mul`` node inserted before it applying the
            inverse scale to its activation input; layers with a
            non-constant, non-2-D weight, or whose activation input isn't a
            plain 2-D tensor matching the weight's reduction dimension, are
            left untouched
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)

    nodes = list(graph.node)
    candidates = []
    for node in nodes:
        match = _match_matmul_like(node)
        if match is None:
            continue
        x_name, w_name, weight_transposed = match
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue
        candidates.append((node, x_name, w_name, weight_transposed))

    if not candidates:
        return out

    probe_names = sorted({x_name for _, x_name, _, _ in candidates})
    probe_model = _add_probe_outputs(out, probe_names)

    act_absmax: Dict[str, np.ndarray] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            if x.ndim != 2:
                continue
            m = np.abs(x).max(axis=0)
            act_absmax[name] = (
                m if name not in act_absmax else np.maximum(act_absmax[name], m)
            )

    for node, x_name, w_name, weight_transposed in candidates:
        acts = act_absmax.get(x_name)
        if acts is None:
            continue  # never observed as a plain 2-D tensor; skip

        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if weight_transposed else w.T  # [N, K], output channel first
        k = w_nk.shape[1]
        if acts.shape[0] != k:
            continue  # activation's feature dim doesn't match K; skip

        act_channel = np.maximum(acts, epsilon)
        weight_channel = np.maximum(np.abs(w_nk).max(axis=0), epsilon)  # [K]

        ref = float(np.exp(np.mean(np.log(act_channel))))  # geometric mean
        ref = max(ref, epsilon)
        outlier_ratio = float(act_channel.max() / ref)

        if outlier_ratio >= outlier_ratio_threshold:
            ratio = act_channel / ref
            s = ref * np.log2(1.0 + ratio)
        else:
            s = (act_channel**alpha) / (weight_channel ** (1.0 - alpha))
        s = np.maximum(s, epsilon)

        w_new_nk = w_nk * s[np.newaxis, :]
        w_new = w_new_nk if weight_transposed else w_new_nk.T
        w_new = w_new.reshape(dim0, dim1).astype(np.float32)
        w_init.CopyFrom(onnx.numpy_helper.from_array(w_new, name=w_name))

        inv_s = (1.0 / s).astype(np.float32)
        scale_name = _unique_name(f"{x_name}_fptq_inv_scale", taken_names)
        graph.initializer.append(onnx.numpy_helper.from_array(inv_s, name=scale_name))
        scaled_name = _unique_name(f"{x_name}_fptq_scaled", taken_names)
        mul_node = onnx.helper.make_node(
            "Mul",
            [x_name, scale_name],
            [scaled_name],
            name=_unique_name(f"{x_name}_fptq_mul", taken_names),
        )
        node_idx = next(i for i, n in enumerate(graph.node) if n is node)
        graph.node.insert(node_idx, mul_node)
        node.input[0] = scaled_name

    return out
