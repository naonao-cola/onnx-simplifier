"""Outlier Suppression (Wei, Zhang, Zhang, Gong, Zhang, Zhang, Chi, Yuan and
Liu, 2022, "Outlier Suppression: Pushing the Limit of Low-bit Transformer
Language Models", NeurIPS 2022, https://arxiv.org/abs/2209.13325). The
original paper this repo's own :mod:`onnxsim.outlier_suppression_plus`
extends -- that module's own docstring already flags this earlier paper as
"distinct and not what this module ports"; this module ports it.

:mod:`onnxsim.smoothquant` and :mod:`onnxsim.outlier_suppression_plus` both
migrate activation quantization difficulty into the weight via a per-channel
scale, realized by inserting a new ``Mul`` node (and, for OS+, ``Sub``/``Add``
too) right before the consuming MatMul/Gemm. Outlier Suppression's own
"Gamma Migration" does the same *kind* of per-channel scale migration, but
realizes it completely differently when the activation being scaled is a
``LayerNormalization``'s own output (transformers' by far most common
producer of a Linear layer's input): instead of inserting a node, it folds
the scale directly into the LayerNormalization's own affine parameters,
adding **zero** runtime nodes at all.

The algebra: ``LayerNormalization`` computes
``out = normalize(x) * gamma + beta`` (elementwise per channel, ``beta``
optional). Dividing the *whole* affine output by a per-channel scale ``s``
is itself just

    out / s = normalize(x) * (gamma / s) + (beta / s)

-- i.e. exactly what a LayerNormalization with ``gamma' = gamma / s`` and
``beta' = beta / s`` already computes, with no new node needed. Scaling a
downstream consumer's weight rows by the same ``s`` (:mod:`onnxsim.
smoothquant`'s own compensating step) then makes the composition exact,
identical in spirit to how :mod:`onnxsim.smoothquant` compensates its own
inserted ``Mul``, but here the "insertion" costs nothing at runtime because
the LayerNormalization node already existed and already had to compute an
affine transform anyway.

This is a narrower structural target than :mod:`onnxsim.smoothquant`'s own
"any MatMul/Gemm with a 2-D activation input" -- gamma migration is only
correct when the LayerNormalization's output has **no other consumer**
besides the MatMul/Gemm layers being compensated (dividing its output by
``s`` changes what *every* consumer of that tensor sees, and an
uncompensated consumer -- e.g. a residual ``Add``, or the LayerNormalization
output being a graph output itself -- would silently see the wrong,
scaled-down activation). This module therefore only migrates a
``LayerNormalization`` node whose output feeds exclusively into one or more
plain MatMul/vanilla-Gemm nodes (as their activation input) and is not
itself a graph output -- exactly the shape a transformer's own QKV or MLP
input projection takes (one shared pre-projection ``LayerNorm`` feeding
several parallel `Linear`s, or one feeding a single `Linear`, and nothing
else), which is also the paper's own primary target. A LayerNormalization
with any other kind of consumer is left completely untouched, not partially
migrated.

The per-channel scale itself reuses :mod:`onnxsim.smoothquant`'s own
closed-form, alpha-parameterized formula (``s_j = max(|X_j|) ** alpha /
max(|W_j|) ** (1 - alpha)``, maximized over every compensated consumer's own
weight when there is more than one), matching that module's own documented
practice of a single fixed ``alpha`` rather than a per-layer search.
Deliberately not ported: the paper's own second contribution, "Token-Wise
Clipping" (a searched, rather than plain min-max/entropy, activation
clipping range for the subsequent quantizer) -- a calibration-range-search
technique orthogonal to this module's own scale migration, and out of this
module's own scope (:func:`onnxsim.calibrate`'s existing ``"minmax"``/
``"entropy"`` methods already cover the "how to pick a clip range" question
this repo answers elsewhere).

Like :mod:`onnxsim.smoothquant`/:mod:`onnxsim.outlier_suppression_plus`,
this only performs the *migration*: :func:`apply_outlier_suppression`
returns a float model, provably equivalent to the input up to floating-point
rounding -- no quantization happens here. The result is meant to be fed to a
W8A8 quantizer afterwards (e.g. :func:`onnxsim.quantize_static`).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.smoothquant import _match_matmul_like


def apply_outlier_suppression(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    alpha: float = 0.5,
    epsilon: float = 1e-5,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies Outlier Suppression's "Gamma Migration" to every
    ``LayerNormalization`` node whose output feeds exclusively into one or
    more plain MatMul/vanilla-Gemm layers (and is not itself a graph
    output), using real calibration activations. See this module's own
    docstring for the technique. Returns a float model -- pass the result
    to a W8A8 quantizer (e.g. :func:`onnxsim.quantize_static`) to actually
    quantize it.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            LayerNormalization output channel's activation range on. Each
            batch is a ``{input_name: np.ndarray}`` dict matching
            ``model``'s graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and :func:`onnxsim.load_huggingface_calibration_data`
            (real data, a much more representative migration than random
            input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param alpha: the migration strength, identical in meaning to
            :func:`onnxsim.apply_smoothquant`'s own ``alpha``
    :param epsilon: floor applied to every per-channel activation/weight
            max-abs value before computing the scale, avoiding a divide-by-
            zero on an all-zero channel
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched ``LayerNormalization``'s
            ``scale``/``bias`` initializers divided by the migration scale
            in place, and every compensated consumer's weight rows
            multiplied by the same scale in place -- no new nodes are ever
            inserted. A ``LayerNormalization`` with any consumer other than
            a plain MatMul/vanilla-Gemm (as its activation input), or whose
            output is itself a graph output, is left completely untouched.
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
    graph_output_names = {o.name for o in graph.output}

    candidates = []  # (ln_node, gamma_init, beta_init_or_None, consumers)
    for ln in graph.node:
        if ln.op_type != "LayerNormalization" or len(ln.input) < 2:
            continue
        gamma_init = initializer_map.get(ln.input[1])
        if (
            gamma_init is None
            or gamma_init.data_type != onnx.TensorProto.FLOAT
            or len(gamma_init.dims) != 1
        ):
            continue
        beta_init = None
        if len(ln.input) >= 3 and ln.input[2]:
            beta_init = initializer_map.get(ln.input[2])
            if beta_init is None or list(beta_init.dims) != list(gamma_init.dims):
                continue  # malformed/non-constant bias -- decline, don't guess

        ln_out = ln.output[0]
        if ln_out in graph_output_names:
            continue  # an external consumer would see the scaled-down value

        consumers = []
        declined = False
        for node in graph.node:
            if ln_out not in node.input:
                continue
            match = _match_matmul_like(node)
            if match is None:
                declined = True
                break
            x_name, w_name, weight_transposed = match
            if x_name != ln_out:
                declined = True  # ln_out feeds a weight/bias slot, not activation
                break
            w_init = initializer_map.get(w_name)
            if (
                w_init is None
                or w_init.data_type != onnx.TensorProto.FLOAT
                or len(w_init.dims) != 2
            ):
                declined = True
                break
            k_dim = w_init.dims[1] if weight_transposed else w_init.dims[0]
            if k_dim != gamma_init.dims[0]:
                declined = True
                break
            consumers.append((node, w_init, weight_transposed))

        if declined or not consumers:
            continue
        candidates.append((ln, gamma_init, beta_init, consumers))

    if not candidates:
        return out

    probe_names = sorted({ln.output[0] for ln, _, _, _ in candidates})
    probe_model = _add_probe_outputs(out, probe_names)

    act_absmax: Dict[str, np.ndarray] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            if x.ndim < 1:
                continue
            m = np.abs(x).max(axis=tuple(range(x.ndim - 1)))
            act_absmax[name] = (
                m if name not in act_absmax else np.maximum(act_absmax[name], m)
            )

    for ln, gamma_init, beta_init, consumers in candidates:
        acts = act_absmax.get(ln.output[0])
        if acts is None or acts.shape[0] != gamma_init.dims[0]:
            continue  # never observed, or a rank/shape mismatch; skip

        weight_channel = np.full(acts.shape, epsilon, dtype=np.float64)
        for _, w_init, weight_transposed in consumers:
            w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
            w_nk = w if weight_transposed else w.T  # [N, K]
            weight_channel = np.maximum(weight_channel, np.abs(w_nk).max(axis=0))  # [K]

        act_channel = np.maximum(acts, epsilon)
        s = (act_channel**alpha) / (weight_channel ** (1.0 - alpha))
        s = np.maximum(s, epsilon)

        gamma = onnx.numpy_helper.to_array(gamma_init).astype(np.float64)
        gamma_init.CopyFrom(
            onnx.numpy_helper.from_array(
                (gamma / s).astype(np.float32), name=gamma_init.name
            )
        )
        if beta_init is not None:
            beta = onnx.numpy_helper.to_array(beta_init).astype(np.float64)
            beta_init.CopyFrom(
                onnx.numpy_helper.from_array(
                    (beta / s).astype(np.float32), name=beta_init.name
                )
            )

        for _, w_init, weight_transposed in consumers:
            w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
            dim0, dim1 = w.shape
            w_nk = w if weight_transposed else w.T  # [N, K]
            w_new_nk = w_nk * s[np.newaxis, :]
            w_new = w_new_nk if weight_transposed else w_new_nk.T
            w_new = w_new.reshape(dim0, dim1).astype(np.float32)
            w_init.CopyFrom(onnx.numpy_helper.from_array(w_new, name=w_init.name))

    return out
