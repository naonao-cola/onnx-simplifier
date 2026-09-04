"""Outlier Suppression+ (Wei et al., 2023, "Outlier Suppression+: Accurate
quantization of large language models by equivalent and optimal shifting and
scaling", EMNLP 2023, https://arxiv.org/abs/2304.09145). Its predecessor,
Outlier Suppression (the same authors' earlier NeurIPS 2022 paper), is
distinct and not what this module ports. onnxsim ports the *algorithm*, not
any framework's code, per the same rationale as :mod:`onnxsim.smoothquant`.

:mod:`onnxsim.smoothquant` migrates per-channel activation quantization
difficulty into the weight via a single elementwise *scale* -- ``s_j``
shrinks activation channel ``j``'s range at the cost of expanding weight row
``j``'s range. That works well for channels whose outliers are large in
*magnitude* but roughly *symmetric* around zero. Outlier Suppression+'s
observation: many transformer activation channels (especially post-LayerNorm)
carry a large, consistent *asymmetric* component too -- a channel sitting
mostly on one side of zero -- which a symmetric scale cannot address at all
(scaling a lopsided range by any positive constant leaves it just as
lopsided), forcing a symmetric quantizer's range to cover the channel's full
excursion from its own worst-case value down to (or up from) zero.

Outlier Suppression+ therefore adds a **channel-wise shift** ahead of
SmoothQuant's own scale: for each input channel ``j``,

    z_j = (max(X_j) + min(X_j)) / 2

(the midpoint of that channel's observed calibration range) is subtracted
from ``X_j`` before scaling, re-centering it around zero and roughly halving
the range a symmetric quantizer must cover for a channel that was originally
one-sided. Shifting an affine layer's input is not, on its own, a free
transformation the way scaling is -- ``(X - z) @ W`` differs from ``X @ W``
by exactly ``z @ W``, a *constant* (calibration-independent, per-output-
-channel) vector -- so this module folds that constant back in as an
additive correction on the layer's own output, via a new ``Add`` node
inserted right after it (the same "measure a per-channel constant, fold it
back in with an ``Add``" mechanics :func:`onnxsim.correct_bias` already
uses, though what's being corrected for here is an exact algebraic identity
from the shift, not an empirically-measured quantization error). The result:
``Y = ((X - z) / s) @ (W * s)^T + z @ W^T`` is *exactly* ``X @ W^T`` for any
choice of ``z``/``s`` (up to floating-point rounding) -- provably so, not
just approximately, since every step is a linear re-parameterization of the
same affine map, never an approximation.

For the scaling step itself, this module reuses
:mod:`onnxsim.smoothquant`'s own closed-form, alpha-parameterized formula
(``s_j = max(|X_j - z_j|) ** alpha / max(|W_j|) ** (1 - alpha)``), applied to
the now-*shifted* activation rather than the raw one, matching
:mod:`onnxsim.smoothquant`'s own documented practice of using a single fixed
``alpha`` rather than a per-layer search. The Outlier Suppression+ paper
additionally proposes its own iterative grid refinement on top of that
formula to squeeze out a further, typically small, improvement over the
plain closed form -- that refinement is not reproduced here; what onnxsim
ports is the paper's headline structural contribution over SmoothQuant (the
shift), not its secondary scale-search refinement.

Like :func:`onnxsim.apply_smoothquant`, this only performs the *migration*:
:func:`apply_outlier_suppression_plus` returns a float model, provably
equivalent to the input up to floating-point rounding -- no quantization
happens here. The result is meant to be fed to a W8A8 quantizer afterwards
(e.g. :func:`onnxsim.quantize_static`/:func:`onnxsim.quantize_qoperator_gemm`),
exactly how :func:`onnxsim.apply_smoothquant` is used.
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


def apply_outlier_suppression_plus(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    alpha: float = 0.5,
    epsilon: float = 1e-5,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies Outlier Suppression+'s channel-wise shifting and scaling to
    every MatMul/vanilla-Gemm layer with a constant 2-D float32 weight and a
    plain 2-D activation input, using real calibration activations. See this
    module's own docstring for the technique. Returns a float model -- pass
    the result to a W8A8 quantizer (e.g. :func:`onnxsim.quantize_static`) to
    actually quantize it.

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
    :param alpha: the scaling step's migration strength, identical in
            meaning to :func:`onnxsim.apply_smoothquant`'s own ``alpha``
            (applied to the *shifted* activation's range rather than the
            raw activation's)
    :param epsilon: floor applied to every per-channel activation/weight
            max-abs value before computing the scale, avoiding a divide-by-
            zero on an all-zero (or exactly-centered) channel
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight columns rescaled
            in place, a new ``Mul``+``Sub`` pair inserted before it applying
            the shift and scale to its activation input, and a new ``Add``
            inserted after it restoring the shift's constant contribution to
            the output; layers with a non-constant, non-2-D weight, or whose
            activation input isn't a plain 2-D tensor matching the weight's
            reduction dimension, are left untouched
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

    act_max: Dict[str, np.ndarray] = {}
    act_min: Dict[str, np.ndarray] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            if x.ndim != 2:
                continue
            mx, mn = x.max(axis=0), x.min(axis=0)
            act_max[name] = mx if name not in act_max else np.maximum(act_max[name], mx)
            act_min[name] = mn if name not in act_min else np.minimum(act_min[name], mn)

    for node, x_name, w_name, weight_transposed in candidates:
        mx = act_max.get(x_name)
        mn = act_min.get(x_name)
        if mx is None:
            continue  # never observed as a plain 2-D tensor; skip

        w_init = initializer_map[w_name]
        w_nk_orig = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        dim0, dim1 = w_nk_orig.shape
        w_nk = w_nk_orig if weight_transposed else w_nk_orig.T  # [N, K]
        k = w_nk.shape[1]
        if mx.shape[0] != k:
            continue  # activation's feature dim doesn't match K; skip

        # Channel-wise shift: recenters each channel around zero. The
        # shifted channel's own max-abs is exactly half its observed
        # range -- no second pass over calibration data needed.
        z = (mx + mn) / 2.0  # [K]
        shifted_absmax = np.maximum((mx - mn) / 2.0, epsilon)  # [K]

        weight_channel = np.maximum(np.abs(w_nk).max(axis=0), epsilon)  # [K]
        s = (shifted_absmax**alpha) / (weight_channel ** (1.0 - alpha))
        s = np.maximum(s, epsilon)

        # Exact algebraic fold: Y = ((X - z) / s) @ (W * s)^T + z @ W^T,
        # using the *original* (unscaled) weight for the correction term --
        # see this module's own docstring.
        correction = (w_nk @ z).astype(np.float32)  # [N]

        w_smooth_nk = w_nk * s[np.newaxis, :]
        w_new = w_smooth_nk if weight_transposed else w_smooth_nk.T
        w_new = w_new.reshape(dim0, dim1).astype(np.float32)
        w_init.CopyFrom(onnx.numpy_helper.from_array(w_new, name=w_name))

        z_name = _unique_name(f"{x_name}_os_plus_shift", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(z.astype(np.float32), name=z_name)
        )
        shifted_name = _unique_name(f"{x_name}_os_plus_shifted", taken_names)
        sub_node = onnx.helper.make_node(
            "Sub",
            [x_name, z_name],
            [shifted_name],
            name=_unique_name(f"{x_name}_os_plus_sub", taken_names),
        )

        inv_s = (1.0 / s).astype(np.float32)
        scale_name = _unique_name(f"{x_name}_os_plus_inv_scale", taken_names)
        graph.initializer.append(onnx.numpy_helper.from_array(inv_s, name=scale_name))
        scaled_name = _unique_name(f"{x_name}_os_plus_scaled", taken_names)
        mul_node = onnx.helper.make_node(
            "Mul",
            [shifted_name, scale_name],
            [scaled_name],
            name=_unique_name(f"{x_name}_os_plus_mul", taken_names),
        )

        node_idx = next(i for i, n in enumerate(graph.node) if n is node)
        graph.node.insert(node_idx, mul_node)
        graph.node.insert(node_idx, sub_node)
        node.input[0] = scaled_name

        # Restore the shift's constant contribution via a new Add right
        # after the layer's own output, renaming its output to a fresh
        # internal name (mirroring onnxsim.bias_correction._apply_correction)
        # so every existing downstream consumer keeps working unmodified.
        node_idx = next(i for i, n in enumerate(graph.node) if n is node)
        pre_name = _unique_name(f"{node.output[0]}_os_plus_pre_shift", taken_names)
        original_output = node.output[0]
        node.output[0] = pre_name
        correction_name = _unique_name(
            f"{original_output}_os_plus_correction", taken_names
        )
        graph.initializer.append(
            onnx.numpy_helper.from_array(correction, name=correction_name)
        )
        add_node = onnx.helper.make_node(
            "Add",
            [pre_name, correction_name],
            [original_output],
            name=_unique_name(f"{original_output}_os_plus_add", taken_names),
        )
        graph.node.insert(node_idx + 1, add_node)

    return out
