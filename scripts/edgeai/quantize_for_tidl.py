#!/usr/bin/env python3
"""Produce a TIDL-scheme-matching QDQ model with onnxsim's own quantizer.

edgeai-tidl-tools' own quantization docs (``docs/quantization.md``) list
three ways to get a fixed-point model into TIDL: its own PTQ (calibration
run by the real compiler itself), a pre-quantized ONNX QDQ model
(``advanced_options:prequantized_model=1``), or a hand-written
"quantization proto" bypassing calibration with externally supplied
min/max ranges. This module is about the second path, using
``onnxsim.calibration.quantize_static``/``quantize_static_int16`` (already
in this repo's own package) as the "external quantization algorithm" --
edgeai-tidl-tools' own docs name using one as a first-class use case for
pre-quantized models ("Pre-quantized Models... Using your own quantization
algorithms").

**Structurally, the match is exact.** Verified directly against
``docs/quantization.md``'s per-layer quantization-restrictions table for
``TIDL_ConvolutionLayer``/``TIDL_InnerProductLayer`` (the layers
``quantize_static`` targets -- Conv, MatMul, "vanilla" Gemm):
``quantize_static``'s weight ``DequantizeLinear`` is symmetric
(``zero_point`` all-zero) and per-channel (``axis=0``, the output-channel
axis, one scale per output channel); its activation ``QuantizeLinear``/
``DequantizeLinear`` is per-tensor (scalar scale/zero-point) with a
non-zero ``zero_point`` (asymmetric) -- exactly "Weights: Symmetric,
Per-channel" / "Activations: Asymmetric, Per-tensor" from that table.
``check_tidl_qdq_scheme()`` below checks this on any model, not just this
module's own output.

**Running it through the real compiler is a different story, confirmed by
actually trying it (not assumed):** feeding ``quantize_static``'s QDQ
output back to the real ``TIDLCompilationProvider`` with
``advanced_options:prequantized_model=1`` **segfaults the x86 PC compiler**
in the `tidl_tools` release this was tested against (``11_02_20_00``) --
reproduced on both a `Conv`-only and a separate `MatMul`-only model (so not
an op-specific or onnxsim-specific-shape issue), always at the same point
in the log (`TIDL_runtimesOptimizeNet`, right after
"`[Optimization for subgraph_N Started]`"). The same exact models compile
fine through TIDL's *own* PTQ path (the plain float model,
`advanced_options:prequantized_model` left at its default 0) -- see
``scripts/edgeai/real_compile.py``. A parallel attempt at the third
path -- `advanced_options:quant_params_proto_path` in "write mode", meant
to dump a skeleton file with TIDL's own internal layer names to then
overwrite with better-calibrated ranges -- did not produce the requested
file in testing either (only an always-empty,
differently-named debug artifact under `tempDir`); unlike the segfault,
this one is inconclusive rather than confirmed broken, so it isn't wired
up here.

**So, until TI's compiler handles this without crashing:** use this module
to produce and structurally validate a QDQ model (for inspection, for
another QDQ-aware runtime, or for a future/patched `tidl_tools` release),
but compile via TIDL's own calibration on the *float* model for now --
plain ``tensor_bits: 8``/``16`` provider options on the model
``onnxsim.simplify()`` (not ``quantize_for_tidl()``) produced, per
``real_compile.py``'s existing, confirmed-working path.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import onnx
from onnx import numpy_helper

PRECISIONS = ("int8", "int8_a16")


def quantize_for_tidl(
    model: onnx.ModelProto,
    precision: str = "int8",
    calibration_data: Optional[Sequence[dict]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    method: str = "minmax",
) -> onnx.ModelProto:
    """Quantize `model` to the QDQ form matching TIDL's documented scheme.

    :param precision: ``"int8"`` (INT8 weight, per-channel symmetric; UINT8
        activation, per-tensor asymmetric -- via
        :func:`onnxsim.calibration.quantize_static`) or ``"int8_a16"``
        (same weight scheme, UINT16 activation -- via
        :func:`onnxsim.calibration.quantize_static_int16`, useful for an
        activation TIDL's own 16-bit mixed-precision path would also
        widen, though this module does not itself drive that path -- see
        this module's docstring).
    :param calibration_data: forwarded to ``quantize_static*`` --
        representative input batches; random data is generated when
        omitted (see that function for why real data calibrates better).
    :param method: forwarded to ``quantize_static*``'s own ``calibrate()``
        call -- ``"minmax"`` (default), ``"entropy"``, or ``"mse"``.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")

    from onnxsim.calibration import quantize_static, quantize_static_int16

    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    if precision == "int8_a16":
        # quantize_static_int16 needs opset >= 21 (uint16 Quantize/
        # DequantizeLinear) -- confirmed: below that, it silently returns
        # the model unquantized rather than raising, so this bump is not
        # optional plumbing, it's what makes the call actually do anything.
        for opset in model.opset_import:
            if opset.domain == "" and opset.version < 21:
                bumped = onnx.ModelProto()
                bumped.CopyFrom(model)
                for bumped_opset in bumped.opset_import:
                    if bumped_opset.domain == "":
                        bumped_opset.version = 21
                model = bumped
                break

    quantize_fn = quantize_static if precision == "int8" else quantize_static_int16
    return quantize_fn(
        model,
        calibration_data=calibration_data,
        num_calibration_samples=num_calibration_samples,
        seed=seed,
        method=method,
    )


def _initializer_arrays(model: onnx.ModelProto) -> dict:
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def _dq_axis(node: onnx.NodeProto) -> int:
    for attr in node.attribute:
        if attr.name == "axis":
            return attr.i
    return 1  # ONNX default for QuantizeLinear/DequantizeLinear


def check_tidl_qdq_scheme(model: onnx.ModelProto) -> List[str]:
    """Check a QDQ model against `docs/quantization.md`'s per-layer table.

    Only checks the two constraints that table states unconditionally for
    a quantized Conv/Gemm/MatMul weight and activation (see this module's
    docstring) -- not the full per-layer table, and only for
    initializer-backed scale/zero-point (a `DequantizeLinear` whose scale
    comes from a `Constant` node rather than an initializer is not
    checked). Returns human-readable violations; empty does not mean the
    model is otherwise TIDL-correct, only that these two checks passed.
    """
    issues = []
    initializer_names = {init.name for init in model.graph.initializer}
    arrays = _initializer_arrays(model)

    for node in model.graph.node:
        if node.op_type != "DequantizeLinear" or len(node.input) < 3:
            continue
        data_name, scale_name, zp_name = node.input[0], node.input[1], node.input[2]
        if scale_name not in arrays or zp_name not in arrays:
            continue
        scale, zero_point = arrays[scale_name], arrays[zp_name]
        label = node.name or data_name
        is_weight = data_name in initializer_names

        if is_weight:
            if scale.size > 1 and _dq_axis(node) != 0:
                issues.append(
                    f"{label}: per-channel weight DequantizeLinear should use "
                    f"axis=0 (output-channel), got axis={_dq_axis(node)}"
                )
            if np.any(zero_point != 0):
                issues.append(
                    f"{label}: weight DequantizeLinear should be symmetric "
                    f"(zero_point all 0), got {zero_point}"
                )
        elif scale.size > 1:
            issues.append(
                f"{label}: activation DequantizeLinear should be per-tensor "
                f"(scalar scale), got shape {scale.shape}"
            )
    return issues
