"""Optional bridge to PPQ (OpenPPL's PyTorch Post-Training Quantization
framework, https://github.com/OpenPPL/ppq -- ``pip install ppq``), for
calibration-based static quantization via PPQ's own calibration/observer
algorithms and quantization scheduler instead of onnxsim's own
:func:`onnxsim.quantize_static`.

**Confirmed, not speculative: the latest PyPI release (``ppq==0.6.6`` at the
time this was written) cannot be imported at all in an environment with the
modern ``onnx``/``protobuf`` versions onnxsim itself requires.** Verified by
directly attempting the import in this repo's own dev environment -- two
independent, unrelated failures, either one fatal on its own:

1. ``import ppq`` -> ``TypeError: Descriptors cannot be created directly.``
   from ``ppq/parser/caffe/ppl_caffe_pb2.py`` -- that file is protoc-
   generated code frozen against a pre-3.19 ``protobuf`` runtime API; modern
   ``protobuf`` (needed by current ``onnx``/``onnxruntime``, both already
   onnxsim dependencies) rejects it outright.
2. Even with ``PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`` set to work
   around (1): ``ppq/parser/onnx_parser.py`` does ``from onnx import
   helper, mapping, numpy_helper`` -- ``onnx.mapping`` was removed from the
   public ``onnx`` package (renamed internal to ``onnx._mapping``) in a
   version newer than PPQ has ever been updated against.

So this module's ``PPQ_AVAILABLE`` will be ``False`` in any environment
that also has a current ``onnx`` installed -- which, since onnxsim itself
requires one, means essentially always right now. **The integration code
below is written directly against PPQ's real, documented top-level API
(``ppq.api.quantize_onnx_model``/``export_ppq_graph``, read from the
installed package's own source) but has NOT been executed end to end in
this session, because PPQ cannot be imported to run it against.** Treat it
as a best-effort bridge for whoever runs this in an environment where PPQ's
own incompatibilities have been worked around (e.g. a separate, older-
``onnx``/``protobuf`` virtualenv) -- not as confirmed-working code the way
the rest of this project's claims are.

PPQ's own quantization execution is PyTorch-based (it JIT-traces the ONNX
graph through its own ``TorchExecutor``, not onnxruntime), so this needs a
real ``torch`` install too, on top of ``ppq`` -- both optional, checked
lazily, same graceful-degradation convention as
``scripts/axera/pulsar2_quantizer.py``'s onnxruntime dependency.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
from typing import Dict, Iterable, Optional

import numpy as np
import onnx

PPQ_AVAILABLE = False
_UNAVAILABLE_REASON: Optional[str] = None

# PPQ prints its own ASCII-art banner as a side effect of `import ppq`, even
# on the failing import this module confirmed is the normal case today (see
# this module's docstring) -- silence it so a caller importing onnxsim
# doesn't get that banner unexpectedly in its own logs.
with contextlib.redirect_stdout(io.StringIO()):
    try:
        import torch
        from ppq.api import export_ppq_graph, quantize_onnx_model
        from ppq.api.setting import QuantizationSettingFactory
        from ppq.core import TargetPlatform

        PPQ_AVAILABLE = True
    except Exception as exc:  # pragma: no cover - confirmed to always fail today
        _UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"


def unavailable_reason() -> Optional[str]:
    """Why the PPQ bridge is unusable on this host, or None if usable.

    See this module's docstring: confirmed to always be unusable today
    against any environment with a current ``onnx`` install.
    """
    return _UNAVAILABLE_REASON


def quantize_with_ppq(
    model: onnx.ModelProto,
    calibration_data: Iterable[Dict[str, np.ndarray]],
    *,
    calib_steps: int = 16,
) -> onnx.ModelProto:
    """Statically quantize ``model`` using PPQ's own calibration/quantization
    pipeline, targeting ``TargetPlatform.ONNXRUNTIME`` (PPQ's QDQ-format
    ONNX Runtime export target) with PPQ's default quantization setting.

    ``calibration_data`` is a sequence of ``{input_name: np.ndarray}``
    batches, matching every other calibration-data convention in this
    project (see e.g. ``onnxsim.calibration.generate_random_calibration_
    data``) -- converted here to the ``dict``-of-``torch.Tensor`` form
    PPQ's own executor accepts for a multi-input graph (confirmed from
    ``BaseQuantizer.quantize``'s own type hints: ``inputs: Union[torch.
    Tensor, list, dict]``).

    Runs entirely on CPU (``device="cpu"``) -- no CUDA requirement.

    Raises if PPQ is unavailable (see this module's docstring for why that
    is the confirmed, expected outcome today); check `PPQ_AVAILABLE` first.
    """
    if not PPQ_AVAILABLE:
        raise RuntimeError(f"PPQ unavailable: {_UNAVAILABLE_REASON}")

    calibration_data = list(calibration_data)
    if not calibration_data:
        raise ValueError("calibration_data must have at least one batch")

    def to_torch(batch: Dict[str, np.ndarray]) -> Dict[str, "torch.Tensor"]:
        return {name: torch.from_numpy(np.asarray(arr)) for name, arr in batch.items()}

    torch_batches = [to_torch(batch) for batch in calibration_data]
    dummy_inputs = torch_batches[0]

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.onnx")
        onnx.save(model, in_path)

        quantized_graph = quantize_onnx_model(
            onnx_import_file=in_path,
            calib_dataloader=torch_batches,
            calib_steps=min(calib_steps, len(torch_batches)),
            input_shape=None,
            inputs=dummy_inputs,
            platform=TargetPlatform.ONNXRUNTIME,
            setting=QuantizationSettingFactory.default_setting(),
            device="cpu",
        )

        out_prefix = os.path.join(td, "out")
        export_ppq_graph(
            graph=quantized_graph,
            platform=TargetPlatform.ONNXRUNTIME,
            graph_save_to=out_prefix,
        )
        out_path = out_prefix if out_prefix.endswith(".onnx") else out_prefix + ".onnx"
        return onnx.load(out_path)
