"""A PPQ-API-shaped quantization shim backed entirely by onnxsim's own
:func:`onnxsim.quantize_static` -- no PPQ install required or even
possible (see :mod:`onnxsim.ppq_integration`'s module docstring: PPQ 0.6.6,
its latest PyPI release as of this writing, cannot be imported at all
alongside a modern ``onnx``/``protobuf``, and appears unmaintained -- its
last release predates both of those incompatibilities being introduced
upstream).

Rather than trying to get the real PPQ running, this module reproduces the
small, commonly-used slice of PPQ's own public calling convention --
``quantize_onnx_model()``, ``export_ppq_graph()``,
``TargetPlatform.ONNXRUNTIME``, ``QuantizationSettingFactory.
default_setting()`` (names and parameter lists read directly from PPQ
0.6.6's own ``ppq/api/interface.py`` and ``ppq/api/setting.py`` source) --
so an existing PPQ-based calibration script can switch to onnxsim with a
one-line import change:

    # before
    from ppq.api import quantize_onnx_model, export_ppq_graph
    from ppq.api.setting import QuantizationSettingFactory
    from ppq.core import TargetPlatform

    # after
    from onnxsim.ppq_compat import quantize_onnx_model, export_ppq_graph
    from onnxsim.ppq_compat import QuantizationSettingFactory, TargetPlatform

    graph = quantize_onnx_model(
        onnx_import_file="model.onnx",
        calib_dataloader=my_loader,
        calib_steps=32,
        input_shape=[1, 3, 224, 224],
        platform=TargetPlatform.ONNXRUNTIME,
    )
    export_ppq_graph(graph, TargetPlatform.ONNXRUNTIME, "model.quant")

with the call sites otherwise unchanged. The actual quantization math is
entirely onnxsim's own ``quantize_static`` (MinMax/entropy/mse calibration,
asymmetric UINT8 activations + per-output-channel symmetric INT8 weights),
not a reimplementation of PPQ's own algorithms.

**Scope, deliberately narrower than real PPQ:**

- Only ``TargetPlatform.ONNXRUNTIME`` (PPQ's own QDQ-format ONNX Runtime
  export target) is supported -- real PPQ has ~20 target platforms for
  per-vendor backend-specific formats (PPL_CUDA_INT8, SNPE_INT8, ...) with
  no onnxsim equivalent. Any other platform raises ``NotImplementedError``.
- No custom per-layer dispatch table, mixed-precision policy, or graph
  optimization passes -- ``QuantizationSetting``'s only field this shim
  reads is ``calib_algorithm`` (mapped to ``quantize_static``'s
  ``method=`` parameter); every other PPQ setting field is a no-op here.
- ``quantize_onnx_model`` returns a plain ``onnx.ModelProto`` (onnxsim's
  own QDQ-quantized graph), not PPQ's own ``ppq.IR.BaseGraph`` -- there is
  no PPQ installed to construct one. ``export_ppq_graph`` therefore just
  calls ``onnx.save`` under the hood.
- No torch dependency: ``calib_dataloader`` may yield ``numpy.ndarray``,
  ``dict``/``list``/``tuple`` of them, or (duck-typed, via ``.detach()``)
  ``torch.Tensor`` -- whatever an existing PPQ calibration loader already
  produces -- converted to onnxsim's own ``{input_name: np.ndarray}``
  calibration-batch convention.

This module and :mod:`onnxsim.ppq_integration` (the real, optional bridge
to an actually-installed PPQ, confirmed broken today) solve the same
problem from opposite directions; unless you specifically need PPQ's own
richer dispatch/optimization machinery in an environment where its two
confirmed import failures have been separately worked around, prefer this
module -- it always works, with no extra install.
"""

from __future__ import annotations

import itertools
from enum import IntEnum
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import onnx

from onnxsim.calibration import quantize_static


class TargetPlatform(IntEnum):
    """A minimal stand-in for PPQ's own ``ppq.core.TargetPlatform`` --
    only the one member this shim actually supports, kept at PPQ's own
    real numeric value (confirmed from ``ppq/core/quant.py``) so a direct
    integer comparison against a real PPQ ``TargetPlatform.ONNXRUNTIME``
    still holds true.
    """

    ONNXRUNTIME = -7


class QuantizationSetting:
    """A minimal stand-in for PPQ's own (much richer, deeply nested)
    ``ppq.api.setting.QuantizationSetting``. Only ``calib_algorithm`` is
    read by :func:`quantize_onnx_model` here; every other attribute a
    caller may set has no effect -- this shim has no dispatcher, no
    per-layer optimization passes, and no mixed-precision policy to
    configure.
    """

    def __init__(self) -> None:
        self.calib_algorithm: str = "minmax"


class QuantizationSettingFactory:
    """Stand-in for PPQ's own ``ppq.api.setting.QuantizationSettingFactory``."""

    @staticmethod
    def default_setting() -> QuantizationSetting:
        return QuantizationSetting()


def _method_from_setting(setting: Optional[Any]) -> str:
    """Best-effort mapping from a ``QuantizationSetting``'s calibration
    method to one of ``quantize_static``'s ``method=`` values. Looked up
    defensively via ``getattr`` (rather than an isinstance check against
    this module's own ``QuantizationSetting``) so a real PPQ
    ``QuantizationSetting`` -- whose equivalent field actually lives
    nested under ``setting.quantize_activation_setting.calib_algorithm``,
    per PPQ 0.6.6's own ``ppq/api/setting.py`` -- degrades to onnxsim's
    default rather than raising, since PPQ can't be installed in this
    environment to confirm that nested path against a real instance.
    """
    if setting is None:
        return "minmax"
    algo = getattr(setting, "calib_algorithm", None)
    if algo is None:
        nested = getattr(setting, "quantize_activation_setting", None)
        algo = getattr(nested, "calib_algorithm", None) if nested is not None else None
    if algo is None:
        return "minmax"
    algo = str(algo).lower()
    if algo in ("kl", "entropy", "klqf"):
        return "entropy"
    if algo == "mse":
        return "mse"
    return "minmax"


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):  # duck-typed torch.Tensor -- no hard torch import
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _batch_to_calibration_dict(
    batch: Any, input_names: List[str]
) -> Dict[str, np.ndarray]:
    if isinstance(batch, dict):
        return {name: _to_numpy(v) for name, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        if len(batch) != len(input_names):
            raise ValueError(
                f"calib_dataloader yielded {len(batch)} arrays but the model "
                f"has {len(input_names)} inputs {input_names!r}"
            )
        return {name: _to_numpy(v) for name, v in zip(input_names, batch)}
    if len(input_names) != 1:
        raise ValueError(
            f"calib_dataloader yielded a single array but the model has "
            f"{len(input_names)} inputs {input_names!r} -- yield a dict or "
            f"a list/tuple of arrays per batch instead, one per input."
        )
    return {input_names[0]: _to_numpy(batch)}


def quantize_onnx_model(
    onnx_import_file: Union[str, onnx.ModelProto],
    calib_dataloader: Optional[Iterable[Any]] = None,
    calib_steps: Optional[int] = None,
    input_shape: Optional[List[int]] = None,
    platform: TargetPlatform = TargetPlatform.ONNXRUNTIME,
    input_dtype: Optional[Any] = None,
    inputs: Optional[List[Any]] = None,
    setting: Optional[QuantizationSetting] = None,
    collate_fn: Optional[Callable[[Any], Any]] = None,
    device: str = "cpu",
    verbose: int = 0,
    do_quantize: bool = True,
) -> onnx.ModelProto:
    """Drop-in-callable-shaped replacement for PPQ's own
    ``ppq.api.quantize_onnx_model`` (same parameter names/order, read from
    PPQ 0.6.6's own source), backed by :func:`onnxsim.quantize_static`
    instead of PPQ's quantizer -- see this module's docstring for the
    scope this narrows PPQ's own behavior down to.

    ``input_shape``/``input_dtype``/``inputs``/``device`` are accepted for
    call-site compatibility but unused: onnxsim's own quantizer traces
    calibration shapes/dtypes directly from ``calib_dataloader``'s batches
    and always runs on CPU.

    :raises NotImplementedError: if ``platform`` is anything other than
            ``TargetPlatform.ONNXRUNTIME``.
    :raises TypeError: if ``do_quantize`` is true and ``calib_dataloader``
            or ``calib_steps`` is omitted -- matching real PPQ's own error
            for the same condition.
    """
    if platform != TargetPlatform.ONNXRUNTIME:
        raise NotImplementedError(
            f"onnxsim.ppq_compat only supports platform=TargetPlatform.ONNXRUNTIME "
            f"(QDQ-format quantization); got {platform!r}. Real PPQ's other target "
            f"platforms have no onnxsim equivalent -- see this module's docstring."
        )

    model = (
        onnx.load(onnx_import_file)
        if isinstance(onnx_import_file, str)
        else onnx_import_file
    )

    if not do_quantize:
        return model

    if calib_dataloader is None or calib_steps is None:
        raise TypeError(
            "Quantization needs a valid calib_dataloader and calib_steps setting."
        )

    input_names = [inp.name for inp in model.graph.input]
    batches = []
    for batch in itertools.islice(calib_dataloader, calib_steps):
        if collate_fn is not None:
            batch = collate_fn(batch)
        batches.append(_batch_to_calibration_dict(batch, input_names))

    method = _method_from_setting(setting)
    return quantize_static(model, calibration_data=batches, method=method)


def export_ppq_graph(
    graph: onnx.ModelProto,
    platform: TargetPlatform = TargetPlatform.ONNXRUNTIME,
    graph_save_to: Optional[str] = None,
    config_save_to: Optional[str] = None,
    copy_graph: bool = True,
    **kwargs: Any,
) -> None:
    """Drop-in-callable-shaped replacement for PPQ's own
    ``ppq.api.export_ppq_graph``. ``graph`` here is a plain
    ``onnx.ModelProto`` (this module's ``quantize_onnx_model`` return
    value, not PPQ's own ``ppq.IR.BaseGraph``), so this is just
    ``onnx.save`` -- matching real PPQ's own behavior of appending the
    ``.onnx`` extension for you when ``graph_save_to`` doesn't already end
    with one.

    ``config_save_to`` and ``copy_graph`` are accepted for call-site
    compatibility but unused: onnxsim's QDQ nodes carry their own
    quantization parameters directly in the graph, so there is no separate
    quantization-config file to write.
    """
    if platform != TargetPlatform.ONNXRUNTIME:
        raise NotImplementedError(
            f"onnxsim.ppq_compat only supports platform=TargetPlatform.ONNXRUNTIME; "
            f"got {platform!r}."
        )
    if graph_save_to is None:
        raise ValueError("graph_save_to is required")
    path = graph_save_to if graph_save_to.endswith(".onnx") else graph_save_to + ".onnx"
    onnx.save(graph, path)
