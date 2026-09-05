"""Convert a (simplified) ONNX model to TensorFlow Lite via `onnx2tf
<https://github.com/PINTO0309/onnx2tf>`_.

``onnxsim/tflite_export.py`` ships a hand-written ONNX-to-TensorFlow translator
covering a practical subset of ops. onnx2tf is a separate, actively maintained
project that does the same job with far broader op coverage (~200 ops) and years of
production hardening across real-world model zoos -- at the cost of being a much
heavier dependency (it pulls its own TensorFlow, onnxruntime, onnx-graphsurgeon, and
a couple dozen small ``*4onnx`` helper packages) and changing the model's public
input/output tensor layout by default (it converts every tensor of rank >= 3 to a
channel-last convention, not just 4-D image tensors -- see onnx2tf's own
``keep_ncw_or_nchw_or_ncdhw_input_names`` and related options to pin specific inputs
to their original layout instead).

This module is the ``backend="onnx2tf"`` implementation behind
``onnxsim.export_tflite`` / ``tflite_export.convert_to_tflite`` -- use it when a model
hits an unsupported op in the built-in translator. onnx2tf is an **optional**
dependency: nothing here is imported unless that backend is actually selected.

onnx2tf.convert() unconditionally tries to download a small sample-image ``.npy``
file from its GitHub releases for internal per-op numeric self-verification,
whenever any graph input looks image-shaped (rank-4, channel-last-3 once converted to
onnx2tf's own layout convention) -- even when no verification flag is passed, and
with no offline fallback if the network is unavailable or blocked. The actual pixel
values only need to be finite and correctly shaped (they sanity-check onnx2tf's own
lowering, not onnxsim's output), so this module monkeypatches that one function for
the duration of the call to hand back a small in-memory random array instead of
letting it hit the network.
"""

import contextlib
import os
import tempfile
from typing import Any, Dict, Optional
from unittest import mock

import numpy as np
import onnx

_ONNX2TF_INSTALL_HINT = (
    "onnx2tf is required for the 'onnx2tf' TFLite export backend but is not "
    "installed. Install it with `pip install onnx2tf` -- note this is a heavy "
    "dependency: it pulls its own TensorFlow, onnxruntime, and onnx-graphsurgeon."
)


def has_onnx2tf() -> bool:
    """Whether onnx2tf is importable in this environment."""
    try:
        import onnx2tf  # noqa: F401
    except ImportError:
        return False
    return True


def _import_onnx2tf():
    try:
        import onnx2tf
    except ImportError as exc:
        raise RuntimeError(_ONNX2TF_INSTALL_HINT) from exc
    return onnx2tf


def _random_calibration_data() -> np.ndarray:
    """Stand-in for onnx2tf's ``download_test_image_data()``: same shape/dtype
    (20 128x128 RGB images) as the real sample it fetches from GitHub releases,
    but generated in-memory instead of over the network."""
    return np.random.default_rng(0).random((20, 128, 128, 3), dtype=np.float32)


@contextlib.contextmanager
def _isolated_cwd():
    """Run onnx2tf.convert() with its current working directory pointed at a
    scratch directory, restoring the caller's cwd afterward.

    onnx2tf.convert() writes small housekeeping artifacts relative to the
    process's current working directory even with ``disable_model_save=True``
    (e.g. an empty ``saved_model`` directory, observed with onnx2tf 1.29) --
    isolating cwd keeps those out of the caller's project directory.
    """
    prev_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="onnxsim_onnx2tf_") as tmp_dir:
        os.chdir(tmp_dir)
        try:
            yield
        finally:
            os.chdir(prev_cwd)


def convert_to_tflite_via_onnx2tf(
    model: onnx.ModelProto, **onnx2tf_kwargs: Any
) -> bytes:
    """Convert an ONNX model to an in-memory TFLite flatbuffer (``bytes``) using
    onnx2tf as the ONNX-to-TensorFlow backend.

    Parameters
    ----------
    model:
        The ONNX model to convert. Typically the output of :func:`onnxsim.simplify`.
    **onnx2tf_kwargs:
        Forwarded to ``onnx2tf.convert()`` -- e.g. ``keep_ncw_or_nchw_or_ncdhw_input_names``
        (list of input names to keep in their original ONNX layout instead of
        onnx2tf's default channel-last conversion), ``batch_size``, or
        ``output_integer_quantized_tflite``. ``onnx_graph`` is always ``model``
        and ``disable_model_save``/``non_verbose`` default to ``True`` unless
        overridden. See onnx2tf's own documentation for the full option list.

    Returns
    -------
    bytes
        The serialized ``.tflite`` flatbuffer, produced by handing onnx2tf's
        in-memory Keras model to ``tf.lite.TFLiteConverter.from_keras_model``.

    Raises
    ------
    RuntimeError
        If onnx2tf is not installed, or conversion fails (onnx2tf's own error,
        naming the unsupported op/feature, is preserved in the message).
    """
    onnx2tf = _import_onnx2tf()
    import onnx2tf.onnx2tf as onnx2tf_impl
    import tensorflow as tf

    kwargs: Dict[str, Any] = {"disable_model_save": True, "non_verbose": True}
    kwargs.update(onnx2tf_kwargs)

    with mock.patch.object(
        onnx2tf_impl, "download_test_image_data", _random_calibration_data
    ):
        with _isolated_cwd():
            try:
                keras_model = onnx2tf.convert(onnx_graph=model, **kwargs)
            except Exception as exc:
                raise RuntimeError(f"onnx2tf conversion failed: {exc}") from exc

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    try:
        return converter.convert()
    except Exception as exc:
        raise RuntimeError(
            f"TFLite conversion of the onnx2tf-produced model failed: {exc}"
        ) from exc


def export_tflite_via_onnx2tf(
    model: onnx.ModelProto,
    output_path: Optional[str] = None,
    **kwargs: Any,
) -> bytes:
    """Convert ``model`` to TFLite via onnx2tf, optionally saving it to
    ``output_path``. Other keyword arguments are forwarded to
    :func:`convert_to_tflite_via_onnx2tf`. See that function for details.
    """
    tflite_model = convert_to_tflite_via_onnx2tf(model, **kwargs)
    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(tflite_model)
    return tflite_model
