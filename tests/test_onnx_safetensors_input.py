"""onnxsim's model loading is a spec-compliant ONNX external-data reader (see
``onnx::optimization::loadModel``/``loadExternalDataForModel`` in
``third_party/onnx-optimizer``): a ``TensorProto`` with
``data_location == EXTERNAL`` is hydrated by opening its ``location`` file and
reading ``length`` bytes starting at ``offset`` -- nothing about that path
cares what else is in the file or what its extension is.

https://github.com/justinchuby/onnx-safetensors represents an ONNX model's
weights exactly this way: it writes an ordinary ``.safetensors`` file (an
8-byte little-endian header length, a JSON header describing each tensor's
``data_offsets``, then the raw tensor bytes) and points each initializer's
``external_data`` at it with ``offset = data_offsets[0] + header_size + 8``
and ``length = data_offsets[1] - data_offsets[0]`` -- i.e. it reuses onnx's
own external-data mechanism, with the safetensors JSON header simply skipped
over by the offset. That means a model written by
``onnx_safetensors.save_file``/``save_model``/``load_file_as_external_data``
already IS a standard external-data ONNX model, loadable by onnxsim (or any
other spec-compliant ONNX consumer) with no onnx-safetensors-specific code on
onnxsim's side at all.

This suite locks that interop in from both directions: a hand-rolled writer
that reproduces onnx-safetensors' exact on-disk layout (so the test carries no
extra runtime dependency and mirrors the ``test_import_gguf_weights.py``
style), plus a real round trip through the ``onnx_safetensors`` package
itself when it's installed.
"""

import json
import os
import struct

import numpy as np
import onnx
import onnx.checker
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

_NUMPY_DTYPE_TO_SAFETENSORS = {
    np.dtype("float32"): "F32",
    np.dtype("int64"): "I64",
}


def _write_safetensors_like_onnx_safetensors(path, arrays):
    """Write ``path`` as a real safetensors file and return, for each name,
    the ``(location, offset, length)`` onnx external_data triple
    onnx-safetensors' own ``_read_safetensors`` would compute for it (see
    ``onnx_safetensors/_safetensors_io.py``): ``offset`` is
    ``data_offsets[0] + header_size + 8``, ``length`` is
    ``data_offsets[1] - data_offsets[0]``, and ``location`` is the file's own
    basename (relative to the ``.onnx`` file, as onnx-safetensors writes it).
    """
    header = {}
    blobs = []
    cursor = 0
    for name, arr in arrays.items():
        raw = arr.tobytes()
        header[name] = {
            "dtype": _NUMPY_DTYPE_TO_SAFETENSORS[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [cursor, cursor + len(raw)],
        }
        blobs.append(raw)
        cursor += len(raw)
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for raw in blobs:
            f.write(raw)

    header_size = len(header_bytes)
    location = os.path.basename(path)
    refs = {}
    for name, meta in header.items():
        begin, end = meta["data_offsets"]
        refs[name] = (location, begin + header_size + 8, end - begin)
    return refs


def _external_initializer(name, arr, location, offset, length):
    tensor = onnx.numpy_helper.from_array(arr, name)
    tensor.ClearField("raw_data")
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for key, value in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value
    return tensor


def _build_matmul_add_model(w, b):
    # X: [K, K] (K = w.shape[0]) so X @ W is always valid regardless of
    # whether W itself is square; Y: [K, N] (N = w.shape[1] == b.shape[0]).
    k = w.shape[0]
    x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [k, k])
    y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [k, b.shape[0]])
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W"], ["MM"]),
        onnx.helper.make_node("Add", ["MM", "B"], ["Y"]),
    ]
    return nodes, x, y


def _make_safetensors_backed_model(tmp_path, w, b):
    st_path = str(tmp_path / "model.safetensors")
    refs = _write_safetensors_like_onnx_safetensors(st_path, {"W": w, "B": b})

    nodes, x, y = _build_matmul_add_model(w, b)
    w_init = _external_initializer("W", w, *refs["W"])
    b_init = _external_initializer("B", b, *refs["B"])
    graph = onnx.helper.make_graph(nodes, "g", [x], [y], initializer=[w_init, b_init])
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9
    )

    model_path = str(tmp_path / "model.onnx")
    onnx.save(model, model_path)
    return model_path


def test_checker_accepts_safetensors_external_data(tmp_path):
    # Sanity check on the fixture itself: this is a standard, spec-valid
    # external-data model as far as plain onnx is concerned (see this
    # module's docstring) -- the interesting question is only whether
    # onnxsim's own loader (a *different* implementation of the same spec,
    # in third_party/onnx-optimizer) agrees.
    w = np.random.RandomState(0).rand(4, 4).astype(np.float32)
    b = np.random.RandomState(1).rand(4).astype(np.float32)
    model_path = _make_safetensors_backed_model(tmp_path, w, b)
    reloaded = onnx.load(model_path)
    onnx.checker.check_model(reloaded, full_check=True)
    got_w = onnx.numpy_helper.to_array(
        next(i for i in reloaded.graph.initializer if i.name == "W")
    )
    np.testing.assert_array_equal(got_w, w)


def test_simplify_from_path_reads_safetensors_external_data(tmp_path):
    # The path-based entry point (also what the CLI uses) loads the model
    # entirely inside the C++ core via onnx-optimizer's own external-data
    # loader -- never touching onnx-safetensors or the `onnx` package's
    # loader at all.
    w = np.random.RandomState(2).rand(3, 3).astype(np.float32)
    b = np.random.RandomState(3).rand(3).astype(np.float32)
    model_path = _make_safetensors_backed_model(tmp_path, w, b)

    model_opt, check_ok = onnxsim.simplify(model_path, check_n=3)
    assert check_ok

    # MatMul(X, W) + B has no non-constant inputs besides X, so constant
    # folding collapses W/B straight into the Add -- this only produces the
    # right answer if the actual safetensors-backed bytes (not garbage/zeros)
    # were read.
    x = np.random.RandomState(4).rand(3, 3).astype(np.float32)
    expected = x @ w + b
    from onnxsim import backend

    outputs = backend.run_model(model_opt, {"X": x})
    np.testing.assert_allclose(outputs["Y"], expected, rtol=1e-5, atol=1e-6)


def test_simplify_from_loaded_modelproto_reads_safetensors_external_data(tmp_path):
    # The in-memory entry point: `onnx.load` (an independent, third-party
    # implementation of the same external-data spec) resolves the reference
    # into `raw_data` before onnxsim ever sees the model.
    w = np.random.RandomState(5).rand(2, 5).astype(np.float32)
    b = np.random.RandomState(6).rand(5).astype(np.float32)
    model_path = _make_safetensors_backed_model(tmp_path, w, b)

    model = onnx.load(model_path)  # load_external_data=True (default)
    for init in model.graph.initializer:
        assert init.data_location != onnx.TensorProto.EXTERNAL

    model_opt, check_ok = onnxsim.simplify(model, check_n=3)
    assert check_ok


def test_onnx_safetensors_package_round_trip(tmp_path):
    onnx_safetensors = pytest.importorskip("onnx_safetensors")

    w = np.random.RandomState(7).rand(4, 4).astype(np.float32)
    b = np.random.RandomState(8).rand(4).astype(np.float32)
    nodes, x, y = _build_matmul_add_model(w, b)
    w_init = onnx.numpy_helper.from_array(w, "W")
    b_init = onnx.numpy_helper.from_array(b, "B")
    graph = onnx.helper.make_graph(nodes, "g", [x], [y], initializer=[w_init, b_init])
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9
    )

    model_path = str(tmp_path / "model.onnx")
    # Writes model.onnx + model.safetensors: the exact two-file,
    # standard-external-data layout https://github.com/justinchuby/onnx-safetensors
    # produces -- no onnxsim involvement whatsoever.
    onnx_safetensors.save_model(model, model_path)
    assert os.path.exists(str(tmp_path / "model.safetensors"))

    model_opt, check_ok = onnxsim.simplify(model_path, check_n=3)
    assert check_ok
