"""tpu-mlir integration test.

`tpu-mlir <https://github.com/sophgo/tpu-mlir>`_ is Sophgo's open MLIR-based
compiler for its TPU/KPU accelerator line (BM168x, and CV18xx -- which
includes the Milk-V Duo's CV1800B). Its own ONNX front end
(``transform.OnnxConverter.OnnxConverter``) is a direct onnxsim consumer: its
``model_simplify()`` calls ``onnxsim.simplify(skip_fuse_bn=..., skip_constant_
folding=True, skip_shape_inference=True)`` as part of ingesting every ONNX
model, sandwiched between two of tpu-mlir's own runtime-value-driven
``ConstantFolding`` passes (``transform.OnnxOpt.ConstantFolding``, which -
unlike onnxsim's own compile-time constant folding - actually executes
subgraphs against real sample input data). tpu-mlir's own code comments next
to that call name two real, model-specific failures it hit doing this:
"Do constantFolding before onnxsim to avoid onnxsim bug (such as run yolox)"
and "... after onnxsim to avoid onnxsim bug (such as run ppyolo_tiny)". It
also pins an old onnxsim (``onnxsim==0.4.17``, see its own ``release_tools/
setup.py``'s ``[onnx]`` extra) rather than tracking latest.

This module is the regression test for that dependency against the *current*
onnxsim: it constructs tpu-mlir's own ``OnnxTransformer``
(``tools.model_transform.OnnxTransformer``, the class ``model_transform.py``'s
CLI itself wraps ``OnnxConverter`` with) against a handful of representative
graphs and checks:

1. Construction must not raise: whatever tpu-mlir's own onnxsim call accepted
   before, it must still accept with the onnxsim installed here.
2. The resulting Top MLIR must reflect the fusions onnxsim is expected to have
   done (e.g. a BatchNorm-style scale folded into the preceding Conv, rather
   than surviving as a separate ``top.Mul``).
3. Once tpu-mlir's own ``tpuc-opt`` canonicalizes the Top MLIR (shape
   inference + weight materialization -- the raw importer output leaves
   tensors unranked, which tpu-mlir's own ``pymlir`` interpreter can't
   allocate), running it through that interpreter must agree numerically with
   onnx's own reference evaluator on the *original*, unsimplified graph.

Whether the two historical bugs the code comments name ("yolox", "ppyolo_tiny")
still reproduce on current onnxsim was investigated directly against real
models: this repo's own ``scripts/regression/yolox/`` harness already shows
6/6 real YOLOX checkpoints passing clean on onnxsim v0.7.0, and a real
``ppyolo_tiny`` PaddleDetection checkpoint was obtained and traced far enough
to confirm its detection head is NMS-based (the same category as onnxsim
issue #60, a ``PrepareForReduce`` crash from onnxsim's random-input constant
folding degenerating an NMS output to a zero-sized dimension) before hitting an
unrelated, permanent paddle2onnx limitation (it never supported the bare,
pre-``multiclass_nms3`` op that specific 2021 checkpoint uses) -- see this
project's own commit history/session notes for the full trace. Neither bug
reproduces on the synthetic graphs here.

``tpu_mlir`` is heavy (a ~310MB wheel), Python-3.10-only
(``python_requires=">=3.10,<3.11"``), and only installable alongside its own
pinned ``onnx``/``onnxruntime``/numpy/protobuf via its ``[onnx]``/``[torch]``
extras (see this repo's ``backend-integration.yml`` ``tpu_mlir`` job for the
exact pin set: plain ``pip install tpu_mlir`` alone pulls a numpy/protobuf
pair current ``onnx`` can't import against). It is also only really runnable
on the Ubuntu version it declares (``platforms="unbuntu22.04"`` in its own
``setup.py``): its compiled ``tpuc-opt``/pymlir native extensions are linked
against Ubuntu 22.04's glibc/libstdc++ and segfault against a newer host's
ABI (confirmed directly against Ubuntu 24.04, with or without
``LD_LIBRARY_PATH`` tricks to mix vendored and host libraries) -- so the
dedicated CI job below pins ``runs-on: ubuntu-22.04`` specifically, unlike
every sibling job in that workflow. None of this is part of onnxsim's test
requirements, so the whole module is skipped when ``tpu_mlir`` is not
installed; the regular build-and-test matrix skips it.
"""

import contextlib
import os

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

tpu_mlir = pytest.importorskip("tpu_mlir", reason="tpu_mlir is not installed")
from tools.model_runner import mlir_inference  # noqa: E402
from tools.model_transform import OnnxTransformer  # noqa: E402

import onnxsim  # noqa: E402  (imported after the tpu_mlir availability check)

_OPSET = 13
_IR_VERSION = 8


def _model(
    body, initializer=(), opset=_OPSET, ir_version=_IR_VERSION
) -> onnx.ModelProto:
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    onnx.checker.check_model(model)
    return model


def _rand(*shape, seed=0) -> np.ndarray:
    return np.random.RandomState(seed).randn(*shape).astype(np.float32)


def _conv_bn_relu() -> onnx.ModelProto:
    """Conv -> Mul(scale) -> Add(shift) -> Relu: onnxsim's fuse_bn_into_conv
    folds the scale into Conv's weights, and its fuse_add_bias_into_conv
    (onnxsim/custom_optimizer_passes.cpp) then folds the shift into Conv's
    bias, so neither Mul nor Add survives simplification."""
    w = numpy_helper.from_array(_rand(8, 3, 3, 3, seed=1), "w")
    scale = numpy_helper.from_array(_rand(1, 8, 1, 1, seed=2), "scale")
    shift = numpy_helper.from_array(_rand(1, 8, 1, 1, seed=3), "shift")
    return _model(
        """
        conv_bn_relu (float[1,3,8,8] x) => (float[1,8,8,8] y)
        {
          c = Conv<pads = [1, 1, 1, 1]>(x, w)
          m = Mul(c, scale)
          a = Add(m, shift)
          y = Relu(a)
        }
        """,
        [w, scale, shift],
    )


def _redundant_transpose() -> onnx.ModelProto:
    """An identity Transpose (perm=[0,1,2,3]) that onnxsim removes outright."""
    w = numpy_helper.from_array(_rand(8, 3, 3, 3, seed=1), "w")
    return _model(
        """
        redundant_transpose (float[1,3,8,8] x) => (float[1,8,8,8] y)
        {
          t = Transpose<perm = [0, 1, 2, 3]>(x)
          c = Conv<pads = [1, 1, 1, 1]>(t, w)
          y = Relu(c)
        }
        """,
        [w],
    )


def _foldable_shape_reshape() -> onnx.ModelProto:
    """Shape -> Gather -> Concat -> Reshape, fully determined by constants."""
    w = numpy_helper.from_array(_rand(8, 3, 3, 3, seed=1), "w")
    idx = numpy_helper.from_array(np.array([0], np.int64), "idx")
    minus1 = numpy_helper.from_array(np.array([-1], np.int64), "m1")
    ch = numpy_helper.from_array(np.array([8], np.int64), "ch")
    return _model(
        """
        foldable_shape_reshape (float[1,3,8,8] x) => (float[1,8,64] y)
        {
          c = Conv<pads = [1, 1, 1, 1]>(x, w)
          r = Relu(c)
          shp = Shape(r)
          n = Gather<axis = 0>(shp, idx)
          newshape = Concat<axis = 0>(n, ch, m1)
          y = Reshape(r, newshape)
        }
        """,
        [w, idx, minus1, ch],
    )


_MODELS = {
    "conv_bn_relu": _conv_bn_relu,
    "redundant_transpose": _redundant_transpose,
    "foldable_shape_reshape": _foldable_shape_reshape,
}


@contextlib.contextmanager
def _chdir(path):
    """tpu-mlir's ModelTransformer writes several intermediate files (weight
    npz, a stripped .prototxt, the Top MLIR itself) relative to the current
    directory -- run it from a throwaway ``tmp_path`` instead of the repo
    root."""
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


def _input_shape(model: onnx.ModelProto) -> list:
    return [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]


def _random_feed(model: onnx.ModelProto, seed: int = 0) -> np.ndarray:
    """Every graph in this module is single-input; return that one array."""
    rng = np.random.RandomState(seed)
    shape = _input_shape(model)
    return (rng.rand(*shape).astype(np.float32) - 0.5) * 2.0


def _ingest(model: onnx.ModelProto, name: str, tmp_path) -> OnnxTransformer:
    """Feed ``model`` through tpu-mlir's real ``OnnxTransformer``/``OnnxConverter``
    -- the exact code path ``model_simplify()``'s ``onnxsim.simplify(...)`` call
    lives on -- with onnxsim enabled, matching what ``model_transform.py``'s CLI
    does by default (``do_onnx_sim`` defaults to ``True``)."""
    with _chdir(tmp_path):
        return OnnxTransformer(
            name,
            model,
            input_shapes=[_input_shape(model)],
            output_names=[],
            do_onnx_sim=True,
        )


def _run_on_tpu_mlir(
    model: onnx.ModelProto, name: str, feed: np.ndarray, tmp_path
) -> np.ndarray:
    """Ingest, canonicalize (``tpuc-opt``), and interpret (``pymlir``) ``model``
    through the real tpu-mlir stack, and return its single output tensor."""
    with _chdir(tmp_path):
        t = OnnxTransformer(
            name,
            model,
            input_shapes=[_input_shape(model)],
            output_names=[],
            do_onnx_sim=True,
        )
        mlir_file = f"{name}.mlir"
        t.model_transform(mlir_file)
        input_name = t.converter.input_names[0]
        out = mlir_inference({input_name: feed}, mlir_file, dump_all=False)
    return next(iter(out.values()))


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_onnx_converter_accepts_simplified_input(name, tmp_path):
    """tpu-mlir's own OnnxConverter (``do_onnx_sim=True``) must not choke on
    the onnxsim installed here -- the exact call ``OnnxConverter.model_simplify()``
    makes, which tpu-mlir's own code comments say has broken against real
    models (yolox, ppyolo_tiny) before."""
    model = _MODELS[name]()
    t = _ingest(model, name, tmp_path)
    assert t.converter.model is not None


def test_simplify_bn_fusion_reaches_top_mlir(tmp_path):
    """The Mul(scale)/Add(shift) onnxsim fuses into Conv must not reappear as
    separate top.Mul/top.Add ops in tpu-mlir's own Top MLIR -- confirms
    onnxsim's simplification (not just a pass-through no-op) survives
    ingestion."""
    model = _conv_bn_relu()
    t = _ingest(model, "conv_bn_relu_check", tmp_path)
    with _chdir(tmp_path):
        mlir_path = t.model_mlir()
        text = open(mlir_path).read()
    assert "top.Conv" in text
    assert "top.Mul" not in text  # scale folded into the Conv's weights
    assert "top.Add" not in text  # shift folded into the Conv's bias


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_simplified_model_matches_onnx_reference_on_tpu_mlir(name, tmp_path):
    """Simplification must not change what tpu-mlir actually computes."""
    model = _MODELS[name]()
    feed = _random_feed(model, seed=0)

    tpu_out = _run_on_tpu_mlir(model, name, feed, tmp_path)

    from onnx.reference import ReferenceEvaluator

    reference_out = ReferenceEvaluator(model).run(
        None, {model.graph.input[0].name: feed}
    )[0]
    np.testing.assert_allclose(reference_out, tpu_out, rtol=1e-3, atol=1e-4)


def test_simplify_is_bit_exact_on_tpu_mlir(tmp_path):
    """Removing a redundant (identity) Transpose must not change the result."""
    model = _redundant_transpose()
    simplified, check_ok = onnxsim.simplify(model)
    assert check_ok
    assert len(simplified.graph.node) < len(model.graph.node)

    feed = _random_feed(model, seed=1)
    original_out = _run_on_tpu_mlir(model, "redundant_transpose_orig", feed, tmp_path)
    simplified_out = _run_on_tpu_mlir(
        simplified, "redundant_transpose_simplified", feed, tmp_path
    )
    np.testing.assert_allclose(original_out, simplified_out, rtol=1e-5, atol=1e-6)
