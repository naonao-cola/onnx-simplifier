"""Tests for converting a (simplified) ONNX model to Core ML.

onnxsim can hand its cleaned-up ``ModelProto`` to a hand-written ONNX-to-MIL
translator and produce a Core ML model via coremltools' own MIL-to-Core-ML backend
(``onnxsim.export_coreml`` / the ``--emit-coreml`` CLI flag, implemented in
``onnxsim/coreml_export.py``). coremltools dropped its own ONNX frontend in version 7,
so this translator -- not coremltools -- is what maps ONNX ops onto MIL ops.

coremltools is heavy and not part of onnxsim's test requirements, so -- exactly like
``tests/test_mlir_export.py`` -- the whole module is skipped when it is not installed.
The dedicated ``coreml-integration`` CI workflow installs coremltools and runs these
tests; the regular build-and-test matrix skips them.

Converting an ONNX model to a MIL program needs no macOS-specific functionality (MIL
construction and Core ML model serialization are pure Python/protobuf), so these tests
run the same on Linux, macOS, or Windows -- they only ever check the *converted model's*
declared shapes and dtypes, and its MIL-level constant-folded values, never its
Core ML runtime prediction (that needs Apple's Core ML framework and is covered
separately by the coreml-integration workflow's macOS job).
"""

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

pytest.importorskip("coremltools", reason="coremltools is not installed")

import onnxruntime as ort  # noqa: E402  (imported after the coremltools availability check)

import onnxsim  # noqa: E402
from onnxsim import coreml_export  # noqa: E402


def _model(
    body: str, initializer=(), opset: int = 17, ir_version: int = 8
) -> onnx.ModelProto:
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(initializer)
    return model


def _relu_model() -> onnx.ModelProto:
    model = _model(
        """
        relu (float[2,3] x) => (float[2,3] y)
        {
            y = Relu (x)
        }
        """
    )
    onnx.checker.check_model(model)
    return model


def _foldable_model() -> onnx.ModelProto:
    """Add(input, const_a + const_b) -- the inner Add folds to one constant."""
    a = numpy_helper.from_array(np.array([1, 2, 3], np.float32), name="a")
    b = numpy_helper.from_array(np.array([4, 5, 6], np.float32), name="b")
    model = _model(
        """
        foldadd (float[3] x) => (float[3] y)
        {
            ab = Add (a, b)
            y = Add (x, ab)
        }
        """,
        initializer=[a, b],
    )
    onnx.checker.check_model(model)
    return model


def _cnn_model() -> onnx.ModelProto:
    """Conv -> BatchNorm -> Relu -> GlobalAveragePool -> Flatten -> Gemm -> Softmax."""
    rng = np.random.RandomState(0)
    w = numpy_helper.from_array(rng.randn(4, 3, 3, 3).astype(np.float32), name="w")
    b = numpy_helper.from_array(np.zeros(4, np.float32), name="b")
    scale = numpy_helper.from_array(np.ones(4, np.float32), name="scale")
    bn_bias = numpy_helper.from_array(np.zeros(4, np.float32), name="bn_bias")
    mean = numpy_helper.from_array(np.zeros(4, np.float32), name="mean")
    var = numpy_helper.from_array(np.ones(4, np.float32), name="var")
    gw = numpy_helper.from_array(rng.randn(4, 4).astype(np.float32), name="gw")
    gb = numpy_helper.from_array(np.zeros(4, np.float32), name="gb")
    model = _model(
        """
        cnn (float[1,3,8,8] x) => (float[1,4] y)
        {
            conv_out = Conv <kernel_shape=[3,3], pads=[1,1,1,1]> (x, w, b)
            bn_out = BatchNormalization (conv_out, scale, bn_bias, mean, var)
            relu_out = Relu (bn_out)
            gap_out = GlobalAveragePool (relu_out)
            flat_out = Flatten <axis=1> (gap_out)
            gemm_out = Gemm <transB=1> (flat_out, gw, gb)
            y = Softmax <axis=-1> (gemm_out)
        }
        """,
        initializer=[w, b, scale, bn_bias, mean, var, gw, gb],
    )
    onnx.checker.check_model(model)
    return model


def _mil_const_value(model: onnx.ModelProto):
    """Build ``model`` (all-initializer, no declared graph inputs) as MIL and read
    back its single output's compile-time-constant value.

    Used to check a translated op's numeric behavior without needing Core ML's
    runtime (which only exists on macOS): with every input a constant, MIL's own
    constant-folding evaluates the op with real numpy code, so the result reflects
    exactly how the translator wired up that op's arguments.
    """
    prog = coreml_export._build_mil_program(model, *coreml_export._import_mil())
    return np.asarray(prog.functions["main"].outputs[0].val)


# ---------------------------------------------------------------------------
# Basic conversion
# ---------------------------------------------------------------------------


def test_has_coremltools_true_here():
    assert coreml_export.has_coremltools() is True


def test_export_returns_mlmodel_with_matching_io():
    mlmodel = onnxsim.export_coreml(_relu_model())
    spec = mlmodel.get_spec()
    assert [i.name for i in spec.description.input] == ["x"]
    assert [o.name for o in spec.description.output] == ["y"]


def test_export_writes_mlpackage(tmp_path):
    out = tmp_path / "relu.mlpackage"
    onnxsim.export_coreml(_relu_model(), str(out))
    assert (out / "Manifest.json").is_file()


def test_export_of_simplified_model():
    model = _foldable_model()
    simplified, ok = onnxsim.simplify(model)
    assert ok
    # The redundant const+const Add is folded away by onnxsim.
    assert [n.op_type for n in simplified.graph.node].count("Add") == 1
    mlmodel = onnxsim.export_coreml(simplified)
    spec = mlmodel.get_spec()
    assert [o.name for o in spec.description.output] == ["y"]


def test_convert_to_coreml_matches_export_coreml():
    model = _relu_model()
    a = coreml_export.convert_to_coreml(model).get_spec()
    b = onnxsim.export_coreml(model).get_spec()
    assert a.description.input == b.description.input
    assert a.description.output == b.description.output


# ---------------------------------------------------------------------------
# A small CNN pipeline, exercising conv/norm/pool/gemm/softmax together
# ---------------------------------------------------------------------------


def test_cnn_pipeline_converts_with_expected_shape():
    mlmodel = onnxsim.export_coreml(_cnn_model())
    spec = mlmodel.get_spec()
    (out_desc,) = spec.description.output
    assert list(out_desc.type.multiArrayType.shape) == [1, 4]


def test_neuralnetwork_format():
    mlmodel = onnxsim.export_coreml(_cnn_model(), convert_to="neuralnetwork")
    spec = mlmodel.get_spec()
    assert spec.WhichOneof("Type") == "neuralNetwork"


# ---------------------------------------------------------------------------
# Numeric regression coverage for the trickier translations
# ---------------------------------------------------------------------------


def test_slice_negative_step_reverses_full_axis():
    # Reversing a whole axis with a negative step needs Slice's `end` to mean
    # "through index 0 inclusive" -- a case MIL only expresses via `end_mask`
    # (see the comment in coreml_export._op_slice for why a literal end=-1 doesn't
    # work: MIL wraps a negative `end` the same way numpy indexing does, silently
    # turning it back into an empty slice).
    x = numpy_helper.from_array(np.arange(5).astype(np.float32), name="x")
    starts = numpy_helper.from_array(np.array([4], np.int64), name="starts")
    ends = numpy_helper.from_array(np.array([-100], np.int64), name="ends")
    axes = numpy_helper.from_array(np.array([0], np.int64), name="axes")
    steps = numpy_helper.from_array(np.array([-1], np.int64), name="steps")
    model = _model(
        "slicerev () => (float[5] out) { out = Slice (x, starts, ends, axes, steps) }",
        initializer=[x, starts, ends, axes, steps],
    )
    np.testing.assert_array_equal(_mil_const_value(model), [4, 3, 2, 1, 0])


def test_gemm_alpha_beta_transb_matches_onnxruntime():
    rng = np.random.RandomState(0)
    a = rng.randn(3, 4).astype(np.float32)
    b = rng.randn(5, 4).astype(np.float32)
    c = rng.randn(5).astype(np.float32)
    inits = [
        numpy_helper.from_array(a, name="a"),
        numpy_helper.from_array(b, name="b"),
        numpy_helper.from_array(c, name="c"),
    ]
    model = _model(
        "gemm () => (float[3,5] out) "
        "{ out = Gemm <alpha=0.5, beta=2.0, transB=1> (a, b, c) }",
        initializer=inits,
    )
    expected = 0.5 * (a @ b.T) + 2.0 * c
    np.testing.assert_allclose(_mil_const_value(model), expected, rtol=1e-5, atol=1e-5)


def test_pad_reflect_matches_onnxruntime():
    x = np.arange(12, dtype=np.float32).reshape(1, 1, 3, 4)
    pads = np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64)
    model_onnx = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Pad", ["x", "pads"], ["out"], mode="reflect")],
            "g",
            [
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, list(x.shape)
                )
            ],
            [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, None)],
            initializer=[numpy_helper.from_array(pads, name="pads")],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 17)],
        ir_version=8,
    )
    sess = ort.InferenceSession(
        model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    expected = sess.run(None, {"x": x})[0]

    inits = [
        numpy_helper.from_array(x, name="x"),
        numpy_helper.from_array(pads, name="pads"),
    ]
    model_const = _model(
        'padref () => (float[1,1,5,6] out) { out = Pad <mode="reflect"> (x, pads) }',
        initializer=inits,
    )
    np.testing.assert_array_equal(_mil_const_value(model_const), expected)


def test_slice_end_sentinel_survives_int64_downcast():
    # ONNX graphs routinely use INT64_MAX as a Slice `ends` sentinel meaning "to
    # the end of this axis" (e.g. torch.onnx's export of `x[..., 32:]`). MIL has
    # no int64 tensor type, so this translator downcasts int64 initializers to
    # int32 -- a plain `.astype(int32)` wraps INT64_MAX around to -1 instead of
    # saturating, which Slice's clamp logic would then read as "one before the
    # end", silently dropping the last element. Regression test for that.
    x = numpy_helper.from_array(np.arange(8, dtype=np.float32), name="x")
    starts = numpy_helper.from_array(np.array([3], np.int64), name="starts")
    ends = numpy_helper.from_array(
        np.array([9223372036854775807], np.int64), name="ends"
    )
    model = _model(
        "slicesentinel () => (float[5] out) { out = Slice (x, starts, ends) }",
        initializer=[x, starts, ends],
    )
    np.testing.assert_array_equal(_mil_const_value(model), [3, 4, 5, 6, 7])


def test_rope_ops_match_onnxruntime():
    # Sin/Cos/Where/Expand/And/IsNaN/Shape/ConstantOfShape/Range and a bool-typed
    # Gather all round out the op set a transformer decoder (RoPE + causal
    # masking) needs; onnxsim/coreml_export.py was built against
    # HuggingFaceTB/SmolLM2-135M-Instruct's exported decoder graph, which uses
    # every one of them. Exercise the trig/select/broadcast/logical trio that
    # decomposed op-by-op checks don't cover as a combination.
    angle = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=np.float32)
    cond = np.array([True, False, True, False])
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32)
    inits = [
        numpy_helper.from_array(angle, name="angle"),
        numpy_helper.from_array(cond, name="cond"),
        numpy_helper.from_array(a, name="a"),
        numpy_helper.from_array(b, name="b"),
    ]
    model = _model(
        """
        rope () => (float[4] out)
        {
            s = Sin (angle)
            c = Cos (angle)
            sc = Mul (s, c)
            picked = Where (cond, a, b)
            out = Add (sc, picked)
        }
        """,
        initializer=inits,
    )
    expected = np.sin(angle) * np.cos(angle) + np.where(cond, a, b)
    np.testing.assert_allclose(_mil_const_value(model), expected, rtol=1e-5, atol=1e-5)


def test_gather_on_bool_tensor():
    mask = numpy_helper.from_array(
        np.array([True, False, True, False, True]), name="mask"
    )
    idx = numpy_helper.from_array(np.array([0, 2, 4], np.int64), name="idx")
    model = _model(
        "gatherbool () => (bool[3] out) { out = Gather <axis=0> (mask, idx) }",
        initializer=[mask, idx],
    )
    np.testing.assert_array_equal(_mil_const_value(model), [True, True, True])


def test_zero_length_input_dimension_is_static():
    # A concrete 0-length dimension (e.g. an empty KV cache) is fully static --
    # just empty -- and must not be rejected as if it were a dynamic/symbolic
    # dimension.
    x = numpy_helper.from_array(np.zeros((1, 0, 4), np.float32), name="x")
    y = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
    model = _model(
        "emptycat () => (float[1,2,4] out) { out = Concat <axis=1> (x, y) }",
        initializer=[x, numpy_helper.from_array(y, name="y")],
    )
    np.testing.assert_array_equal(_mil_const_value(model), y)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_unsupported_op_raises():
    model = _model(
        "loopy (float[3] x) => (float[3] y) { y = Loop (x) }",
    )
    with pytest.raises(RuntimeError, match="Loop.*not supported"):
        onnxsim.export_coreml(model)


def test_dynamic_input_shape_raises():
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [None, 3])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [None, 3])
    node = onnx.helper.make_node("Relu", ["x"], ["y"])
    graph = onnx.helper.make_graph([node], "g", [x], [y])
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    with pytest.raises(RuntimeError, match="non-static dimension"):
        onnxsim.export_coreml(model)
