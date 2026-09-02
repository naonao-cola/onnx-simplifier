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
    prog, _flexible_inputs = coreml_export._build_mil_program(
        model, *coreml_export._import_mil()
    )
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


def test_gemm_alpha_beta_fp16_matches_expected():
    # Regression test: the alpha/beta scale factors used to be created as a bare
    # Python float, which MIL infers as fp32 regardless of context -- multiplying
    # it against an fp16 operand raised a dtype-mismatch error (found while
    # converting an fp16-exported multi-billion-parameter LLM).
    rng = np.random.RandomState(0)
    a = rng.randn(3, 4).astype(np.float16)
    b = rng.randn(5, 4).astype(np.float16)
    c = rng.randn(5).astype(np.float16)
    inits = [
        numpy_helper.from_array(a, name="a"),
        numpy_helper.from_array(b, name="b"),
        numpy_helper.from_array(c, name="c"),
    ]
    model = _model(
        "gemm () => (float16[3,5] out) "
        "{ out = Gemm <alpha=0.5, beta=2.0, transB=1> (a, b, c) }",
        initializer=inits,
    )
    expected = 0.5 * (a.astype(np.float32) @ b.astype(np.float32).T) + 2.0 * c.astype(
        np.float32
    )
    np.testing.assert_allclose(_mil_const_value(model), expected, rtol=1e-2, atol=1e-2)


def test_neg_fp16_matches_expected():
    # Same class of bug as the Gemm fp16 case above, in Neg's `mul(x, -1)`
    # lowering.
    x = np.array([1.5, -2.0, 0.0], dtype=np.float16)
    model = _model(
        "neg () => (float16[3] y) { y = Neg (x) }",
        initializer=[numpy_helper.from_array(x, name="x")],
    )
    np.testing.assert_array_equal(_mil_const_value(model), -x.astype(np.float32))


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
# matmul_to_conv (opt-in: lower a linear-projection MatMul to conv1x1 -- see
# convert_to_coreml's docstring and scripts/apple/README.md's "Theoretical
# ceiling" section for why). MIL's `conv` has no `value_inference`, unlike
# `matmul`/`transpose`/`const`, so `_mil_const_value` (which needs the whole
# graph to constant-fold to a single value) can't check a conv1x1 output
# directly. Instead these tests pull the `conv` op's own (constant-folded)
# *inputs* -- which do fold, since they're `transpose`/`const` outputs -- and
# apply conv1x1's documented semantics (K=1, stride=1, pad=valid: a per-position
# linear map) in plain numpy, checked against the same MatMul reference the
# rewrite is replacing.
# ---------------------------------------------------------------------------


def _build_ops(model: onnx.ModelProto, dynamic_shapes=None, matmul_to_conv=False):
    """Like ``_mil_const_value``, but returns the built MIL function itself
    (not just a folded output value) -- for inspecting *which* ops got
    emitted, or reading an intermediate (not just the final output) value.
    """
    prog, flexible_inputs = coreml_export._build_mil_program(
        model,
        *coreml_export._import_mil(),
        dynamic_shapes,
        matmul_to_conv,
    )
    return prog.functions["main"], flexible_inputs


def _matmul_projection_model(a: np.ndarray, w: np.ndarray) -> onnx.ModelProto:
    """``out = a @ w`` with both as initializers -- the shape every
    attention/MLP projection in a transformer decoder takes: ``a`` is
    ``[batch, sequence, C_in]`` (rank 3), ``w`` is a constant ``[C_in, C_out]``.
    """
    n, s, c_in = a.shape
    c_out = w.shape[1]
    inits = [numpy_helper.from_array(a, name="a"), numpy_helper.from_array(w, name="w")]
    model = _model(
        f"mm () => (float[{n},{s},{c_out}] out) {{ out = MatMul (a, w) }}",
        initializer=inits,
    )
    onnx.checker.check_model(model)
    return model


def test_matmul_to_conv_disabled_by_default():
    a = np.random.RandomState(0).randn(1, 3, 4).astype(np.float32)
    w = np.random.RandomState(1).randn(4, 5).astype(np.float32)
    model = _matmul_projection_model(a, w)

    func, _ = _build_ops(model)  # matmul_to_conv defaults to False
    op_types = [op.op_type for op in func.operations]
    assert "matmul" in op_types
    assert "conv" not in op_types
    np.testing.assert_allclose(np.asarray(func.outputs[0].val), a @ w, atol=1e-5)


def test_matmul_to_conv_rewrites_rank3_constant_projection():
    a = np.random.RandomState(0).randn(1, 3, 4).astype(np.float32)
    w = np.random.RandomState(1).randn(4, 5).astype(np.float32)
    model = _matmul_projection_model(a, w)

    func, _ = _build_ops(model, matmul_to_conv=True)
    op_types = [op.op_type for op in func.operations]
    assert "conv" in op_types
    assert "matmul" not in op_types

    (conv_op,) = [op for op in func.operations if op.op_type == "conv"]
    x_in, w_in = conv_op.inputs["x"], conv_op.inputs["weight"]
    # x transposed to conv1d's [n, C_in, L] layout; weight reshaped from
    # MatMul's [C_in, C_out] to conv's [C_out, C_in, K=1].
    np.testing.assert_allclose(x_in.val, a.transpose(0, 2, 1))
    np.testing.assert_allclose(w_in.val, w.T[:, :, None])

    # conv1x1 (K=1, stride=1, pad=valid) is, by definition, a per-position
    # linear map -- apply that directly and check it lines up with the plain
    # MatMul it's replacing.
    manual_conv_out = np.einsum("ncl,oc->nol", x_in.val, w_in.val[:, :, 0])
    np.testing.assert_allclose(manual_conv_out.transpose(0, 2, 1), a @ w, atol=1e-5)


def test_matmul_to_conv_falls_back_for_non_constant_weight():
    # Same shapes as the rewritten case above, but `w` is a declared graph
    # input (not an initializer) -- not compile-time-constant, so the
    # translator must leave this on the matmul path even with
    # matmul_to_conv=True.
    a = numpy_helper.from_array(
        np.random.RandomState(0).randn(1, 3, 4).astype(np.float32), name="a"
    )
    model = _model(
        "mm (float[4,5] w) => (float[1,3,5] out) { out = MatMul (a, w) }",
        initializer=[a],
    )
    onnx.checker.check_model(model)

    func, _ = _build_ops(model, matmul_to_conv=True)
    op_types = [op.op_type for op in func.operations]
    assert "matmul" in op_types
    assert "conv" not in op_types


def test_matmul_to_conv_falls_back_for_rank2_input():
    # Same constant-weight shape as the rewritten case, but `a` is rank 2 (no
    # separate batch/sequence axes) -- only rank-3 x is handled (see
    # _matmul_as_conv1x1's docstring), so this must still use matmul.
    a = np.random.RandomState(0).randn(3, 4).astype(np.float32)
    w = np.random.RandomState(1).randn(4, 5).astype(np.float32)
    inits = [numpy_helper.from_array(a, name="a"), numpy_helper.from_array(w, name="w")]
    model = _model(
        "mm () => (float[3,5] out) { out = MatMul (a, w) }", initializer=inits
    )
    onnx.checker.check_model(model)

    func, _ = _build_ops(model, matmul_to_conv=True)
    op_types = [op.op_type for op in func.operations]
    assert "matmul" in op_types
    assert "conv" not in op_types
    np.testing.assert_allclose(np.asarray(func.outputs[0].val), a @ w, atol=1e-5)


def test_matmul_to_conv_composes_with_dynamic_sequence_length():
    # The actual target shape this rewrite exists for: a rank-3 input whose
    # middle (sequence) axis is dynamic, exactly like export_llm_to_coreml.py's
    # KV-cache decoder graphs. `a` can't be an initializer here (dynamic_shapes
    # only applies to declared graph inputs), so this only checks the program
    # builds and emits `conv` -- not a folded numeric value.
    w = numpy_helper.from_array(
        np.random.RandomState(1).randn(4, 5).astype(np.float32), name="w"
    )
    model = _model(
        "mm (float[1,seq,4] a) => (float[1,seq,5] out) { out = MatMul (a, w) }",
        initializer=[w],
    )
    onnx.checker.check_model(model)

    func, flexible_inputs = _build_ops(
        model, dynamic_shapes={"seq": (1, 4, 16)}, matmul_to_conv=True
    )
    op_types = [op.op_type for op in func.operations]
    assert "conv" in op_types
    assert flexible_inputs is not None and len(flexible_inputs) == 1


# ---------------------------------------------------------------------------
# Dynamic shapes (opt-in flexible input dimensions, e.g. a growing KV cache)
# ---------------------------------------------------------------------------


def test_dynamic_shapes_declares_flexible_input_range():
    model = _model(
        "relu (float[N,3] x) => (float[N,3] y) { y = Relu (x) }",
    )
    onnx.checker.check_model(model)
    mlmodel = onnxsim.export_coreml(model, dynamic_shapes={"N": (1, 2, 8)})
    (in_desc,) = mlmodel.get_spec().description.input
    arr = in_desc.type.multiArrayType
    assert list(arr.shape) == [2, 3]
    assert [(r.lowerBound, r.upperBound) for r in arr.shapeRange.sizeRanges] == [
        (1, 8),
        (3, 3),
    ]


def test_dynamic_shapes_shared_dim_param_varies_together():
    # Two inputs sharing the same dim_param (like a KV cache's many
    # past_key_values.*.key/value inputs sharing `past_sequence_length`) must
    # resolve to the same symbol and flexible range.
    model = _model(
        "add (float[N,3] x, float[N,3] y) => (float[N,3] z) { z = Add (x, y) }",
    )
    onnx.checker.check_model(model)
    mlmodel = onnxsim.export_coreml(model, dynamic_shapes={"N": (1, 2, 8)})
    x_desc, y_desc = mlmodel.get_spec().description.input
    for desc in (x_desc, y_desc):
        arr = desc.type.multiArrayType
        assert [(r.lowerBound, r.upperBound) for r in arr.shapeRange.sizeRanges][0] == (
            1,
            8,
        )


def test_dynamic_shapes_composite_dim_param_needs_own_entry():
    # ONNX exporters sometimes emit a derived dim_param like
    # "past_sequence_length + sequence_length" as its own literal string (e.g. on
    # an attention mask) rather than deriving it from its terms -- giving
    # dynamic_shapes entries for "P" and "Q" alone must not satisfy "P + Q". The
    # parser can't spell a composite dim_param directly (it only accepts plain
    # identifiers in a shape), so build the base graph with a placeholder dim and
    # overwrite it with the literal composite string.
    #
    # Uses dim names ("P"/"Q") not reused by any other test in this module: a
    # RuntimeError raised partway through building the MIL program (as this one
    # deliberately triggers) leaves that dim's coremltools ``Symbol`` registered
    # process-wide with no cleanup, so a later test reusing the same name would
    # spuriously fail with "Symbol ... is used already".
    model = _model(
        "add (float[P,3] x, float[Q,3] y, float[K,3] mask) => (float[K,3] z) { z = Identity (mask) }",
    )
    for d in (
        model.graph.input[2].type.tensor_type.shape.dim[0],
        model.graph.output[0].type.tensor_type.shape.dim[0],
    ):
        d.Clear()
        d.dim_param = "P + Q"
    onnx.checker.check_model(model)

    with pytest.raises(RuntimeError, match=r"non-static dimension \('P \+ Q'\)"):
        onnxsim.export_coreml(model, dynamic_shapes={"P": (1, 2, 8), "Q": (1, 2, 8)})


def test_dynamic_axis_slice_with_runtime_only_bound():
    # Regression test for the SmolLM2 KV-cache export bug: a Slice whose `ends`
    # value is itself only known at runtime (derived from Gather(Shape(x)) of a
    # dynamically-shaped input, not a compile-time constant) used to crash with
    # "'NoneType' object is not iterable" because the translator assumed
    # `ends.val` was always available. Here `n` (fed as `ends`) is exactly such a
    # runtime-only value, and axis 0 (the sliced axis) is itself the dynamic
    # dimension.
    model = _model(
        """
        slice_dyn (float[N,8] x) => (float[N,8] y)
        {
            zero = Constant <value_ints=[0]> ()
            axis0 = Constant <value_ints=[0]> ()
            shp = Shape (x)
            idx0 = Constant <value_ints=[0]> ()
            n = Gather <axis=0> (shp, idx0)
            y = Slice (x, zero, n, axis0)
        }
        """,
    )
    onnx.checker.check_model(model)
    mlmodel = onnxsim.export_coreml(model, dynamic_shapes={"N": (1, 2, 8)})
    (in_desc,) = mlmodel.get_spec().description.input
    arr = in_desc.type.multiArrayType
    assert [(r.lowerBound, r.upperBound) for r in arr.shapeRange.sizeRanges][0] == (
        1,
        8,
    )


def test_dynamic_shapes_expand_to_runtime_shape():
    # Regression test: Expand's target shape derived from Shape(x) of a
    # dynamically-shaped input is not a compile-time constant either -- this
    # used to raise "Expand requires a compile-time-constant target 'shape'
    # input" (fixed via a fill+broadcast-add lowering for the dynamic case).
    model = _model(
        """
        expand_dyn (float[N,4] x) => (float[N,4] y)
        {
            shp = Shape (x)
            one = Constant <value_float = 1.0> ()
            y = Expand (one, shp)
        }
        """,
    )
    onnx.checker.check_model(model)
    mlmodel = onnxsim.export_coreml(model, dynamic_shapes={"N": (1, 2, 8)})
    (out_desc,) = mlmodel.get_spec().description.output
    assert out_desc.name == "y"


def test_dynamic_shapes_expand_fp16_to_runtime_shape():
    # Same scenario as test_dynamic_shapes_expand_to_runtime_shape, but fp16:
    # the fallback's `fill`+`add` used to hardcode an fp32 zero regardless of
    # `x`'s actual dtype, so this raised a dtype-mismatch error against fp16.
    model = _model(
        """
        expand_dyn (float16[N,4] x) => (float16[N,4] y)
        {
            shp = Shape (x)
            one = Constant <value_float = 1.0> ()
            one16 = Cast <to = 10> (one)
            y = Expand (one16, shp)
        }
        """,
    )
    onnx.checker.check_model(model)
    mlmodel = onnxsim.export_coreml(model, dynamic_shapes={"N": (1, 2, 8)})
    (out_desc,) = mlmodel.get_spec().description.output
    assert out_desc.name == "y"


def test_constant_of_shape_dynamic_fp16():
    # Same class of bug as the Expand case above, in ConstantOfShape's dynamic
    # `fill` fallback: extracting the fill value via numpy's `.item()` silently
    # discarded its fp16 dtype, so `fill` produced an fp32 tensor instead. The
    # text-format parser has no syntax for a node's tensor-valued attribute, so
    # this one is built with onnx.helper (see CLAUDE.md's note on that exception).
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT16, ["N", 4])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT16, ["N", 4])
    shape_node = onnx.helper.make_node("Shape", ["x"], ["shp"])
    value = numpy_helper.from_array(np.array([2.0], dtype=np.float16))
    cos_node = onnx.helper.make_node("ConstantOfShape", ["shp"], ["y"], value=value)
    graph = onnx.helper.make_graph([shape_node, cos_node], "g", [x], [y])
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    mlmodel = onnxsim.export_coreml(model, dynamic_shapes={"N": (1, 2, 8)})
    (out_desc,) = mlmodel.get_spec().description.output
    assert out_desc.name == "y"


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
