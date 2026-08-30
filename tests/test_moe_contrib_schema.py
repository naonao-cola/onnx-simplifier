"""Tests for the ``com.microsoft`` ``MoE``/``QMoE`` schema registration added
to ``onnxsim/contrib_schemas.cpp``.

Before this, a model containing a real ONNX Runtime Mixture-of-Experts node
was opaque to onnxsim: its schema lived outside ONNX (ONNX Runtime's own
``contrib_defs.cc``), so ``onnx::shape_inference::InferShapes`` stopped dead
at the node -- the same problem ``QLinearAdd`` and ``Attention`` had before
their own schemas were registered here (see this file's own comment and
``tests/test_fuse_attention.py``). These tests check that a bare ``MoE``/
``QMoE`` node's output shape is now resolved by plain ``onnxsim.simplify()``,
the same way the other contrib-op registrations in this file are tested.

Unlike ``test_fuse_attention.py``/``test_dynamic_quantize_attention.py``,
``MoE`` additionally carries a context-dependent reference *decomposition*
(``BuildMoEFunctionBody`` in ``contrib_schemas.cpp``) into plain ONNX ops --
nothing in onnxsim's own simplification pipeline inlines schema-level
function bodies today (``include_inline_functions`` only inlines a model's
own ``model.functions()``, see ``onnxsim.cpp``), so that decomposition
doesn't yet change what ``simplify()`` produces. Its numeric correctness
against ONNX Runtime's real MoE kernel is covered by
``onnxsim/contrib_schemas_moe_test.cpp`` (structural checks) and was
additionally verified by hand against a real ``onnxruntime`` session; see
that file's top comment for what a full wire-up would need.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=18, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _moe_weights(rng, num_experts, hidden_size, inter_size):
    fc1 = _f32(
        rng.standard_normal((num_experts, inter_size, hidden_size)) * 0.1, "fc1_w"
    )
    fc2 = _f32(
        rng.standard_normal((num_experts, hidden_size, inter_size)) * 0.1, "fc2_w"
    )
    return [fc1, fc2]


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def test_moe_output_shape_is_inferred():
    num_tokens, hidden_size, inter_size, num_experts, k = 4, 6, 8, 3, 2
    model = _model(
        f"""
        agraph (float[{num_tokens},{hidden_size}] input,
                float[{num_tokens},{num_experts}] router_probs)
              => (float[?,?] output)
        {{
          moe_out = com.microsoft.MoE
              <k: int = {k}, activation_type: string = "relu",
               normalize_routing_weights: int = 1>
              (input, router_probs, fc1_w, , fc2_w)
          output = Identity(moe_out)
        }}
        """,
        initializer=_moe_weights(
            np.random.default_rng(0), num_experts, hidden_size, inter_size
        ),
    )
    onnx.checker.check_model(model)

    simplified, ok = onnxsim.simplify(
        model, check_n=0, perform_optimization=False, skip_constant_folding=True
    )
    assert ok
    out_shape = simplified.graph.output[0].type.tensor_type.shape
    dims = [
        d.dim_value if d.HasField("dim_value") else d.dim_param for d in out_shape.dim
    ]
    assert dims == [num_tokens, hidden_size]


def test_moe_output_shape_is_inferred_for_3d_input():
    # (batch_size, sequence_length, hidden_size), the other shape MoE's own
    # schema documents.
    batch, seq, hidden_size, inter_size, num_experts, k = 2, 3, 6, 8, 3, 2
    num_tokens = batch * seq
    model = _model(
        f"""
        agraph (float[{batch},{seq},{hidden_size}] input,
                float[{num_tokens},{num_experts}] router_probs)
              => (float[?,?,?] output)
        {{
          moe_out = com.microsoft.MoE
              <k: int = {k}, activation_type: string = "gelu",
               normalize_routing_weights: int = 0>
              (input, router_probs, fc1_w, , fc2_w)
          output = Identity(moe_out)
        }}
        """,
        initializer=_moe_weights(
            np.random.default_rng(1), num_experts, hidden_size, inter_size
        ),
    )
    onnx.checker.check_model(model)

    simplified, ok = onnxsim.simplify(
        model, check_n=0, perform_optimization=False, skip_constant_folding=True
    )
    assert ok
    out_shape = simplified.graph.output[0].type.tensor_type.shape
    dims = [
        d.dim_value if d.HasField("dim_value") else d.dim_param for d in out_shape.dim
    ]
    assert dims == [batch, seq, hidden_size]


def test_qmoe_output_shape_is_inferred():
    # QMoE is registered for shape inference only (no reference
    # decomposition -- dequantizing its many quant_type layouts is out of
    # scope, see contrib_schemas.cpp's MakeQMoESchema comment), but its
    # output shape should still resolve the same way MoE's does.
    num_tokens, hidden_size, num_experts, k = 4, 6, 3, 2
    model = _model(
        f"""
        agraph (float[{num_tokens},{hidden_size}] input,
                float[{num_tokens},{num_experts}] router_probs)
              => (float[?,?] output)
        {{
          moe_out = com.microsoft.QMoE <k: int = {k}, activation_type: string = "relu">
              (input, router_probs, fc1_w, , , fc2_w)
          output = Identity(moe_out)
        }}
        """
    )
    # fc1_w/fc2_w are left as graph inputs (uint8, opaque packed layout) --
    # this test only cares about the *output* shape, not running the model.
    fc1_w = onnx.helper.make_tensor_value_info("fc1_w", onnx.TensorProto.UINT8, ["?"])
    fc2_w = onnx.helper.make_tensor_value_info("fc2_w", onnx.TensorProto.UINT8, ["?"])
    model.graph.input.extend([fc1_w, fc2_w])
    onnx.checker.check_model(model)

    simplified, ok = onnxsim.simplify(
        model, check_n=0, perform_optimization=False, skip_constant_folding=True
    )
    assert ok
    out_shape = simplified.graph.output[0].type.tensor_type.shape
    dims = [
        d.dim_value if d.HasField("dim_value") else d.dim_param for d in out_shape.dim
    ]
    assert dims == [num_tokens, hidden_size]


def test_moe_with_fc3_output_shape_is_inferred():
    # com.microsoft.MoE with a separate fc3 weight and activation_type="silu"
    # is the real Mixtral-style gate/up/down-projection convention
    # onnxruntime-genai's Phi-3.5-MoE builder exports (see
    # generate_moe_function_templates.py's module docstring for the fc3(x) *
    # silu(fc1(x)) formula and where it's transcribed from). Unlike
    # test_simplify_round_trips_a_runnable_moe_model below, this can't be
    # round-tripped through a real onnxruntime CPU session: ONNX Runtime's
    # own CPU MoE kernel rejects fc3 unconditionally ("FC3 is not
    # implemented for CPU MoE"), so check_n=0 here only exercises shape
    # inference -- the same as test_qmoe_output_shape_is_inferred above, and
    # for the same reason (no CPU-runnable reference to check_n against).
    num_tokens, hidden_size, inter_size, num_experts, k = 4, 6, 8, 3, 2
    fc3_w = _f32(
        np.random.default_rng(4).standard_normal((num_experts, inter_size, hidden_size))
        * 0.1,
        "fc3_w",
    )
    model = _model(
        f"""
        agraph (float[{num_tokens},{hidden_size}] input,
                float[{num_tokens},{num_experts}] router_probs)
              => (float[?,?] output)
        {{
          moe_out = com.microsoft.MoE
              <k: int = {k}, activation_type: string = "silu",
               normalize_routing_weights: int = 0>
              (input, router_probs, fc1_w, , fc2_w, , fc3_w)
          output = Identity(moe_out)
        }}
        """,
        initializer=[
            *_moe_weights(
                np.random.default_rng(4), num_experts, hidden_size, inter_size
            ),
            fc3_w,
        ],
    )
    onnx.checker.check_model(model)

    simplified, ok = onnxsim.simplify(
        model, check_n=0, perform_optimization=False, skip_constant_folding=True
    )
    assert ok
    out_shape = simplified.graph.output[0].type.tensor_type.shape
    dims = [
        d.dim_value if d.HasField("dim_value") else d.dim_param for d in out_shape.dim
    ]
    assert dims == [num_tokens, hidden_size]


def test_simplify_round_trips_a_runnable_moe_model():
    # onnxsim.simplify's own check_n mechanism runs the model through
    # onnxruntime before and after simplification and compares outputs --
    # this both confirms simplify() doesn't corrupt a real MoE node it now
    # understands the shape of, and exercises ONNX Runtime's actual native
    # MoE kernel (distinct from BuildMoEFunctionBody's reference body, which
    # nothing in this call path invokes).
    num_tokens, hidden_size, inter_size, num_experts, k = 4, 6, 8, 3, 2
    model = _model(
        f"""
        agraph (float[{num_tokens},{hidden_size}] input,
                float[{num_tokens},{num_experts}] router_probs)
              => (float[{num_tokens},{hidden_size}] output)
        {{
          output = com.microsoft.MoE
              <k: int = {k}, activation_type: string = "silu",
               normalize_routing_weights: int = 1>
              (input, router_probs, fc1_w, , fc2_w)
        }}
        """,
        initializer=_moe_weights(
            np.random.default_rng(2), num_experts, hidden_size, inter_size
        ),
    )
    onnx.checker.check_model(model)

    simplified, ok = onnxsim.simplify(model, check_n=3)
    assert ok

    rng = np.random.default_rng(3)
    feeds = {
        "input": rng.standard_normal((num_tokens, hidden_size)).astype(np.float32),
        "router_probs": rng.standard_normal((num_tokens, num_experts)).astype(
            np.float32
        ),
    }
    before = _run(model, feeds)
    after = _run(simplified, feeds)
    for b, a in zip(before, after):
        np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-5)


def test_simplify_round_trips_a_runnable_swiglu_moe_model():
    # swiglu (interleaved, swiglu_fusion=1) is gpt-oss-20b's real
    # convention, and -- unlike fc3/use_sparse_mixer -- the one activation
    # ONNX Runtime's own CPU MoE kernel actually implements (its
    # constructor throws unless swiglu_fusion == 1 for a SwiGLU node), so
    # this can be round-tripped through a real onnxruntime session the same
    # way test_simplify_round_trips_a_runnable_moe_model does for silu.
    # fc1_w is twice as wide here (fusion_size=2: gate/linear interleaved
    # column by column in its own Gemm output).
    num_tokens, hidden_size, inter_size, num_experts, k = 4, 6, 8, 3, 2
    fc1_w = _f32(
        np.random.default_rng(6).standard_normal(
            (num_experts, 2 * inter_size, hidden_size)
        )
        * 0.1,
        "fc1_w",
    )
    fc2_w = _f32(
        np.random.default_rng(7).standard_normal((num_experts, hidden_size, inter_size))
        * 0.1,
        "fc2_w",
    )
    model = _model(
        f"""
        agraph (float[{num_tokens},{hidden_size}] input,
                float[{num_tokens},{num_experts}] router_probs)
              => (float[{num_tokens},{hidden_size}] output)
        {{
          output = com.microsoft.MoE
              <k: int = {k}, activation_type: string = "swiglu",
               swiglu_fusion: int = 1, activation_alpha: float = 1.702,
               activation_beta: float = 1.0, normalize_routing_weights: int = 1>
              (input, router_probs, fc1_w, , fc2_w)
        }}
        """,
        initializer=[fc1_w, fc2_w],
    )
    onnx.checker.check_model(model)

    simplified, ok = onnxsim.simplify(model, check_n=3)
    assert ok

    rng = np.random.default_rng(8)
    feeds = {
        "input": rng.standard_normal((num_tokens, hidden_size)).astype(np.float32),
        "router_probs": rng.standard_normal((num_tokens, num_experts)).astype(
            np.float32
        ),
    }
    before = _run(model, feeds)
    after = _run(simplified, feeds)
    for b, a in zip(before, after):
        np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-5)
