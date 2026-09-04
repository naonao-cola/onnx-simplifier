"""Tests for the ``rewrite_bool_where`` C++ pass
(onnxsim/passes/rewrite_bool_where.h).

ONNX's operator spec allows a ``Where`` node's two *data* operands (``x``/
``y`` -- not ``cond``, which the spec always types ``bool``) to themselves be
``bool`` tensors. ONNX Runtime's CPU execution provider, however, only
registers a ``Where`` kernel for
``string``/``float``/``double``/``int32``/``int64``/``uint8`` -- confirmed
straight from ``onnxruntime/core/providers/cpu/cpu_execution_provider.cc`` --
so a spec-legal, ``onnx.checker``-clean model with a bool-operand ``Where``
fails to even *load* on ORT's CPU EP with "NOT_IMPLEMENTED ... Could not find
an implementation for Where". onnxsim rewrites such a node into
``Cast<to=BOOL>(Where(cond, Cast<to=INT32>(x), Cast<to=INT32>(y)))``, which
computes the same result through a type ORT's CPU EP does support.

This surfaces in practice via jax2onnx's export of JAX/Flax attention-mask
code that combines two boolean masks with a ``jnp.where`` ahead of a later
cast to a float bias (see tests/test_jax_real_model_integration.py's comment
on why FlaxBertModel/Gemma aren't covered there yet).
"""

import numpy as np
import onnx
import pytest
from onnx import parser

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=13, ir_version=10):
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
    return model


def _ort_session(model):
    return ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def test_bool_where_fails_to_load_on_ort_before_the_rewrite():
    # Anchor for the bug this pass works around: a spec-legal, checker-clean
    # bool-operand Where is rejected by ORT's CPU EP at session-creation time,
    # not merely slow or numerically off.
    model = _model(
        """
        g (bool[4] cond, bool[4] x, bool[4] y) => (bool[4] z)
        {
          z = Where(cond, x, y)
        }
        """
    )
    onnx.checker.check_model(model)
    with pytest.raises(Exception, match="Where"):
        _ort_session(model)


def test_bool_where_operands_rewritten_for_ort():
    model = _model(
        """
        g (bool[4] cond, bool[4] x, bool[4] y) => (bool[4] z)
        {
          z = Where(cond, x, y)
        }
        """
    )
    onnx.checker.check_model(model)

    # check_n=0: onnxsim's own correctness check runs the *original* model
    # through the same (onnxruntime) backend for comparison, which would hit
    # the exact same load failure this pass exists to route around -- that
    # limitation is orthogonal to this pass, which only needs to fix the
    # *simplified* output. Numeric correctness is instead checked directly
    # below, against plain numpy.
    sim_model, check_ok = onnxsim.simplify(model, check_n=0)
    assert check_ok
    onnx.checker.check_model(sim_model)

    # No Where node should still have bool data operands: either the node is
    # gone (fully constant-folded, not expected here since cond/x/y are graph
    # inputs) or it now reads/writes int32 around Cast<BOOL> nodes.
    value_types = {
        vi.name: vi.type.tensor_type.elem_type
        for vi in list(sim_model.graph.input)
        + list(sim_model.graph.value_info)
        + list(sim_model.graph.output)
    }
    for node in sim_model.graph.node:
        if node.op_type != "Where":
            continue
        for data_input in node.input[1:]:
            assert value_types.get(data_input) != onnx.TensorProto.BOOL

    sess = _ort_session(sim_model)
    cond = np.array([True, False, True, False])
    x = np.array([True, True, False, False])
    y = np.array([False, False, True, True])
    (out,) = sess.run(None, {"cond": cond, "x": x, "y": y})
    np.testing.assert_array_equal(out, np.where(cond, x, y))


def test_bool_where_broadcasting_preserved():
    # cond/x/y have different (numpy-broadcastable) shapes; the rewrite must
    # not disturb Where's own broadcasting -- only the two data operands'
    # dtype changes, never their shape.
    model = _model(
        """
        g (bool[1,4] cond, bool[4] x, bool[3,1] y) => (bool[3,4] z)
        {
          z = Where(cond, x, y)
        }
        """
    )
    onnx.checker.check_model(model)

    sim_model, check_ok = onnxsim.simplify(model, check_n=0)
    assert check_ok
    onnx.checker.check_model(sim_model)

    sess = _ort_session(sim_model)
    cond = np.array([[True, False, True, False]])
    x = np.array([True, True, False, False])
    y = np.array([[True], [False], [True]])
    (out,) = sess.run(None, {"cond": cond, "x": x, "y": y})
    np.testing.assert_array_equal(out, np.where(cond, x, y))
