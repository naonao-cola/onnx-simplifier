"""Tests for ``onnxsim/rustnn_runtime.py``: lowering ONNX graphs onto rustnn's
native WebNN implementation (through pywebnn) and checking the results
against onnx's reference evaluator.

The op-coverage check needs neither rustnn nor pywebnn; everything that
builds a real WebNN graph skips unless ``pywebnn`` is installed and
:func:`onnxsim.rustnn_runtime.probe_rustnn` can run its canary. The
``rustnn`` job in ``.github/workflows/backend-integration.yml`` installs it
on Linux; the ``rustnn-webnn`` job in ``.github/workflows/apple-integration.yml``
runs the numerical tests on macOS for ``ONNXSIM_RUSTNN_DEVICE_TYPES=cpu,npu``
(``npu`` is Core ML in pywebnn 0.5.12).
"""

import os

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser
from onnx.reference import ReferenceEvaluator

from onnxsim import rustnn_runtime
from onnxsim.rustnn_runtime import (
    RustnnSession,
    WebnnLoweringError,
    find_unsupported_webnn_ops,
    probe_rustnn,
)


def _model(body, initializer=(), opset=18, ir_version=8):
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


def _weight(name, *shape, seed=0, positive=False):
    arr = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    return numpy_helper.from_array(np.abs(arr) + 0.5 if positive else arr, name)


@pytest.fixture(scope="module")
def rustnn_cpu():
    pytest.importorskip(
        "webnn", reason="pywebnn (rustnn's Python bindings) is not installed"
    )
    ok, reason = probe_rustnn("cpu")
    if not ok:
        pytest.skip(f"rustnn cpu context unavailable: {reason}")
    return "cpu"


# WebNN device types the numerical tests run on (comma separated). A type
# whose canary fails on this machine -- e.g. "npu" off macOS -- is skipped.
_DEVICE_TYPES = [
    d.strip()
    for d in os.environ.get("ONNXSIM_RUSTNN_DEVICE_TYPES", "cpu").split(",")
    if d.strip()
]


@pytest.fixture(scope="module", params=_DEVICE_TYPES)
def rustnn_device(request):
    pytest.importorskip(
        "webnn", reason="pywebnn (rustnn's Python bindings) is not installed"
    )
    ok, reason = probe_rustnn(request.param)
    if not ok:
        pytest.skip(f"rustnn {request.param} context unavailable: {reason}")
    return request.param


def _backend(device_type):
    probe = _model(
        """
        g (float[2] x) => (float[2] y)
        {
          y = Relu(x)
        }
        """
    )
    return RustnnSession(probe, device_type=device_type).backend_info()["backend"]


def _tolerance(device_type):
    # Core ML may compute in float16 (the Neural Engine and GPU always do).
    return 1e-4 if device_type == "cpu" else 2e-2


def test_find_unsupported_webnn_ops_lists_uncovered_op_types():
    model = _model(
        """
        g (float[1,2,4,4] x) => (float[1,2,8,8] y)
        {
          r = Relu(x)
          reps = Constant<value = int64[4] {1, 4, 1, 1}>()
          t = Tile(r, reps)
          y = DepthToSpace<blocksize = 2>(t)
        }
        """
    )
    assert find_unsupported_webnn_ops(model) == {"DepthToSpace": 1, "Tile": 1}


def test_find_unsupported_webnn_ops_flags_custom_domains():
    model = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 18, "com.microsoft": 1]>
        g (float[4] x) => (float[4] y)
        {
          y = com.microsoft.FastGelu(x)
        }
        """
    )
    assert find_unsupported_webnn_ops(model) == {"com.microsoft::FastGelu": 1}


def test_find_unsupported_webnn_ops_empty_for_covered_graph():
    model = _model(
        """
        g (float[2,3] x) => (float[3,2] y)
        {
          t = Transpose<perm = [1, 0]>(x)
          y = Relu(t)
        }
        """
    )
    assert find_unsupported_webnn_ops(model) == {}


# Each case: (model, input shapes). Covers every lowering helper at least once.
_PARITY_CASES = {
    "conv_relu_maxpool": (
        _model(
            """
            g (float[1,3,16,16] x) => (float[1,8,8,8] y)
            {
              c = Conv<pads = [1, 1, 1, 1]>(x, w, b)
              r = Relu(c)
              y = MaxPool<kernel_shape = [2, 2], strides = [2, 2]>(r)
            }
            """,
            [_weight("w", 8, 3, 3, 3), _weight("b", 8, seed=1)],
        ),
        {"x": (1, 3, 16, 16)},
    ),
    "grouped_conv_same_upper": (
        _model(
            """
            g (float[1,4,15,15] x) => (float[1,6,8,8] y)
            {
              y = Conv<auto_pad = "SAME_UPPER", strides = [2, 2], group = 2>(x, w)
            }
            """,
            [_weight("w", 6, 2, 3, 3)],
        ),
        {"x": (1, 4, 15, 15)},
    ),
    "conv_transpose": (
        _model(
            """
            g (float[1,4,5,5] x) => (float[1,3,11,11] y)
            {
              y = ConvTranspose<strides = [2, 2]>(x, w, b)
            }
            """,
            [_weight("w", 4, 3, 3, 3), _weight("b", 3, seed=1)],
        ),
        {"x": (1, 4, 5, 5)},
    ),
    "gemm_transb": (
        _model(
            """
            g (float[4,8] x) => (float[4,5] y)
            {
              y = Gemm<transB = 1, alpha = 0.5>(x, w, b)
            }
            """,
            [_weight("w", 5, 8), _weight("b", 5, seed=1)],
        ),
        {"x": (4, 8)},
    ),
    "reshape_transpose_concat": (
        _model(
            """
            g (float[2,3,4] x) => (float[2,4,6] y)
            {
              s = Constant<value = int64[3] {0, 4, -1}>()
              t = Transpose<perm = [0, 2, 1]>(x)
              r = Reshape(x, s)
              y = Concat<axis = -1>(t, r)
            }
            """
        ),
        {"x": (2, 3, 4)},
    ),
    "softmax_layernorm": (
        _model(
            """
            g (float[2,5,8] x) => (float[2,5,8] y)
            {
              s = Softmax<axis = -1>(x)
              y = LayerNormalization<axis = -1>(s, gamma, beta)
            }
            """,
            [_weight("gamma", 8), _weight("beta", 8, seed=1)],
        ),
        {"x": (2, 5, 8)},
    ),
    "strided_slice_reduce_mean": (
        _model(
            """
            g (float[3,10,6] x) => (float[3,1,3] y)
            {
              starts = Constant<value = int64[2] {1, -6}>()
              ends = Constant<value = int64[2] {9, 100}>()
              axes = Constant<value = int64[2] {1, 2}>()
              steps = Constant<value = int64[2] {2, 2}>()
              s = Slice(x, starts, ends, axes, steps)
              raxes = Constant<value = int64[1] {1}>()
              y = ReduceMean<keepdims = 1>(s, raxes)
            }
            """
        ),
        {"x": (3, 10, 6)},
    ),
    "batchnorm_gap_flatten": (
        _model(
            """
            g (float[2,4,6,6] x) => (float[2,4] y)
            {
              n = BatchNormalization(x, scale, bias, mean, var)
              p = GlobalAveragePool(n)
              y = Flatten(p)
            }
            """,
            [
                _weight("scale", 4),
                _weight("bias", 4, seed=1),
                _weight("mean", 4, seed=2),
                _weight("var", 4, seed=3, positive=True),
            ],
        ),
        {"x": (2, 4, 6, 6)},
    ),
    "squeeze_gather_unsqueeze_expand": (
        _model(
            """
            g (float[1,3,1,4] x) => (float[2,3,2] y)
            {
              saxes = Constant<value = int64[2] {0, 2}>()
              q = Squeeze(x, saxes)
              idx = Constant<value = int64[2] {-1, 1}>()
              gg = Gather<axis = 1>(q, idx)
              uaxes = Constant<value = int64[1] {0}>()
              u = Unsqueeze(gg, uaxes)
              shape = Constant<value = int64[3] {2, 3, 2}>()
              y = Expand(u, shape)
            }
            """
        ),
        {"x": (1, 3, 1, 4)},
    ),
    "clip_greater_where": (
        _model(
            """
            g (float[4,4] x) => (float[4,4] y)
            {
              lo = Constant<value = float {-0.5}>()
              hi = Constant<value = float {0.5}>()
              c = Clip(x, lo, hi)
              zero = Constant<value = float {0.0}>()
              m = Greater(x, zero)
              e = Erf(x)
              y = Where(m, c, e)
            }
            """
        ),
        {"x": (4, 4)},
    ),
    "reflect_pad_avgpool_split": (
        _model(
            """
            g (float[1,2,6,6] x) => (float[1,1,4,4] y1, float[1,1,4,4] y2)
            {
              pads = Constant<value = int64[8] {0, 0, 1, 1, 0, 0, 1, 1}>()
              p = Pad<mode = "reflect">(x, pads)
              a = AveragePool<kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]>(p)
              y1, y2 = Split<num_outputs = 2, axis = 1>(a)
            }
            """
        ),
        {"x": (1, 2, 6, 6)},
    ),
    "batched_matmul_argmax": (
        _model(
            """
            g (float[2,3,4] x) => (int64[2,3] y)
            {
              m = MatMul(x, w)
              y = ArgMax<axis = -1, keepdims = 0>(m)
            }
            """,
            [_weight("w", 4, 5)],
        ),
        {"x": (2, 3, 4)},
    ),
}


# Cases whose ops the Core ML path refuses (rustnn 0.5.12 computes them
# wrongly there; see rustnn_runtime._Lowering._reject_on_coreml callers).
_COREML_REJECTED = {
    "clip_greater_where": "Where",
    "reflect_pad_avgpool_split": "Pad",
    "softmax_layernorm": "LayerNormalization",
    "strided_slice_reduce_mean": "Slice",
}


def _check_parity(case, device_type):
    model, shapes = _PARITY_CASES[case]
    onnx.checker.check_model(model, full_check=True)
    rng = np.random.default_rng(0)
    feeds = {k: rng.standard_normal(s).astype(np.float32) for k, s in shapes.items()}
    expected = ReferenceEvaluator(model).run(None, feeds)

    session = RustnnSession(model, device_type=device_type)
    got = session.run(feeds)

    assert session.output_names == [o.name for o in model.graph.output]
    tol = _tolerance(device_type)
    for name, ref in zip(session.output_names, expected):
        assert got[name].dtype == ref.dtype
        np.testing.assert_allclose(got[name], ref, rtol=tol, atol=tol)


def _check_coreml_parity(case, device_type):
    if case in _COREML_REJECTED:
        model, _ = _PARITY_CASES[case]
        with pytest.raises(WebnnLoweringError, match="Core ML backend") as e:
            RustnnSession(model, device_type=device_type)
        assert _COREML_REJECTED[case] in str(e.value)
    else:
        _check_parity(case, device_type)


@pytest.mark.parametrize("case", sorted(_PARITY_CASES))
def test_rustnn_matches_reference(rustnn_device, case):
    if _backend(rustnn_device) == "coreml":
        _check_coreml_parity(case, rustnn_device)
    else:
        _check_parity(case, rustnn_device)


@pytest.mark.parametrize("case", sorted(_PARITY_CASES))
def test_coreml_workaround_lowering_matches_reference_on_cpu(
    rustnn_cpu, case, monkeypatch
):
    # The Core ML path rewrites the graph (biases as explicit adds, 1-D
    # outputs reshaped back, int32 arg-reductions, non-float outputs
    # returned as float32 and cast back).
    # Forcing that path on ONNX Runtime's CPU backend checks the rewrites
    # themselves are exact, independently of Core ML's own numerics.
    monkeypatch.setattr(rustnn_runtime, "_is_coreml", lambda context: True)
    _check_coreml_parity(case, rustnn_cpu)


def test_rustnn_runs_simplified_model(rustnn_device):
    # Reshape's shape comes from Shape/Gather/Concat here: WebNN has no
    # lowering for that runtime shape arithmetic, and Reshape needs a
    # constant shape, so this only lowers once onnxsim folds it.
    import onnxsim

    model = _model(
        """
        g (float[2,3,4] x) => (float[2,12] y)
        {
          s = Shape(x)
          i0 = Constant<value = int64[1] {0}>()
          d0 = Gather(s, i0)
          minus1 = Constant<value = int64[1] {-1}>()
          shape = Concat<axis = 0>(d0, minus1)
          r = Reshape(x, shape)
          y = Relu(r)
        }
        """
    )
    with pytest.raises(WebnnLoweringError, match="'Shape'"):
        RustnnSession(model, device_type=rustnn_device)

    simplified, ok = onnxsim.simplify(model)
    assert ok
    x = np.random.default_rng(0).standard_normal((2, 3, 4)).astype(np.float32)
    got = RustnnSession(simplified, device_type=rustnn_device).run({"x": x})
    tol = _tolerance(rustnn_device)
    np.testing.assert_allclose(
        got["y"], np.maximum(x.reshape(2, 12), 0), rtol=tol, atol=tol
    )


def test_rustnn_rejects_non_constant_reshape_shape(rustnn_cpu):
    model = _model(
        """
        g (float[2,6] x, int64[2] shape) => (float[3,4] y)
        {
          y = Reshape(x, shape)
        }
        """
    )
    with pytest.raises(WebnnLoweringError, match="must be a constant"):
        RustnnSession(model, device_type=rustnn_cpu)


def test_rustnn_symbolic_input_needs_input_shapes(rustnn_cpu):
    model = _model(
        """
        g (float[N,3] x) => (float[N,3] y)
        {
          y = Relu(x)
        }
        """
    )
    with pytest.raises(WebnnLoweringError, match="non-static dimension"):
        RustnnSession(model, device_type=rustnn_cpu)

    session = RustnnSession(model, device_type=rustnn_cpu, input_shapes={"x": [2, 3]})
    x = np.array([[-1, 0, 1], [2, -2, 3]], np.float32)
    np.testing.assert_array_equal(session.run({"x": x})["y"], np.maximum(x, 0))


def test_rustnn_rejects_3d_conv(rustnn_cpu):
    model = _model(
        """
        g (float[1,1,4,4,4] x) => (float[1,1,2,2,2] y)
        {
          y = Conv(x, w)
        }
        """,
        [_weight("w", 1, 1, 3, 3, 3)],
    )
    with pytest.raises(WebnnLoweringError, match="only 2-D"):
        RustnnSession(model, device_type=rustnn_cpu)


def test_rustnn_benchmark_reports_timing(rustnn_device):
    model = _model(
        """
        g (float[8,8] x) => (float[8,8] y)
        {
          y = MatMul(x, x)
        }
        """
    )
    x = np.eye(8, dtype=np.float32)
    timing, out = RustnnSession(model, device_type=rustnn_device).benchmark(
        {"x": x}, warmup=1, runs=3
    )
    assert timing.runs == 3
    assert 0 < timing.min_ms <= timing.median_ms
    np.testing.assert_allclose(out["y"], x, atol=_tolerance(rustnn_device))


def test_probe_rustnn_reports_missing_pywebnn(monkeypatch):
    def missing():
        raise ImportError("no pywebnn")

    monkeypatch.setattr(rustnn_runtime, "_require_webnn", missing)
    probe_rustnn.cache_clear()
    try:
        ok, reason = probe_rustnn("cpu", "auto")
    finally:
        probe_rustnn.cache_clear()
    assert not ok
    assert "no pywebnn" in reason
