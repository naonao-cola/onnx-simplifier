"""Tests for onnxsim's TVM-style RPC (``onnxsim.rpc``): a loopback server, a client session, a
tracker, and remote constant folding through ``onnxsim.simplify``."""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
import onnxsim.rpc as rpc
from onnxsim.rpc import _protocol as proto
from onnxsim.rpc.server import RPCServer
from onnxsim.rpc.tracker import Tracker

pytest.importorskip("onnxruntime")


def _model(body: str, opset: int = 13) -> onnx.ModelProto:
    return parser.parse_model(f'<ir_version: 8, opset_import: ["" : {opset}]> {body}')


ADD_RELU = """
agraph (float[2,3] x, float[2,3] y) => (float[2,3] z) {
  s = Add(x, y)
  z = Relu(s)
}
"""

FOLDABLE = """
agraph (float[2,3] x) => (float[2,3] z) {
  a = Constant<value = float[2,3] {1, 2, 3, 4, 5, 6}>()
  b = Constant<value = float[2,3] {6, 5, 4, 3, 2, 1}>()
  c = Mul(a, b)
  z = Add(x, c)
}
"""


@pytest.fixture()
def server(tmp_path):
    srv = RPCServer(
        "127.0.0.1", 0, key="testdev", work_dir=str(tmp_path / "ws")
    ).start()
    yield srv
    srv.stop()


def _connect(srv, key="testdev"):
    host, port = srv.address
    return rpc.connect(host, port, key=key)


def test_protocol_roundtrips_every_dtype_and_byte_order():
    tensors = {
        "f": np.arange(6, dtype=np.float32).reshape(2, 3),
        "h": np.array([1.5, -2.0], dtype=np.float16),
        "i": np.array([[1, -2]], dtype=np.int64),
        "b": np.array([True, False]),
        "big": np.arange(4, dtype=">i4"),
    }
    specs, blobs = proto.encode_tensors(tensors)
    back = proto.decode_tensors(specs, blobs)
    for name, value in tensors.items():
        np.testing.assert_array_equal(back[name], value)
    with pytest.raises(proto.RPCError):
        proto.encode_tensors({"s": np.array(["x"])})
    with pytest.raises(proto.RPCError):
        proto.decode_tensors(
            [{"name": "f", "dtype": "float32", "shape": [2]}], [b"\0" * 4]
        )


def test_connect_reports_server_info(server):
    with _connect(server) as session:
        assert session.info["key"] == "testdev"
        assert session.info["onnx"] == onnx.__version__
        assert "CPUExecutionProvider" in session.info["providers"]


def test_wrong_key_is_rejected(server):
    with pytest.raises(rpc.RPCError, match="handshake rejected"):
        _connect(server, key="other")


def test_upload_load_run_matches_local_onnxruntime(server, tmp_path):
    import onnxruntime as ort

    model = _model(ADD_RELU)
    path = tmp_path / "add_relu.onnx"
    onnx.save(model, path)
    x = np.random.default_rng(0).normal(size=(2, 3)).astype("float32")
    y = np.random.default_rng(1).normal(size=(2, 3)).astype("float32")
    expected = ort.InferenceSession(model.SerializeToString()).run(
        None, {"x": x, "y": y}
    )[0]
    with _connect(server) as session:
        name = session.upload(path)
        assert name == "add_relu.onnx"
        remote_model = session.load_model(name)  # by uploaded name
        np.testing.assert_array_equal(remote_model.run({"x": x, "y": y})["z"], expected)
        # bytes, ModelProto and local paths load too
        for source in (model, model.SerializeToString(), str(path)):
            got = session.load_model(source).run({"x": x, "y": y})["z"]
            np.testing.assert_array_equal(got, expected)
        remote_model.close()


def test_time_evaluator_returns_repeat_results(server):
    x = np.ones((2, 3), dtype="float32")
    with _connect(server) as session:
        remote_model = session.load_model(_model(ADD_RELU))
        timing = remote_model.time_evaluator({"x": x, "y": x}, number=3, repeat=4)
    assert len(timing.results) == 4
    assert 0 < timing.min <= timing.median <= timing.max
    assert timing.mean > 0 and timing.std >= 0


def test_large_tensor_roundtrip(server):
    model = _model("agraph (float[512,1024] x) => (float[512,1024] y) { y = Relu(x) }")
    x = np.random.default_rng(2).normal(size=(512, 1024)).astype("float32")
    with _connect(server) as session:
        out = session.load_model(model).run({"x": x})["y"]
    np.testing.assert_array_equal(out, np.maximum(x, 0))


def test_remote_errors_propagate_and_the_session_survives(server):
    with _connect(server) as session:
        with pytest.raises(rpc.RPCError):
            session.load_model(b"not an onnx model")
        with pytest.raises(rpc.RPCError, match="no such model handle"):
            rpc.RemoteModel(session, 999).run({})
        with pytest.raises(rpc.RPCError, match="not available"):
            session.load_model(_model(ADD_RELU), providers=["NoSuchExecutionProvider"])
        with pytest.raises(rpc.RPCError):
            session.load_model("missing-file.onnx")
        x = np.ones((2, 3), dtype="float32")
        got = session.load_model(_model(ADD_RELU)).run({"x": x, "y": x})[
            "z"
        ]  # still usable
        np.testing.assert_array_equal(got, 2 * x)


def test_upload_names_cannot_escape_the_workspace(server, tmp_path):
    with _connect(server) as session:
        assert session.upload(b"data", name="../../evil.bin") == "evil.bin"
        assert session.upload(b"data", name="a b/c?.bin") == "c_.bin"
    workspace = tmp_path / "ws"
    assert sorted(p.name for p in workspace.iterdir()) == ["c_.bin", "evil.bin"]
    assert not (tmp_path / "evil.bin").exists()


def test_tracker_hands_out_servers_round_robin_by_key(tmp_path):
    tracker = Tracker("127.0.0.1", 0).start()
    a = RPCServer("127.0.0.1", 0, key="phone", work_dir=str(tmp_path / "a")).start()
    b = RPCServer("127.0.0.1", 0, key="phone", work_dir=str(tmp_path / "b")).start()
    c = RPCServer("127.0.0.1", 0, key="board", work_dir=str(tmp_path / "c")).start()
    try:
        for srv in (a, b, c):
            srv.register_with_tracker(tracker.address)
        client = rpc.connect_tracker(*tracker.address)
        import time

        deadline = time.time() + 5
        while (
            time.time() < deadline
            and sum(len(v) for v in client.summary().values()) < 3
        ):
            time.sleep(0.05)
        assert sorted(client.summary()) == ["board", "phone"]
        ports = set()
        for _ in range(2):
            with client.request("phone") as session:
                ports.add(session.info["key"])
        assert ports == {"phone"}
        with pytest.raises(rpc.RPCError, match="no server with key"):
            client.request("absent")
    finally:
        for srv in (a, b, c):
            srv.stop()
        tracker.stop()


def test_remote_executor_folds_constants_on_the_server(server):
    model = _model(FOLDABLE)
    local, ok = onnxsim.simplify(model)
    assert ok
    before = server.stats.get("run_once", 0)
    with _connect(server) as session, rpc.remote_executor(session):
        remote, ok = onnxsim.simplify(model)
    assert ok
    assert server.stats.get("run_once", 0) > before  # folding really ran on the server
    assert [n.op_type for n in remote.graph.node] == [
        n.op_type for n in local.graph.node
    ]
    constants = [
        onnx.numpy_helper.to_array(n.attribute[0].t)
        for n in remote.graph.node
        if n.op_type == "Constant"
    ]
    assert len(constants) == 1
    np.testing.assert_array_equal(
        constants[0], np.array([[6, 10, 12], [12, 10, 6]], dtype="float32")
    )


def test_executor_override_is_scoped(server):
    from onnxsim import onnx_simplifier

    assert onnx_simplifier._executor_override.get() is None
    with _connect(server) as session:
        with rpc.remote_executor(session) as executor:
            assert onnx_simplifier._get_model_executor() is executor
        assert onnx_simplifier._executor_override.get() is None
