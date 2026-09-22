"""onnxsim RPC server: run ONNX models on the machine it is started on.

The TVM-style workflow -- start a small server on the target device, forward its port (for
example ``adb forward tcp:9090 tcp:9090``), then upload, run and time models from the host --
without TVM's compiler stack. Models execute with onnxruntime when it is installed (one cached
session per loaded model, so timing excludes session creation) and fall back to onnxsim's
pure-Python reference evaluator otherwise.

Security: a server executes whatever ONNX model a client sends it. It binds to loopback by
default, the shared ``key`` is only an identifier (like TVM's device key), not authentication, and
onnxruntime custom-op libraries are never loaded. Expose it only on a trusted link.
"""

from __future__ import annotations

import os
import platform
import re
import socket
import socketserver
import statistics
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx

from . import _protocol as proto


def _sanitize(name: str) -> str:
    base = os.path.basename(name.replace("\\", "/"))
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip(".")
    if not base:
        raise proto.RPCError(f"invalid file name {name!r}")
    return base


class _Runner:
    """One loaded model: an onnxruntime session when available, else the reference evaluator."""

    def __init__(
        self,
        model_bytes: bytes,
        providers: Optional[Sequence[str]],
        single_threaded: bool,
    ):
        self.model_bytes = model_bytes
        self.providers = list(providers) if providers else None
        self.single_threaded = single_threaded
        self.session = None
        try:
            import onnxruntime as ort
        except ImportError:
            return
        available = ort.get_available_providers()
        chosen = self.providers or ["CPUExecutionProvider"]
        unknown = [p for p in chosen if p not in available]
        if unknown:
            raise proto.RPCError(
                f"execution providers {unknown} are not available here: {available}"
            )
        options = ort.SessionOptions()
        options.log_severity_level = 3
        if single_threaded:
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_bytes, options, providers=chosen)

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.session is not None:
            names = [o.name for o in self.session.get_outputs()]
            return dict(zip(names, self.session.run(None, inputs)))
        from onnxsim import backend

        return dict(backend.run_model(model_bytes_to_proto(self.model_bytes), inputs))


def model_bytes_to_proto(data: bytes) -> onnx.ModelProto:
    model = onnx.ModelProto()
    model.ParseFromString(data)
    return model


class _Handler(socketserver.BaseRequestHandler):
    server: "RPCServer"

    def handle(self) -> None:
        sock = self.request
        models: Dict[int, _Runner] = {}
        counter = 0
        try:
            first, _ = proto.recv_message(sock, self.server.max_blob_bytes)
            client_key = first.get("key", "")
            if first.get("op") != "hello" or (
                self.server.key and client_key != self.server.key
            ):
                proto.send_message(
                    sock,
                    {
                        "ok": False,
                        "error": f"handshake rejected (server key {self.server.key!r})",
                    },
                )
                return
            proto.send_message(sock, {"ok": True, "info": self.server.info()})
            while True:
                try:
                    header, blobs = proto.recv_message(sock, self.server.max_blob_bytes)
                except ConnectionError:
                    return
                op = header.get("op")
                if op == "close":
                    proto.send_message(sock, {"ok": True})
                    return
                try:
                    reply, out_blobs = self._dispatch(
                        op, header, blobs, models, counter
                    )
                    if op == "load_model":
                        counter = reply["handle"]
                    reply["ok"] = True
                except Exception as error:  # noqa: BLE001 - reported to the client
                    if self.server.verbose:
                        traceback.print_exc()
                    reply, out_blobs = (
                        {"ok": False, "error": f"{type(error).__name__}: {error}"},
                        [],
                    )
                proto.send_message(sock, reply, out_blobs)
        except (ConnectionError, proto.RPCError, OSError):
            return
        finally:
            models.clear()

    # ---- operations -----------------------------------------------------------------------

    def _dispatch(
        self, op, header, blobs, models, counter
    ) -> Tuple[Dict[str, Any], List[bytes]]:
        server = self.server
        server.stats[op] = server.stats.get(op, 0) + 1
        if op == "info":
            return {"info": server.info()}, []
        if op == "upload":
            if len(blobs) != 1:
                raise proto.RPCError("upload takes exactly one blob")
            name = _sanitize(header["name"])
            path = os.path.join(server.work_dir, name)
            with open(path, "wb") as f:
                f.write(blobs[0])
            return {"name": name, "size": len(blobs[0])}, []
        if op == "load_model":
            if blobs:
                data = blobs[0]
            else:
                with open(
                    os.path.join(server.work_dir, _sanitize(header["name"])), "rb"
                ) as f:
                    data = f.read()
            runner = _Runner(
                data, header.get("providers"), bool(header.get("single_threaded"))
            )
            handle = counter + 1
            models[handle] = runner
            return {"handle": handle}, []
        if op == "unload":
            models.pop(int(header["handle"]), None)
            return {}, []
        if op == "run":
            runner = self._model(models, header)
            outputs = runner.run(proto.decode_tensors(header["tensors"], blobs))
            specs, out_blobs = proto.encode_tensors(outputs)
            return {"tensors": specs}, out_blobs
        if op == "time":
            runner = self._model(models, header)
            inputs = proto.decode_tensors(header["tensors"], blobs)
            number, repeat = (
                max(int(header.get("number", 1)), 1),
                max(int(header.get("repeat", 1)), 1),
            )
            runner.run(inputs)  # warm-up, as TVM's time_evaluator does before measuring
            results = []
            for _ in range(repeat):
                start = time.perf_counter()
                for _ in range(number):
                    runner.run(inputs)
                results.append((time.perf_counter() - start) / number)
            return {"results": results, "median": statistics.median(results)}, []
        if op == "run_once":
            # One-shot execution without a handle: the model rides in blob 0, inputs after it.
            runner = _Runner(
                blobs[0], header.get("providers"), bool(header.get("single_threaded"))
            )
            outputs = runner.run(proto.decode_tensors(header["tensors"], blobs[1:]))
            specs, out_blobs = proto.encode_tensors(outputs)
            return {"tensors": specs}, out_blobs
        raise proto.RPCError(f"unknown operation {op!r}")

    @staticmethod
    def _model(models: Dict[int, _Runner], header: Dict[str, Any]) -> _Runner:
        try:
            return models[int(header["handle"])]
        except KeyError:
            raise proto.RPCError(
                f"no such model handle {header.get('handle')!r}"
            ) from None


class RPCServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        key: str = "",
        work_dir: Optional[str] = None,
        max_blob_bytes: int = proto.DEFAULT_MAX_BLOB_BYTES,
        verbose: bool = False,
    ):
        super().__init__((host, port), _Handler)
        self.key = key
        self.work_dir = work_dir or os.path.join(os.getcwd(), "onnxsim_rpc_workspace")
        os.makedirs(self.work_dir, exist_ok=True)
        self.max_blob_bytes = max_blob_bytes
        self.verbose = verbose
        self.stats: Dict[str, int] = {}
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> Tuple[str, int]:
        return self.server_address[0], self.server_address[1]

    def info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "protocol": 1,
            "key": self.key,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "onnx": onnx.__version__,
        }
        try:
            import onnxruntime as ort

            info["onnxruntime"] = ort.__version__
            info["providers"] = ort.get_available_providers()
        except ImportError:
            info["onnxruntime"] = None
            info["providers"] = []
        try:
            import onnxsim

            info["onnxsim"] = onnxsim.__version__
        except Exception:  # noqa: BLE001
            info["onnxsim"] = None
        return info

    def start(self) -> "RPCServer":
        """Serve in a background thread (for tests and embedding)."""
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.shutdown()
        self.server_close()

    def register_with_tracker(
        self,
        tracker: Tuple[str, int],
        advertise_host: Optional[str] = None,
        retry: float = 2.0,
    ) -> threading.Thread:
        """Keep this server registered with a tracker (re-registers if the tracker restarts)."""
        host = advertise_host or _default_advertise_host(self.address[0])

        def loop():
            while True:
                try:
                    sock = socket.create_connection(tracker, timeout=10)
                    sock.settimeout(None)
                    proto.send_message(
                        sock,
                        {
                            "op": "register",
                            "key": self.key,
                            "addr": [host, self.address[1]],
                        },
                    )
                    proto.recv_message(sock)
                    while sock.recv(
                        1
                    ):  # the tracker never writes again; blocks until it closes
                        pass
                except OSError:
                    pass
                time.sleep(retry)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        return thread


def _default_advertise_host(bound: str) -> str:
    if bound not in ("0.0.0.0", "::", ""):
        return bound
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
