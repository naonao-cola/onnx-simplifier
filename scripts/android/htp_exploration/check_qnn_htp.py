"""Probe whether ONNX Runtime's QNN Execution Provider actually routes a model to
the Hexagon Tensor Processor (HTP) backend, on THIS host. Requires the
`onnxruntime_qnn` package (see README.md's "environment" section for where a
working install already exists on this machine).

Prints, per model: whether the QNN EP registers at all, what OrtEpDevices it
exposes (which is what determines whether HTP is even selectable here -- see
README.md's "Stage 1" finding), and the resulting session's actual provider
placement (does the model run on QNN, or silently fall back to CPU).
"""

import argparse
import sys

import onnxruntime as ort
import onnxruntime_qnn as qnn_ep


def probe(model_path: str) -> None:
    ort.register_execution_provider_library(qnn_ep.get_ep_name(), qnn_ep.get_library_path())

    print("EP devices visible to this process:")
    for d in ort.get_ep_devices():
        print(f"  ep={d.ep_name!r} vendor={d.ep_vendor!r} device_type={d.device.type} device_vendor={d.device.vendor}")

    print("\n--- providers=[QNNExecutionProvider] (legacy API) ---")
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3  # errors only; flip to 0 for full graph-partition trace
        sess = ort.InferenceSession(
            model_path, sess_options=so, providers=["QNNExecutionProvider"],
            provider_options=[{"backend_path": qnn_ep.get_qnn_htp_path()}],
        )
        print("session created; actual providers used:", sess.get_providers())
    except Exception as e:  # noqa: BLE001 -- reporting whatever QNN raises verbatim
        print("FAILED:", type(e).__name__, str(e)[:300])

    print("\n--- SessionOptions.add_provider (current plugin-EP API) ---")
    try:
        so = ort.SessionOptions()
        so.add_provider(qnn_ep.get_ep_name(), {"backend_path": qnn_ep.get_qnn_htp_path()})
        sess = ort.InferenceSession(model_path, sess_options=so)
        print("session created; actual providers used:", sess.get_providers())
    except Exception as e:  # noqa: BLE001
        print("FAILED:", type(e).__name__, str(e)[:300])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", help="path to an ONNX model, e.g. tiny_qdq_conv.onnx or backbone.onnx")
    args = p.parse_args()
    probe(args.model)
    sys.exit(0)
