"""Static int8 PTQ of openpilot's driving / DM models with onnxsim's whole-graph QDQ quantizer.

Calibration samples come from real route frames (`prepare_inputs.py`), run through the fp32 model in
modeld order so the driving model's recurrent state inputs are realistic too; every `--stride`-th
frame's full input dict is kept. The quantized model is QDQ (uint8 activations, int8 per-channel
weights, int32 biases): exactly the integer contract an HVX `vrmpy` kernel implements.
"""

import argparse
import json
import time

import numpy as np
import onnx
import run_models

from onnxsim.full_qdq import quantize_full_qdq


def driving_samples(model, inputs, n, stride):
    sess = run_models.session(model)
    ins = {i.name: i for i in sess.get_inputs()}
    road = np.concatenate([np.load(f)["road"] for f in inputs.split(",")])
    state = {
        k: np.zeros(ins[k].shape, np.uint8 if "uint8" in ins[k].type else np.float32)
        for k in ins
        if k.startswith("state_")
    }
    fixed = {
        "desire": np.zeros(ins["desire"].shape, np.float32),
        "traffic_convention": np.array([[1, 0]], np.float32),
        "action_t": run_models.ACTION_T,
    }
    names = [o.name for o in sess.get_outputs()]
    out = []
    for i in range(min(n * stride, len(road))):
        feed = {"new_img": road[i], **state, **fixed}
        if i % stride == stride - 1:
            out.append({k: v.copy() for k, v in feed.items()})
        res = dict(zip(names, sess.run(None, feed)))
        for k in state:
            state[k] = res["next_" + k]
    return out


def dm_samples(inputs, n, stride, calib):
    drv = np.concatenate([np.load(f)["driver"] for f in inputs.split(",")])
    c = np.array([calib], np.float32)
    return [
        {"input_img": drv[i], "calib": c}
        for i in range(stride - 1, min(n * stride, len(drv)), stride)
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=["driving", "dm"])
    ap.add_argument("model", help="fp32 model (run_models.py to-fp32)")
    ap.add_argument(
        "inputs",
        help="calibration frames .npz, comma list (segments other than the eval one)",
    )
    ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--stride", type=int, default=9)
    ap.add_argument("--method", default="minmax")
    ap.add_argument("--activation-dtype", default="uint8")
    ap.add_argument("--exclude-op-types", default="", help="comma list kept in float")
    ap.add_argument(
        "--exclude-nodes-prefix",
        default="",
        help="comma list of node-name prefixes kept in float",
    )
    ap.add_argument(
        "--float-after-last-conv",
        action="store_true",
        help="keep every node after the last Conv (the Gemm/MatMul heads) in float",
    )
    ap.add_argument(
        "--policy",
        help="JSON {tensor_dtypes: {tensor: dtype}, exclude_nodes: [...]} "
        "(e.g. from sweep_driving_groups.py), merged with the flags above",
    )
    ap.add_argument("--calib", default="0,0.164,0.005")
    args = ap.parse_args()
    t = time.time()
    data = (
        driving_samples(args.model, args.inputs, args.samples, args.stride)
        if args.kind == "driving"
        else dm_samples(
            args.inputs,
            args.samples,
            args.stride,
            [float(v) for v in args.calib.split(",")],
        )
    )
    m = onnx.load(args.model)
    prefixes = [p for p in args.exclude_nodes_prefix.split(",") if p]
    excl_nodes = [
        n.name for n in m.graph.node if any(n.name.startswith(p) for p in prefixes)
    ]
    if args.float_after_last_conv:
        last = max(i for i, n in enumerate(m.graph.node) if n.op_type == "Conv")
        excl_nodes += [n.name or n.output[0] for n in m.graph.node[last + 1 :]]
    tensor_dtypes = {}
    if args.policy:
        pol = json.load(open(args.policy))
        tensor_dtypes = pol.get("tensor_dtypes", {})
        excl_nodes += pol.get("exclude_nodes", [])
    q = quantize_full_qdq(
        m,
        data,
        activation_dtype=args.activation_dtype,
        method=args.method,
        exclude_op_types=[o for o in args.exclude_op_types.split(",") if o],
        exclude_nodes=excl_nodes,
        tensor_dtypes=tensor_dtypes,
    )
    onnx.save(q, args.out)
    nq = sum(n.op_type == "QuantizeLinear" for n in q.graph.node)
    print(
        f"{args.out}: {len(data)} calibration samples, {nq} QuantizeLinear, {len(excl_nodes)} nodes kept float, {time.time() - t:.0f}s"
    )


if __name__ == "__main__":
    main()
