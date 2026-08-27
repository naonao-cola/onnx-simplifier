#!/usr/bin/env python3
"""Synthetic repro + peak-memory harness for the OOM documented in
bench/TODO_large_decoder_submodule_oom.md ("onnxsim OOMs simplifying a ~5GB
decoder-only transformer submodule").

Builds a decoder-block-shaped ONNX model (repeated self-attention + SwiGLU-MLP
blocks, external-data weights) at any target size, with no torch/transformers
dependency -- tensors are written directly via numpy_helper-free file I/O, so
generating even a multi-GB model costs only O(one tensor) of Python memory
(see the "gen" command's docstring below for why that matters). Then measures
onnxsim.simplify()'s peak RSS on it, in a subprocess so a real OOM-kill doesn't
take down the harness.

See bench/RESULTS_synthetic_decoder_oom.md for what this found.

Usage:
    python bench/decoder_oom_repro.py gen <out_dir> [--layers N] [--hidden H]
        [--ffn F] [--layout many|single]
    python bench/decoder_oom_repro.py measure <model.onnx> [--check-n N]
    python bench/decoder_oom_repro.py matrix <work_dir> [--sizes 5,8] [--keep]
"""

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
from onnx import TensorProto, helper

BENCH = os.path.dirname(os.path.abspath(__file__))

# hidden=2048, ffn=5632 gives ~51.4M params/layer (4*hidden^2 attn +
# 3*hidden*ffn mlp) -- close to a real ~1-2B decoder's per-layer size, so
# --layers alone controls total model size at a realistic per-tensor shape.
DEFAULT_HIDDEN = 2048
DEFAULT_FFN = 5632
PARAMS_PER_LAYER = 4 * DEFAULT_HIDDEN**2 + 3 * DEFAULT_HIDDEN * DEFAULT_FFN


def layers_for_gb(gb):
    return max(1, round(gb * 1e9 / 4 / PARAMS_PER_LAYER))


# --------------------------------------------------------------------------- #
# gen: build the synthetic model
# --------------------------------------------------------------------------- #
def _set_external(tensor, location, offset, length):
    tensor.data_location = TensorProto.EXTERNAL
    tensor.ClearField("external_data")
    for k, v in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = tensor.external_data.add()
        entry.key = k
        entry.value = v


class _ExternalWriter:
    """Writes each tensor's bytes to disk immediately and returns a
    header-only TensorProto (no raw_data) pointing at the write.

    This keeps peak Python memory during generation O(one tensor), not
    O(total weights). An earlier version of this script instead built the
    whole model in memory and called onnx.save_model(...,
    save_as_external_data=True) to let onnx do the raw_data -> file
    conversion; at ~5GB of weights that *generation step itself* was
    OOM-killed by this repo's sandbox (11.4GB RSS for 4.9GB of tensor data --
    2.3x -- via a memcg cap of ~13.3GiB, close to the ~15GB cap the original
    report used). That's a real, separate finding -- building a multi-GB
    ONNX model through onnx's standard Python helper API costs 2x+ the raw
    tensor bytes in peak RSS, relevant to any exporter that goes through it,
    torch.onnx's legacy exporter included -- but it's a confound for what
    this script actually measures (onnxsim's own peak memory on an
    already-exported model), so generation here is kept cheap.
    """

    def __init__(self, out_dir, layout, seed):
        self.out_dir = out_dir
        self.layout = layout
        self.rng = np.random.default_rng(seed)
        self.n_initializers = 0
        self.total_bytes = 0
        if layout == "single":
            self.single_path = os.path.join(out_dir, "model.data")
            self.single_f = open(self.single_path, "wb")
            self.single_offset = 0

    def weight(self, name, shape):
        arr = self.rng.standard_normal(shape, dtype=np.float32) * np.float32(0.02)
        nbytes = arr.nbytes
        tp = TensorProto()
        tp.name = name
        tp.data_type = TensorProto.FLOAT
        tp.dims.extend(shape)

        if self.layout == "many":
            fname = name.replace("/", "_") + ".bin"
            with open(os.path.join(self.out_dir, fname), "wb") as f:
                arr.tofile(f)
            _set_external(tp, fname, 0, nbytes)
        else:
            arr.tofile(self.single_f)
            _set_external(tp, "model.data", self.single_offset, nbytes)
            self.single_offset += nbytes

        self.n_initializers += 1
        self.total_bytes += nbytes
        del arr
        return tp

    def close(self):
        if self.layout == "single":
            self.single_f.close()


def _decoder_layer(x_name, layer_idx, hidden, ffn, writer, initializers):
    prefix = f"layer{layer_idx}"
    nodes = []

    def W(name, shape):
        tp = writer.weight(name, shape)
        initializers.append(tp)
        return name

    attn_in = x_name
    q_w = W(f"{prefix}.q_proj.weight", (hidden, hidden))
    k_w = W(f"{prefix}.k_proj.weight", (hidden, hidden))
    v_w = W(f"{prefix}.v_proj.weight", (hidden, hidden))
    o_w = W(f"{prefix}.o_proj.weight", (hidden, hidden))

    q, k, v = f"{prefix}.q", f"{prefix}.k", f"{prefix}.v"
    nodes.append(helper.make_node("MatMul", [attn_in, q_w], [q]))
    nodes.append(helper.make_node("MatMul", [attn_in, k_w], [k]))
    nodes.append(helper.make_node("MatMul", [attn_in, v_w], [v]))

    kt = f"{prefix}.kt"
    nodes.append(helper.make_node("Transpose", [k], [kt], perm=[0, 2, 1]))
    qk = f"{prefix}.qk"
    nodes.append(helper.make_node("MatMul", [q, kt], [qk + ".pre"]))
    nodes.append(helper.make_node("Softmax", [qk + ".pre"], [qk], axis=-1))
    attn_out_pre = f"{prefix}.attn_out_pre"
    nodes.append(helper.make_node("MatMul", [qk, v], [attn_out_pre]))
    attn_out = f"{prefix}.attn_out"
    nodes.append(helper.make_node("MatMul", [attn_out_pre, o_w], [attn_out]))

    resid1 = f"{prefix}.resid1"
    nodes.append(helper.make_node("Add", [attn_in, attn_out], [resid1]))

    gate_w = W(f"{prefix}.mlp.gate_proj.weight", (hidden, ffn))
    up_w = W(f"{prefix}.mlp.up_proj.weight", (hidden, ffn))
    down_w = W(f"{prefix}.mlp.down_proj.weight", (ffn, hidden))

    gate, up = f"{prefix}.mlp.gate", f"{prefix}.mlp.up"
    nodes.append(helper.make_node("MatMul", [resid1, gate_w], [gate]))
    nodes.append(helper.make_node("MatMul", [resid1, up_w], [up]))
    gate_act = f"{prefix}.mlp.gate_act"
    nodes.append(helper.make_node("Sigmoid", [gate], [gate_act]))
    gate_silu = f"{prefix}.mlp.gate_silu"
    nodes.append(helper.make_node("Mul", [gate, gate_act], [gate_silu]))
    mlp_hidden = f"{prefix}.mlp.hidden"
    nodes.append(helper.make_node("Mul", [gate_silu, up], [mlp_hidden]))
    mlp_out = f"{prefix}.mlp.out"
    nodes.append(helper.make_node("MatMul", [mlp_hidden, down_w], [mlp_out]))

    resid2 = f"{prefix}.resid2"
    nodes.append(helper.make_node("Add", [resid1, mlp_out], [resid2]))

    return nodes, resid2


def gen(out_dir, layers, hidden, ffn, seq_len, layout, seed):
    os.makedirs(out_dir, exist_ok=True)
    writer = _ExternalWriter(out_dir, layout, seed)

    initializers = []
    nodes = []
    x = "inputs_embeds"
    cur = x
    for i in range(layers):
        layer_nodes, cur = _decoder_layer(cur, i, hidden, ffn, writer, initializers)
        nodes.extend(layer_nodes)
    writer.close()

    graph = helper.make_graph(
        nodes,
        "synthetic_decoder",
        [helper.make_tensor_value_info(x, TensorProto.FLOAT, [1, seq_len, hidden])],
        [helper.make_tensor_value_info(cur, TensorProto.FLOAT, [1, seq_len, hidden])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx_path = os.path.join(out_dir, "model.onnx")
    with open(onnx_path, "wb") as f:
        f.write(model.SerializeToString())

    n_files = sum(1 for f in os.listdir(out_dir) if f != "model.onnx")
    print(
        f"{onnx_path}: {writer.n_initializers} initializers, "
        f"{writer.total_bytes / 1e9:.2f} GB, {n_files} external-data file(s), "
        f"layout={layout}"
    )
    return onnx_path, writer.total_bytes


# --------------------------------------------------------------------------- #
# measure: run simplify() on a model in a subprocess, report peak RSS
# --------------------------------------------------------------------------- #
_CHILD_SRC = """
import resource, sys
model, check_n = sys.argv[1], int(sys.argv[2])
import onnx
import onnxsim
model_opt, ok = onnxsim.simplify(model, check_n=check_n)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print("CHILD_RESULT " + repr({"ok": ok, "self_peak_mib": peak}))
"""


def measure(model_path, check_n):
    # NOTE: resource.getrusage(RUSAGE_CHILDREN).ru_maxrss is a monotonic
    # high-water mark across *every* child this process has ever reaped, not
    # a per-call value -- so this function is only accurate as the *first and
    # only* subprocess.run() a given Python process performs. Calling it
    # more than once from the same long-lived process (as an earlier version
    # of `matrix` below did, in-process) makes every measurement after the
    # first report max(this run's true peak, every earlier run's peak) --
    # silently overstating any run that happens to peak lower than a
    # previous one in the same process. `matrix` avoids this by invoking
    # `measure` as a fresh top-level process per row (see run_measure_subprocess).
    import tempfile

    fd, child_script = tempfile.mkstemp(suffix=".py", prefix="_onnxsim_oom_child_")
    with os.fdopen(fd, "w") as f:
        f.write(_CHILD_SRC)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, child_script, model_path, str(check_n)],
            capture_output=True,
            text=True,
        )
    finally:
        os.remove(child_script)
    dt = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    result = {
        "model": model_path,
        "check_n": check_n,
        "exit_code": proc.returncode,
        "killed_by_signal": proc.returncode < 0,
        "time_s": round(dt, 1),
        "peak_rss_child_mib": round(peak, 1),
        "stdout_tail": proc.stdout[-500:],
        "stderr_tail": proc.stderr[-500:],
    }
    print("RESULT " + json.dumps(result))
    return result


def run_measure_subprocess(model_path, check_n):
    """Run `measure` as a brand-new top-level process (see the accuracy note
    in `measure` above for why `matrix` must not call `measure` in-process)."""
    proc = subprocess.run(
        [sys.executable, __file__, "measure", model_path, "--check-n", str(check_n)],
        capture_output=True,
        text=True,
    )
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise RuntimeError(
        f"measure subprocess produced no RESULT line (exit {proc.returncode})"
    )


# --------------------------------------------------------------------------- #
# matrix: gen + measure across layouts/sizes/check_n, report a table
# --------------------------------------------------------------------------- #
def matrix(work_dir, sizes_gb, keep):
    os.makedirs(work_dir, exist_ok=True)
    rows = []
    for gb in sizes_gb:
        layers = layers_for_gb(gb)
        for layout in ("many", "single"):
            d = os.path.join(work_dir, f"{layout}_{gb}gb")
            shutil.rmtree(d, ignore_errors=True)
            model_path, total_bytes = gen(
                d, layers, DEFAULT_HIDDEN, DEFAULT_FFN, 8, layout, seed=0
            )
            for check_n in (0, 1) if layout == "many" else (0,):
                r = run_measure_subprocess(model_path, check_n)
                r["layout"] = layout
                r["total_gb"] = round(total_bytes / 1e9, 2)
                r["ratio"] = round(r["peak_rss_child_mib"] / 1024 / r["total_gb"], 2)
                rows.append(r)
            if not keep:
                shutil.rmtree(d, ignore_errors=True)

    print(
        "\n%-8s %-7s %-8s %-9s %-6s %-8s %s"
        % ("layout", "check_n", "size_GB", "peak_GiB", "ratio", "exit", "time_s")
    )
    for r in rows:
        print(
            "%-8s %-7d %-8.2f %-9.2f %-6.2f %-8d %.1f"
            % (
                r["layout"],
                r["check_n"],
                r["total_gb"],
                r["peak_rss_child_mib"] / 1024,
                r["ratio"],
                r["exit_code"],
                r["time_s"],
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("out_dir")
    g.add_argument("--layers", type=int, default=24)
    g.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    g.add_argument("--ffn", type=int, default=DEFAULT_FFN)
    g.add_argument("--seq-len", type=int, default=8)
    g.add_argument("--layout", choices=["many", "single"], default="many")
    g.add_argument("--seed", type=int, default=0)

    m = sub.add_parser("measure")
    m.add_argument("model")
    m.add_argument("--check-n", type=int, default=0)

    x = sub.add_parser("matrix")
    x.add_argument("work_dir")
    x.add_argument("--sizes", default="5,8", help="comma-separated model sizes in GB")
    x.add_argument(
        "--keep", action="store_true", help="keep generated models afterwards"
    )

    args = ap.parse_args()
    if args.cmd == "gen":
        gen(
            args.out_dir,
            args.layers,
            args.hidden,
            args.ffn,
            args.seq_len,
            args.layout,
            args.seed,
        )
    elif args.cmd == "measure":
        measure(args.model, args.check_n)
    elif args.cmd == "matrix":
        sizes = [float(s) for s in args.sizes.split(",")]
        matrix(args.work_dir, sizes, args.keep)


if __name__ == "__main__":
    main()
