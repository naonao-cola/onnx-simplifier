#!/usr/bin/env python3
"""Measure the peak-memory impact of PR #482 ("Defer loading external tensor
data to reduce memory usage").

PR #482 changed only *Python* code: it stops loading a model's (potentially
multi-GB) external tensor data up front and instead defers that load until the
model is about to be serialized for the C++ simplifier. Every graph-metadata
phase in between runs with the weights left on disk. The C++ simplifier itself
is unchanged by the PR.

This harness quantifies the effect by running the *same* workload against two
checkouts of the pure-Python modules -- the commit just before the PR and the
merge commit -- while sharing one compiled ``onnxsim_cpp2py_export`` extension
(so the identical C++ stage cancels out of any delta).

Two scenarios are measured:

* ``api``  -- ``onnxsim.simplify(path)``           (library entry point)
* ``cli``  -- ``onnxsim.main()`` on ``in -> out``  (command-line tool)

For each we report:

* ``after_load``  RSS right after the first ``onnx.load`` returns -- the
  working set the model occupies throughout the (potentially long) Python-side
  graph transformations. This is what the deferral shrinks.
* ``peak``        process peak RSS (``ru_maxrss``) over the whole run.

Usage::

    python bench/peak_memory.py setup        # build old/new pkgs + gen model
    python bench/peak_memory.py run  <api|cli> <old|new>   # one measurement
    python bench/peak_memory.py report        # run all four and print a table

Environment: needs ``onnx``, ``onnxruntime``, ``psutil`` and a compiled
``onnxsim_cpp2py_export*.so`` (e.g. from ``pip download onnxsim``). Set
``ONNXSIM_SO`` to point at that .so and ``ONNXSIM_REPO`` at this git checkout.
"""
import json
import os
import resource
import subprocess
import sys
import threading
import time

BENCH = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("ONNXSIM_BENCH_WORK", "/tmp/onnxsim_peak_bench")
REPO = os.environ.get("ONNXSIM_REPO", os.path.dirname(BENCH))
SO = os.environ.get("ONNXSIM_SO", "")   # path to onnxsim_cpp2py_export*.so

# Commit just before PR #482, and the PR merge commit.
OLD_SHA = os.environ.get("ONNXSIM_OLD_SHA", "b625e26")
NEW_SHA = os.environ.get("ONNXSIM_NEW_SHA", "7e500fc")

MODEL_DIR = os.path.join(WORK, "model")
MODEL = os.path.join(MODEL_DIR, "model.onnx")


# --------------------------------------------------------------------------- #
# setup: build the two package variants and a large external-data model
# --------------------------------------------------------------------------- #
def _build_variant(name, sha):
    dst = os.path.join(WORK, f"{name}_pkg")
    subprocess.run(f"rm -rf {dst} && mkdir -p {dst}", shell=True, check=True)
    subprocess.run(
        f"git -C {REPO} archive {sha} onnxsim/ | tar -x -C {dst}",
        shell=True, check=True)
    if not SO or not os.path.exists(SO):
        raise SystemExit(
            "Set ONNXSIM_SO to a compiled onnxsim_cpp2py_export*.so "
            "(e.g. `pip download onnxsim` and unzip the wheel).")
    subprocess.run(f"cp {SO} {dst}/onnxsim/", shell=True, check=True)
    # version.py is generated at build time and absent from the source tree.
    vpy = os.path.join(dst, "onnxsim", "version.py")
    if not os.path.exists(vpy):
        open(vpy, "w").write("version = '0.0.0+bench'\n")
    print(f"built {name} ({sha}) -> {dst}")


def _gen_model(n=4096, layers=16):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    os.makedirs(MODEL_DIR, exist_ok=True)
    inits, nodes, prev, total = [], [], "x", 0
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, n])
    for i in range(layers):
        w = ((np.random.rand(n, n).astype(np.float32) - 0.5) * 0.01)
        total += w.nbytes
        inits.append(numpy_helper.from_array(w, f"W{i}"))
        nodes.append(helper.make_node("MatMul", [prev, f"W{i}"], [f"h{i}"]))
        prev = f"h{i}"
    out = helper.make_tensor_value_info(prev, TensorProto.FLOAT, [1, n])
    graph = helper.make_graph(nodes, "matmul_chain", [inp], [out], initializer=inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=10)
    onnx.save(model, MODEL, save_as_external_data=True,
              all_tensors_to_one_file=True, location="model.data")
    print(f"model weights ~= {total/2**20:.0f} MiB (all external), {MODEL}")


def setup():
    _build_variant("old", OLD_SHA)
    _build_variant("new", NEW_SHA)
    _gen_model()


# --------------------------------------------------------------------------- #
# run: one measurement in this (fresh) process
# --------------------------------------------------------------------------- #
def _rss_mib(proc):
    return proc.memory_info().rss / 2**20


def run(scenario, variant):
    import psutil
    pkg = os.path.join(WORK, f"{variant}_pkg")
    sys.path.insert(0, pkg)
    sys.path.insert(0, os.path.join(pkg, "onnxsim"))

    proc = psutil.Process()
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            samples.append(_rss_mib(proc))
            time.sleep(0.002)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    import onnx
    import onnxsim
    import onnxsim.onnx_simplifier as osm

    ck = {}
    real_load = onnx.load_model

    def load_wrap(*a, **k):
        m = real_load(*a, **k)
        ck.setdefault("after_load_MiB", round(_rss_mib(proc), 1))
        ck["load_external_data"] = k.get("load_external_data", "default(True)")
        return m
    onnx.load_model = onnx.load = osm.onnx.load = load_wrap

    # Share the compiled extension across variants: the released .so predates
    # the model-executor argument, so drop it. The C++ work is identical for
    # old and new, so it cancels out of the delta.
    real_c = osm.C.simplify

    def c_shim(*args):
        ck["at_cpp_entry_MiB"] = round(_rss_mib(proc), 1)
        if len(args) == 6:
            args = args[1:]
        return real_c(*args)

    class _C:
        pass
    shim = _C()
    for a in dir(osm.C):
        if not a.startswith("__"):
            setattr(shim, a, getattr(osm.C, a))
    shim.simplify = c_shim
    osm.C = shim

    t0 = time.time()
    if scenario == "api":
        onnxsim.simplify(MODEL, check_n=0)
    elif scenario == "cli":
        out = os.path.join(WORK, f"out_{variant}.onnx")
        sys.argv = ["onnxsim", MODEL, out]
        try:
            onnxsim.main()
        except SystemExit:
            pass
    else:
        raise SystemExit(f"unknown scenario {scenario!r}")
    dt = time.time() - t0
    stop.set()
    th.join()

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MiB
    print("RESULT " + json.dumps({
        "scenario": scenario, "variant": variant,
        "time_s": round(dt, 1),
        "after_load_MiB": ck.get("after_load_MiB"),
        "load_external_data": str(ck.get("load_external_data")),
        "at_cpp_entry_MiB": ck.get("at_cpp_entry_MiB"),
        "peak_maxrss_MiB": round(peak, 1),
    }))


# --------------------------------------------------------------------------- #
# report: run all four measurements (each in a clean subprocess) and tabulate
# --------------------------------------------------------------------------- #
def report():
    rows = {}
    for scenario in ("api", "cli"):
        for variant in ("old", "new"):
            p = subprocess.run(
                [sys.executable, __file__, "run", scenario, variant],
                capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
            if not line:
                print(p.stdout, p.stderr)
                raise SystemExit(f"{scenario}/{variant} produced no RESULT")
            rows[(scenario, variant)] = json.loads(line[0][7:])

    def fmt(scenario):
        o, n = rows[(scenario, "old")], rows[(scenario, "new")]
        d_peak = o["peak_maxrss_MiB"] - n["peak_maxrss_MiB"]
        d_load = o["after_load_MiB"] - n["after_load_MiB"]
        print(f"\n[{scenario}]")
        print(f"  after_load  old={o['after_load_MiB']:>8} MiB  "
              f"new={n['after_load_MiB']:>8} MiB  reduced {d_load:.0f} MiB")
        print(f"  peak_maxrss old={o['peak_maxrss_MiB']:>8} MiB  "
              f"new={n['peak_maxrss_MiB']:>8} MiB  reduced {d_peak:.0f} MiB "
              f"({100*d_peak/o['peak_maxrss_MiB']:.1f}%)")
    for s in ("api", "cli"):
        fmt(s)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "setup":
        setup()
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3])
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)
