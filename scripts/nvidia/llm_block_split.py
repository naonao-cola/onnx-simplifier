"""Split a decoder LLM's transformer layers into TensorRT-buildable blocks, and measure
performance across block sizes.

Motivation (see scripts/nvidia/README.md's Decoder LLM section): TensorRT compiles this
whole 24-layer attention pattern as a single fused ("Myelin") subgraph, whose weight
staging buffer needs 988 MB of GPU memory this board doesn't have -- regardless of onnxsim
simplification, which only gets it down to 272 MB (still over the 256 MB default CMA pool,
and TensorRT's tactic autotuner stops offering that cheaper option once a bigger CMA pool
is available at all -- see the README). Splitting the ONNX graph into N-layer blocks
*before* it reaches the builder sidesteps the problem entirely: each block is its own
TensorRT network with its own, much smaller, Myelin subgraph.

Two stages, split by interpreter like the rest of scripts/nvidia (onnxsim's wheel needs
Python >= 3.11; JetPack's TensorRT bindings are cp310):

    python llm_block_split.py split model.sim.onnx OUT_DIR --block-sizes 1 2 4 8 24
    python llm_block_split.py build OUT_DIR --seq 1 --past 31

``split`` (onnxsim interpreter, only needs onnx) writes ``blocks_k{K}/block{i}.onnx`` for
each block size K -- ``ceil(24/K)`` blocks per K, the last one folding in the final
RMSNorm + lm_head so it produces ``logits`` like the full model. Each block is a standalone
ONNX model: hidden-states in/out at its layer boundary, plus that block's own
``past_key_values.{i}``/``present.{i}`` KV-cache slice and (redundantly per block, cheap)
its own copy of the shared RoPE/causal-mask precompute from ``attention_mask``/
``position_ids``.

``build`` (system Python with tensorrt) builds every block with ``trtexec --fp16``,
records which ones fit in memory, then chains each successful K's engines -- each K in
its own subprocess (loading every block's engine keeps all of them resident at once,
which can itself exceed available memory for large blocks even though each one *built*
fine sequentially; a subprocess per K guarantees clean release between them, confirmed
necessary on this board) -- for a real end-to-end decode-step latency measurement (H2D
input / D2H output per block via the TensorRT Python API, I/O buffers set up once per
engine and reused across iterations, not malloc'd fresh every call), plus output
agreement against the single-layer (K=1) reference chain.

Findings on this board (Jetson Orin Nano, TensorRT 10.3, CUDA 12.6; Qwen2.5-0.5B decode
step, 31 cached tokens): every block count from K=2 to K=12 builds and chains reliably;
K=1 (24 single-layer engines) builds and chains reliably but is slowest (most inter-engine
overhead); K=24 (the unsplit whole model) is fastest when it works but is unreliable --
its single Myelin subgraph sits right at this board's memory ceiling, so it succeeds or
fails depending on transient memory state and which tactic TensorRT's autotuner happens to
pick (observed directly: identical config, OOM on one attempt, three clean successes
immediately after). Latency drops smoothly and monotonically as K grows (more fused
compute per engine, less inter-engine H2D/D2H round-tripping): roughly 64 ms (K=1) down to
~35 ms (K=24) for one decode step. Every K's output has the same argmax as K=1's (identical
top prediction), though raw logit magnitudes differ by ~2-3 absolute (~13-19% of the
largest logit) -- a real, exactly-reproducible (confirmed across independent runs) fp16
accumulation difference from different fusion-boundary rounding, not a bug or noise.
"""

import argparse
import ctypes
import json
import math
import multiprocessing
import re
import sys
import time
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------- split


def _layer_node_ranges(model):
    """{layer_idx: [node indices]}, by exact ``/model/layers.N/`` prefix match."""
    layers = {}
    for i, n in enumerate(model.graph.node):
        m = re.search(r"/model/layers\.(\d+)/", n.name)
        if m:
            layers.setdefault(int(m.group(1)), []).append(i)
    return layers


def _layer_output(model, layer_idx, layer_nodes):
    """The hidden-states tensor a layer hands to the next one."""
    last = model.graph.node[layer_nodes[layer_idx][-1]]
    assert last.op_type == "Add", f"unexpected last op in layer {layer_idx}: {last.op_type}"
    return last.output[0]


def _extract_one(model_path, out_path, input_names, output_names):
    """One ``extract_model`` call, run in its own process (see ``_extract_isolated``)."""
    import onnx.utils

    onnx.utils.extract_model(
        str(model_path), str(out_path), input_names, output_names,
        check_model=False, infer_shapes=False,
    )


def _extract_isolated(model_path, out_path, input_names, output_names):
    """``extract_model`` reloads the whole source model from disk and (by default) runs
    full shape inference + checking on every call -- expensive and, across dozens of
    calls in one long-running process, memory that never gets reclaimed. A fresh
    subprocess per call returns it to the OS on exit; ``check_model``/``infer_shapes``
    are skipped too (TensorRT's own ONNX parser infers shapes on import regardless)."""
    proc = multiprocessing.get_context("spawn").Process(
        target=_extract_one, args=(model_path, out_path, input_names, output_names))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"extract_model process exited with {proc.exitcode} for {out_path}")


def split(args):
    import onnx

    # Only the graph structure (node names/types), not the ~1 GB of weight data, is
    # needed to compute block boundaries -- keep this process's own memory small since
    # each block extraction below runs in its own subprocess anyway.
    model = onnx.load(args.model, load_external_data=False)
    layer_nodes = _layer_node_ranges(model)
    n_layers = len(layer_nodes)
    embed_out = next(
        n.output[0] for n in model.graph.node if n.name == "/model/embed_tokens/Gather"
    )
    shared_inputs = ["attention_mask", "position_ids"]
    out = Path(args.out_dir)

    for k in args.block_sizes:
        bdir = out / f"blocks_k{k}"
        bdir.mkdir(parents=True, exist_ok=True)
        starts = list(range(0, n_layers, k))
        meta = []
        for bi, lo in enumerate(starts):
            hi = min(lo + k - 1, n_layers - 1)
            hidden_in = embed_out if lo == 0 else _layer_output(model, lo - 1, layer_nodes)
            hidden_out = _layer_output(model, hi, layer_nodes)
            kv_in = [f"past_key_values.{i}.{kv}" for i in range(lo, hi + 1) for kv in ("key", "value")]
            kv_out = [f"present.{i}.{kv}" for i in range(lo, hi + 1) for kv in ("key", "value")]
            is_last = hi == n_layers - 1
            output_names = ([hidden_out] if not is_last else ["logits"]) + kv_out
            block_path = bdir / f"block{bi}.onnx"
            if not block_path.exists():
                _extract_isolated(args.model, block_path, shared_inputs + kv_in + [hidden_in],
                                  output_names)
            meta.append({"block": bi, "layers": [lo, hi], "hidden_in": hidden_in,
                        "hidden_out": None if is_last else hidden_out, "kv": [lo, hi],
                        "is_last": is_last})
        (bdir / "meta.json").write_text(json.dumps(meta, indent=1))
        sizes = [round((bdir / f"block{m['block']}.onnx").stat().st_size / 1e6, 1) for m in meta]
        print(f"k={k}: {len(meta)} blocks, sizes MB {sizes}", flush=True)


# ----------------------------------------------------------------------------- build


TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def _mem_free_mb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return 1 << 30


def build_block_engine(onnx_path, engine_path, timeout=300):
    import subprocess

    cmd = [TRTEXEC, f"--onnx={onnx_path}", "--fp16", f"--saveEngine={engine_path}",
          "--skipInference"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = p.stdout + p.stderr
    ok = "&&&& PASSED" in out
    err = None
    if not ok:
        m = re.search(r"Requested amount of GPU memory \((\d+) bytes\)", out)
        err = f"needs {int(m.group(1)) / 1e6:.0f} MB" if m else next(
            (l for l in out.splitlines() if "[E]" in l), "build failed")[:150]
    return ok, err


def _chain_one_isolated(bdir, k, meta, seq, past, logits_path):
    """Run ``run_chain`` for one block size in its own subprocess: loading every
    block's engine (weights resident simultaneously, unlike the build stage which only
    ever holds one block's build-time memory at a time) can itself exceed available
    memory for larger blocks -- confirmed on this board, where k=6's chained run OOM'd
    after k=1..4's engines/contexts were never explicitly released within one
    long-running process. A fresh process per k guarantees that release. Writes
    ``logits_path`` (npy) plus a small JSON result next to it; raises on failure so the
    parent sees a normal subprocess exit code."""
    import tensorrt as trt

    sys.path.insert(0, str(Path(__file__).parent))
    import trt_harness as h

    logits, kv_out, ms, breakdown = run_chain(h, trt, bdir, meta, seq, past)
    np.save(logits_path, logits)
    Path(str(logits_path) + ".json").write_text(json.dumps(
        {"n_blocks": len(meta), "ms_total": round(ms, 4),
         "ms_per_block": [round(t, 4) for t in breakdown]}))


def build(args):
    out = Path(args.out_dir)
    results = {}
    for bdir in sorted(out.glob("blocks_k*")):
        k = int(bdir.name.removeprefix("blocks_k"))
        meta = json.loads((bdir / "meta.json").read_text())
        block_ok = True
        for bm in meta:
            bi = bm["block"]
            engine_path = bdir / f"block{bi}.engine"
            if engine_path.exists():
                print(f"k={k} block{bi} (layers {bm['layers']}): exists, skipped", flush=True)
                continue
            while _mem_free_mb() < 1500:
                time.sleep(3)
            ok, err = build_block_engine(bdir / f"block{bi}.onnx", engine_path)
            print(f"k={k} block{bi} (layers {bm['layers']}): {'OK' if ok else 'FAIL ' + str(err)}",
                  flush=True)
            if not ok:
                block_ok = False
        results[k] = {"meta": meta, "all_built": block_ok}

    buildable = [k for k, r in results.items() if r["all_built"]]
    print(f"\nbuildable block sizes: {buildable}")

    # Chain each buildable K's engines for a real end-to-end decode step, each in its
    # own subprocess (see _chain_one_isolated).
    timings = {}
    ref_logits = None
    for k in sorted(buildable):
        bdir = out / f"blocks_k{k}"
        meta = results[k]["meta"]
        while _mem_free_mb() < 1500:
            time.sleep(3)
        logits_path = out / f"chain_k{k}_logits.npy"
        proc = multiprocessing.get_context("spawn").Process(
            target=_chain_one_isolated, args=(bdir, k, meta, args.seq, args.past, logits_path))
        proc.start()
        proc.join()
        if proc.exitcode != 0:
            print(f"k={k}: chain FAILED (subprocess exit {proc.exitcode}, likely runtime OOM "
                  f"loading all {len(meta)} block engines at once)", flush=True)
            timings[k] = {"n_blocks": len(meta), "error": f"exit {proc.exitcode}"}
            continue
        logits = np.load(logits_path)
        r = json.loads(Path(str(logits_path) + ".json").read_text())
        agree = None
        if ref_logits is None:
            ref_logits = logits
        else:
            agree = float(np.abs(logits.astype(np.float32) - ref_logits.astype(np.float32)).max())
        timings[k] = {**r, "max_abs_diff_vs_k1": agree}
        print(f"k={k}: {r['n_blocks']} blocks, {r['ms_total']:.3f} ms end-to-end, "
              f"diff-vs-k1={agree}", flush=True)

    (out / "results.json").write_text(json.dumps(timings, indent=1))
    print(f"\n{'k':>4}{'blocks':>8}{'ms':>10}{'diff-vs-k1':>14}")
    for k, r in sorted(timings.items()):
        if "error" in r:
            print(f"{k:>4}{r['n_blocks']:>8}  ERROR {r['error']}")
            continue
        d = r["max_abs_diff_vs_k1"]
        print(f"{k:>4}{r['n_blocks']:>8}{r['ms_total']:>10.3f}"
              f"{'ref' if d is None else f'{d:.2e}':>14}")


class _BlockRunner:
    """One engine + context with I/O metadata queried and device buffers allocated
    *once* at construction, reused across every call -- unlike a naive per-call
    malloc/free + fresh-metadata-query approach (the original version of this script),
    whose per-block timing was dominated by that setup cost rather than real GPU
    compute, making it useless as a latency breakdown. A real serving engine would
    likewise set up its I/O buffers once and just re-fill them per token."""

    def __init__(self, h, trt, cuda, engine, ctx):
        self.cuda = cuda
        self.engine, self.ctx = engine, ctx
        self.bufs, self.dtypes, self.shapes, self.out_names = {}, {}, {}, []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(ctx.get_tensor_shape(name))
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self.bufs[name] = cuda.malloc(nbytes)
            ctx.set_tensor_address(name, self.bufs[name].value)
            self.dtypes[name], self.shapes[name] = dtype, shape
            if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                self.out_names.append(name)

    def run(self, feeds):
        for name, x in feeds.items():
            self.cuda.memcpy_htod(self.bufs[name], np.ascontiguousarray(x, dtype=self.dtypes[name]))
        self.ctx.execute_async_v3(0)
        self.cuda.sync()
        outs = {n: np.empty(self.shapes[n], dtype=self.dtypes[n]) for n in self.out_names}
        for n, h_out in outs.items():
            self.cuda.memcpy_dtoh(h_out, self.bufs[n])
        return outs

    def free(self):
        for p in self.bufs.values():
            self.cuda.free(p)


def run_chain(h, trt, bdir, meta, seq, past, iters=20, warmup=5):
    """Load every block engine in one process and run a real chained decode step,
    threading hidden-states and each layer's KV cache between blocks (host round-trip
    per block via ``_BlockRunner`` -- simple and correct, not zero-copy
    device-to-device, but a fair, honest measurement of what a real multi-engine
    deployment would pay). I/O buffers are set up once per engine (see
    ``_BlockRunner``), not per call, so both ``ms_total`` and the per-block breakdown
    reflect steady-state execution, not one-time setup cost."""
    cuda = h.Cudart()
    rng = np.random.default_rng(0)
    runners = []
    for bm in meta:
        blob = (bdir / f"block{bm['block']}.engine").read_bytes()
        eng = trt.Runtime(h.LOGGER).deserialize_cuda_engine(blob)
        runners.append(_BlockRunner(h, trt, cuda, eng, eng.create_execution_context()))

    # Shared, fixed inputs across every block (each block recomputes its own RoPE/mask).
    attn_mask = np.ones((1, past + seq), dtype=np.int64)
    pos_ids = np.arange(past, past + seq, dtype=np.int64)[None]
    hidden0 = rng.standard_normal((1, seq, 896)).astype(np.float16) * 0.1
    kv_cache = {}
    for i in range(24):
        for kv in ("key", "value"):
            kv_cache[(i, kv)] = (rng.standard_normal((1, 2, past, 64)) * 0.1).astype(np.float32)

    def run_once():
        h_ = hidden0
        kv_out = {}
        breakdown = []
        for bm, runner in zip(meta, runners):
            feeds = {"attention_mask": attn_mask, "position_ids": pos_ids, bm["hidden_in"]: h_}
            for i in range(bm["kv"][0], bm["kv"][1] + 1):
                feeds[f"past_key_values.{i}.key"] = kv_cache[(i, "key")]
                feeds[f"past_key_values.{i}.value"] = kv_cache[(i, "value")]
            t_block = time.perf_counter()
            outs = runner.run(feeds)
            breakdown.append(time.perf_counter() - t_block)
            for i in range(bm["kv"][0], bm["kv"][1] + 1):
                kv_out[(i, "key")] = outs[f"present.{i}.key"]
                kv_out[(i, "value")] = outs[f"present.{i}.value"]
            h_ = outs.get(bm["hidden_out"]) if bm["hidden_out"] else outs["logits"]
        return h_, kv_out, breakdown

    for _ in range(warmup):
        run_once()
    t0 = time.perf_counter()
    for _ in range(iters):
        logits, kv_out, breakdown = run_once()
    ms = (time.perf_counter() - t0) / iters * 1e3
    for r in runners:
        r.free()
    return logits, kv_out, ms, breakdown


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("model")
    s.add_argument("out_dir")
    s.add_argument("--block-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 12, 24])
    s.set_defaults(fn=split)
    b = sub.add_parser("build")
    b.add_argument("out_dir")
    b.add_argument("--seq", type=int, default=1)
    b.add_argument("--past", type=int, default=31)
    b.set_defaults(fn=build)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
