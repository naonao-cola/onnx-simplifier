"""Check onnxsim's 2:4 sparsity path against the real TensorRT builder.

Claim under test (``onnxsim/tensorrt_sparsity.py``): TensorRT's N:M structured-sparse math
only ever applies to ``Gemm`` nodes, not ``MatMul``, so a 2:4-pruned weight (from
``apply_magnitude_pruning(n=2, m=4)``) needs ``convert_matmul_to_gemm`` before TensorRT will
use Sparse Tensor Cores on it.

SCOPE NOTE: that docstring's claim is specifically about **ONNX Runtime's TensorRT
execution provider** (``ORT_TENSORRT_SPARSITY_ENABLE=1``), citing
https://github.com/NVIDIA/TensorRT/issues/2271. This script instead builds directly with
TensorRT's own builder (``trtexec`` / the Python ``Builder`` API, same as ``trt_harness.py``)
-- not through ORT's TensorRT EP, which this board cannot run at all (no ``onnxruntime-gpu``
build here offers ``TensorrtExecutionProvider``; see ``run_cuda_feature_notebook.py``'s
findings for why). ORT's TRT EP ultimately hands the same network to TensorRT's builder, so
this is informative but not a direct test of the literal claim -- a real behavior difference
in ORT's own EP-construction code path (independent of what the builder itself now supports)
would not show up here.

Two stages (JetPack's TensorRT bindings are cp310-only, onnxsim needs Python >= 3.11, so
models are exchanged as files, like ``qdq_pairs.py`` / ``trt_harness.py``):

    # 1. onnxsim interpreter -- run from OUTSIDE the repo checkout (the source tree has no
    #    compiled extension), e.g. ``cd /tmp``:
    python sparsity_check.py gen OUT_DIR [--layers 6]
    # 2. system Python with tensorrt + onnx + numpy; needs /usr/src/tensorrt/bin/trtexec:
    python sparsity_check.py trt OUT_DIR [--runs 3]

``gen`` writes ``<shape>.<form>.onnx`` for shape in {2d197, 3d197, 2d2048} (activation
[197,768], [1,197,768], [2048,768]) and form in {dense_matmul, dense_gemm, pruned_matmul,
pruned_gemm}: a stack of ``--layers`` ViT-B MLPs (768->3072 ReLU 3072->768), where ``dense_*``
share random weights, ``pruned_*`` are the same weights after ``apply_magnitude_pruning(n=2,
m=4)`` and ``*_gemm`` are ``convert_matmul_to_gemm`` of the ``*_matmul`` model. It also saves
a fixed input and the fp32 onnxruntime output of the dense and of the pruned MatMul model
(``<shape>.ref_dense.npy`` / ``ref_pruned.npy``) for the numeric check.

``trt`` builds each model with ``trtexec --fp16 --sparsity={disable,enable,force}`` and
reports (a) engine layers whose tactic name is a sparse kernel (``sm80_xmma_sparse_gemm…``),
(b) median-of-``--runs`` mean GPU time (timing is round-robin across all engines to cancel drift), (c) TensorRT's own "eligible / chose sparse tactics"
verbose-log counts, and (d) the max error of the
fp16 sparse engine (python API, ``BuilderFlag.SPARSE_WEIGHTS``) against the fp32 reference.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
SHAPES = {"2d197": (197, 768), "3d197": (1, 197, 768), "2d2048": (2048, 768)}
FORMS = ["dense_matmul", "dense_gemm", "pruned_matmul", "pruned_gemm"]
SPARSE_TACTIC = re.compile(r"spars|spmma|sptensor|sp_?gemm", re.I)


# ----------------------------------------------------------------------------- stage 1


def _mlp_stack(shape, layers, rng):
    import onnx
    import onnx.numpy_helper as nh
    from onnx import parser

    lines, inits, cur = [], [], "X"
    for i in range(layers):
        w1 = (rng.standard_normal((768, 3072)) * np.sqrt(2 / 768)).astype(np.float32)
        w2 = (rng.standard_normal((3072, 768)) * np.sqrt(2 / 3072)).astype(np.float32)
        inits += [nh.from_array(w1, f"W1_{i}"), nh.from_array(w2, f"W2_{i}")]
        lines += [f"h{i} = MatMul({cur}, W1_{i})", f"a{i} = Relu(h{i})",
                  f"o{i} = MatMul(a{i}, W2_{i})"]
        cur = f"o{i}"
    lines.append(f"Y = Identity({cur})")
    dims = ",".join(map(str, shape))
    model = parser.parse_model(
        f'<ir_version: 8, opset_import: ["": 17]> g (float[{dims}] X) => (float[{dims}] Y) '
        "{ " + "\n".join(lines) + " }")
    model.graph.initializer.extend(inits)
    onnx.checker.check_model(model)
    return model


def _drop_unused_initializers(model):
    """``apply_magnitude_pruning`` re-emits each pruned weight under a new name (``_v_N``) and
    leaves the original dense initializer unreferenced; drop those dead copies."""
    used = {i for n in model.graph.node for i in n.input}
    keep = [t for t in model.graph.initializer if t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return model


def _check_24_along_k(model):
    """Every consecutive group of 4 along K (axis 0 of a [K,N] MatMul weight / axis 1 of a
    [N,K] transB Gemm weight) of every MatMul/Gemm weight must hold >= 2 zeros.
    Returns (ok, zero_fraction)."""
    import onnx.numpy_helper as nh

    inits = {t.name: t for t in model.graph.initializer}
    ok, zeros, total = True, 0, 0
    for n in model.graph.node:
        if n.op_type not in ("MatMul", "Gemm") or n.input[1] not in inits:
            continue
        w = nh.to_array(inits[n.input[1]])
        trans_b = any(a.name == "transB" and a.i for a in n.attribute)
        wk = w.T if trans_b else w  # -> [K, N]
        g = (wk == 0).reshape(wk.shape[0] // 4, 4, wk.shape[1]).sum(1)
        ok &= bool((g >= 2).all())
        zeros += int((wk == 0).sum())
        total += wk.size
    return ok, zeros / max(total, 1)


def gen(args):
    import onnx
    import onnxruntime as ort
    import onnxsim
    from onnxsim.tensorrt_sparsity import convert_matmul_to_gemm

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for sname, shape in SHAPES.items():
        rng = np.random.default_rng(0)
        dense = _mlp_stack(shape, args.layers, rng)
        pruned = _drop_unused_initializers(onnxsim.apply_magnitude_pruning(dense, n=2, m=4))
        models = {"dense_matmul": dense, "dense_gemm": convert_matmul_to_gemm(dense),
                  "pruned_matmul": pruned, "pruned_gemm": convert_matmul_to_gemm(pruned)}
        for form, m in models.items():
            onnx.save(m, out / f"{sname}.{form}.onnx")
            ok, zf = _check_24_along_k(m)
            meta[f"{sname}.{form}"] = {
                "gemm": sum(n.op_type == "Gemm" for n in m.graph.node),
                "matmul": sum(n.op_type == "MatMul" for n in m.graph.node),
                "nodes": len(m.graph.node), "zero_frac": round(zf, 3), "is_2to4_along_K": ok}
        x = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
        np.save(out / f"{sname}.x.npy", x)
        for kind in ("dense", "pruned"):  # each form is checked against its own fp32 result
            r = ort.InferenceSession(str(out / f"{sname}.{kind}_matmul.onnx"),
                                     providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
            np.save(out / f"{sname}.ref_{kind}.npy", r)
        ref = np.load(out / f"{sname}.ref_pruned.npy")
        # the Gemm rewrite must be value preserving
        g = ort.InferenceSession(str(out / f"{sname}.pruned_gemm.onnx"),
                                 providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
        meta[f"{sname}.pruned_gemm"]["max_abs_diff_vs_pruned_matmul_fp32"] = float(np.abs(g - ref).max())
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


# ----------------------------------------------------------------------------- stage 2


OOM = re.compile(r"out of memory|allocation failed|could not be allocated|NvMap|OutOfMemory", re.I)


def _trtexec(args, timeout=1800):
    p = subprocess.run([TRTEXEC, *args], capture_output=True, text=True, timeout=timeout)
    return p.stdout + p.stderr


def _mean_ms(text):
    m = re.search(r"GPU Compute Time: min = [\d.]+ ms, max = [\d.]+ ms, mean = ([\d.]+) ms", text)
    return float(m.group(1)) if m else None


def _mem_available_mb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return 1 << 30


def _load_engine(path):
    import tensorrt as trt
    sys.path.insert(0, str(Path(__file__).parent))
    import trt_harness as h

    return h, trt.Runtime(h.LOGGER).deserialize_cuda_engine(path.read_bytes())


def build(onnx_path, mode, out_dir, min_free_mb=2500, attempts=3):
    """Build one engine with ``trtexec --fp16 --sparsity=<mode>``. Jetson's 8 GB is shared
    CPU/GPU memory: when another process is hogging it TensorRT logs an allocation failure,
    *skips* tactics and still "PASSES" with a degraded engine, so a log containing such a
    line is discarded and the build repeated once memory is free again."""
    import time

    engine = out_dir / f"{onnx_path.stem}.{mode}.engine"
    res = {"mode": mode}
    for attempt in range(attempts):
        while _mem_available_mb() < min_free_mb:
            time.sleep(5)
        engine.unlink(missing_ok=True)
        log = _trtexec([f"--onnx={onnx_path}", "--fp16", f"--sparsity={mode}", "--verbose",
                        f"--saveEngine={engine}", "--profilingVerbosity=detailed", "--skipInference"])
        if not OOM.search(log):
            break
        res["oom_retries"] = attempt + 1
    else:
        res["error"] = "CUDA out of memory on every attempt"
        return res
    if not engine.exists():
        res["error"] = next((l for l in log.splitlines() if "[E]" in l), "build failed").strip()[:200]
        return res
    import tensorrt as trt

    _, eng = _load_engine(engine)
    info = eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    layers = json.loads(info)["Layers"]
    res["layers"] = len(layers)
    res["sparse_tactic_layers"] = sum(bool(SPARSE_TACTIC.search(l.get("TacticName") or ""))
                                      for l in layers if isinstance(l, dict))
    # "Found N layer(s) eligible ..." / "Chose M layer(s) using sparse tactics" are logged once per
    # tactic-selection pass; the last pair belongs to the final engine.
    found = re.findall(r"Found (\d+) layer\(s\) eligible to use sparse tactics", log)
    chose = re.findall(r"Chose (\d+) layer\(s\) using sparse tactics", log)
    res["eligible"] = int(found[-1]) if found else None
    res["chosen"] = int(chose[-1]) if chose else None
    return res


def numeric_error(engine_path, x, ref):
    """Max abs error / max |ref| of the saved fp16 engine against the fp32 ORT reference."""
    h, _ = _load_engine(engine_path)
    outs, _ms = h.run_engine(engine_path.read_bytes(), feeds={"X": x}, iters=1, warmup=1)
    y = next(iter(outs.values())).astype(np.float32)
    return float(np.abs(y - ref).max() / (np.abs(ref).max() + 1e-12))


def trt_stage(args):
    out = Path(args.out_dir)
    shapes = args.shapes or list(SHAPES)
    rows = {}
    for sname in shapes:
        x = np.load(out / f"{sname}.x.npy")
        for form in FORMS:
            ref = np.load(out / f"{sname}.ref_{form.split('_')[0]}.npy")
            path = out / f"{sname}.{form}.onnx"
            for mode in ("disable", "enable", "force"):
                r = {"shape": sname, "form": form, **build(path, mode, out)}
                if "error" not in r:
                    r["engine"] = str(out / f"{path.stem}.{mode}.engine")
                    if mode != "force":  # force treats dense weights as sparse: wrong by design
                        r["rel_err_fp16_vs_fp32"] = numeric_error(Path(r["engine"]), x, ref)
                rows[(sname, form, mode)] = r
                print(json.dumps({k: v for k, v in r.items() if k != "engine"}), flush=True)

    # Timing is round-robin over ALL engines (round 1: every engine once, round 2: ...), so slow
    # drift from thermals or a background job hits every variant equally; report the median.
    samples = {k: [] for k, r in rows.items() if "error" not in r}
    for rnd in range(args.runs):
        for k in samples:
            t = _mean_ms(_trtexec([f"--loadEngine={rows[k]['engine']}", "--noDataTransfers",
                                   f"--duration={args.duration}", "--warmUp=500"]))
            if t:
                samples[k].append(t)
        print(f"timing round {rnd + 1}/{args.runs} done", flush=True)
    for k, v in samples.items():
        rows[k]["ms_median"] = round(statistics.median(v), 4)
        rows[k]["ms_spread_pct"] = round((max(v) - min(v)) / min(v) * 100, 1)
        rows[k]["ms_samples"] = [round(t, 4) for t in v]

    result = list(rows.values())
    (out / "results.json").write_text(json.dumps(result, indent=1))
    print(f"\n{'shape':<8}{'form':<15}{'mode':<9}{'layers':>7}{'elig':>5}{'chose':>6}{'sptac':>6}"
          f"{'ms':>9}{'spread%':>8}{'relerr':>10}")
    for r in result:
        if "error" in r:
            print(f"{r['shape']:<8}{r['form']:<15}{r['mode']:<9}  ERROR {r['error']}")
            continue
        e = r.get("rel_err_fp16_vs_fp32")
        nz = lambda v: "-" if v is None else v
        print(f"{r['shape']:<8}{r['form']:<15}{r['mode']:<9}{r['layers']:>7}{nz(r['eligible']):>5}"
              f"{nz(r['chosen']):>6}{r['sparse_tactic_layers']:>6}{r['ms_median']:>9.3f}"
              f"{r['ms_spread_pct']:>8}{'' if e is None else f'{e:>10.2e}'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("out_dir")
    g.add_argument("--layers", type=int, default=6)
    g.set_defaults(fn=gen)
    t = sub.add_parser("trt")
    t.add_argument("out_dir")
    t.add_argument("--runs", type=int, default=3)
    t.add_argument("--duration", type=int, default=3)
    t.add_argument("--shapes", nargs="+", choices=list(SHAPES), help="subset of activation shapes")
    t.set_defaults(fn=trt_stage)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
