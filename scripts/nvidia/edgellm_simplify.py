"""Run onnxsim on a TensorRT Edge-LLM ONNX export directory, ready for Edge-LLM's ``llm_build``.

``tensorrt-edgellm-export`` writes ``<out>/llm/`` = ``model.onnx`` + ``model.onnx.data`` plus
sidecar files the C++ builder/runtime read (``config.json``, ``embedding.safetensors``,
tokenizer and chat-template JSON). This simplifies ``model.onnx`` into a new directory,
copies the sidecars verbatim, and checks what ``llm_build`` depends on survived untouched:
graph input/output names and every custom-domain plugin node's inputs and attributes
(``trt_edgellm::AttentionPlugin`` etc., including their empty-string optional inputs).

    python edgellm_simplify.py EXPORT/llm SIM/llm
    ./build/examples/llm/llm_build --onnxDir SIM/llm --engineDir ENG ...

``rms-stack`` instead writes a plugin-free synthetic model for timing the RMSNorm question
in isolation with ``trtexec``: ``--layers`` Qwen3-0.6B-shaped
[fp32-upcast RMSNorm -> SwiGLU MLP -> residual] blocks, as exported and as onnxsim fuses
them (``rms_decomposed.onnx`` / ``rms_fused.onnx``).

    python edgellm_simplify.py rms-stack OUT_DIR
"""

import argparse
import collections
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, parser

import onnxsim


def _plugins(model):
    return [n for n in model.graph.node if n.domain not in ("", "ai.onnx")]


def simplify_dir(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    model = onnx.load(src / "model.onnx")
    t = time.time()
    sim, check_ok = onnxsim.simplify(model)
    print(f"simplify: {time.time() - t:.1f}s, check_ok={check_ok}")
    before = collections.Counter(n.op_type for n in model.graph.node)
    after = collections.Counter(n.op_type for n in sim.graph.node)
    print(f"nodes: {len(model.graph.node)} -> {len(sim.graph.node)}")
    for op in sorted(set(before) | set(after), key=lambda o: -before[o]):
        if before[op] != after[op]:
            print(f"  {op}: {before[op]} -> {after[op]}")

    problems = []
    for kind in ("input", "output"):
        a = [v.name for v in getattr(model.graph, kind)]
        b = [v.name for v in getattr(sim.graph, kind)]
        if a != b:
            problems.append(f"graph {kind} names changed")
    pa, pb = _plugins(model), _plugins(sim)
    if [(n.domain, n.op_type) for n in pa] != [(n.domain, n.op_type) for n in pb]:
        problems.append("custom-domain plugin nodes added/removed/reordered")
    else:
        for x, y in zip(pa, pb):
            if list(x.input) != list(y.input) or x.attribute != y.attribute:
                problems.append(
                    f"plugin node {x.name} ({x.op_type}) inputs/attributes changed"
                )
    print(f"plugin nodes: {len(pa)} ({', '.join(sorted({n.op_type for n in pa}))})")
    for p in problems:
        print("PROBLEM:", p)

    onnx.save(
        sim, dst / "model.onnx", save_as_external_data=True, location="model.onnx.data"
    )
    for f in src.iterdir():
        if f.name not in ("model.onnx", "model.onnx.data") and f.is_file():
            shutil.copy2(f, dst / f.name)
    return not problems


def rms_stack(out, layers=28, hidden=1024, inter=3072):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    body, inits, x = [], [], "X"
    for i in range(layers):
        body.append(f"""
          xu{i} = Cast<to = 1>({x})
          sq{i} = Pow(xu{i}, two)
          var{i} = ReduceMean<keepdims = 1>(sq{i}, axes)
          ve{i} = Add(var{i}, eps)
          r{i} = Sqrt(ve{i})
          ir{i} = Reciprocal(r{i})
          n{i} = Mul(xu{i}, ir{i})
          nt{i} = Cast<to = 10>(n{i})
          h{i} = Mul(w{i}, nt{i})
          g{i} = MatMul(h{i}, wg{i})
          sg{i} = Sigmoid(g{i})
          si{i} = Mul(g{i}, sg{i})
          u{i} = MatMul(h{i}, wu{i})
          m{i} = Mul(si{i}, u{i})
          d{i} = MatMul(m{i}, wd{i})
          x{i} = Add({x}, d{i})""")
        x = f"x{i}"
        for name, shape, scale, bias in (
            (f"w{i}", (hidden,), 0.02, 1.0),
            (f"wg{i}", (hidden, inter), 0.02, 0.0),
            (f"wu{i}", (hidden, inter), 0.02, 0.0),
            (f"wd{i}", (inter, hidden), 0.02, 0.0),
        ):
            w = bias + scale * rng.standard_normal(shape)
            inits.append(numpy_helper.from_array(w.astype(np.float16), name))
    model = parser.parse_model(f"""
        <ir_version: 10, opset_import: ["": 23]>
        g (float16[B, S, {hidden}] X) => (float16[B, S, {hidden}] {x})
        <float two = {{2.0}}, int64[1] axes = {{-1}}, float eps = {{1e-06}}>
        {{ {"".join(body)} }}
        """)
    model.graph.initializer.extend(inits)
    onnx.save(model, out / "rms_decomposed.onnx")
    sim, _ = onnxsim.simplify(model, check_n=0)
    onnx.save(sim, out / "rms_fused.onnx")
    print(f"rms_decomposed.onnx: {len(model.graph.node)} nodes")
    print(
        f"rms_fused.onnx: {len(sim.graph.node)} nodes, "
        f"{sum(n.op_type == 'RMSNormalization' for n in sim.graph.node)} RMSNormalization"
    )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "rms-stack":
        ap = argparse.ArgumentParser(prog="edgellm_simplify.py rms-stack")
        ap.add_argument("out_dir")
        ap.add_argument("--layers", type=int, default=28)
        args = ap.parse_args(sys.argv[2:])
        rms_stack(args.out_dir, args.layers)
        return 0
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("src_dir", help="Edge-LLM export's llm/ directory")
    ap.add_argument("dst_dir", help="output directory for llm_build --onnxDir")
    args = ap.parse_args()
    return 0 if simplify_dir(args.src_dir, args.dst_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
