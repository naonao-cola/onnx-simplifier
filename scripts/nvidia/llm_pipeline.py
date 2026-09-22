"""Fix a decoder-with-KV-cache LLM's shapes and run onnxsim on it, for TensorRT benchmarking.

Exercises onnxsim passes a CNN/ViT never touches: decomposed RMSNorm (``Pow``/
``ReduceMean``/``Sqrt``/``Div``), decomposed RoPE (``Unsqueeze``/``Mul``/``Neg``/``Concat``
rotate-half), grouped-query-attention head repetition (``Expand``), and a graph with 48
``past_key_values.N.{key,value}`` inputs / 48 ``present.N.{key,value}`` outputs (KV cache)
alongside ``input_ids``/``attention_mask``/``position_ids``.

Targets HF `optimum`-style "always with past" decoder exports (no ``If`` node selecting a
cacheless branch -- ``past_key_values`` are required inputs, so a length-0 KV cache stands
in for the first/prefill call). Tested against
https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct/blob/main/onnx/model_fp16.onnx
(opset 14, ir 10, 2759 nodes, 24 layers, 14 query / 2 KV heads, head_dim 64) but should work
for any model exported the same way.

    python llm_pipeline.py model_fp16.onnx OUT_DIR --seq 1 --past 31    # decode step
    python llm_pipeline.py model_fp16.onnx OUT_DIR --seq 8 --past 0     # prefill
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import onnx

import onnxsim


def fix_shapes(model, batch, seq, past):
    """Pin every dynamic dim (batch_size / sequence_length / past_sequence_length[ + 1])
    to a concrete value, by name -- HF's exporter reuses the same ``dim_param`` string
    across every matching tensor, so a name-keyed substitution catches all of them at
    once (48 KV inputs/outputs included) without walking the graph node by node."""
    model = onnx.ModelProto.FromString(model.SerializeToString())
    subs = {
        "batch_size": batch,
        "sequence_length": seq,
        "past_sequence_length": past,
        "past_sequence_length + 1": past + seq,
    }
    for io in list(model.graph.input) + list(model.graph.output):
        dims = io.type.tensor_type.shape.dim
        for d in dims:
            if d.dim_param in subs:
                v = subs[d.dim_param]
                d.Clear()
                d.dim_value = v
    del model.graph.value_info[:]
    return onnx.shape_inference.infer_shapes(model)


def make_feeds(model, batch, seq, past, vocab, seed=0):
    """Random-but-valid feeds: token ids in-vocab, causal attention_mask of all ones
    (every KV position, past + current, attended to -- the common case), position_ids
    continuing on from the past length."""
    rng = np.random.default_rng(seed)
    feeds = {
        "input_ids": rng.integers(0, vocab, (batch, seq)).astype(np.int64),
        "attention_mask": np.ones((batch, past + seq), dtype=np.int64),
        "position_ids": np.arange(past, past + seq, dtype=np.int64)[None].repeat(batch, 0),
    }
    for i in model.graph.input:
        if i.name.startswith("past_key_values."):
            shape = [d.dim_value for d in i.type.tensor_type.shape.dim]
            # Despite the checkpoint's weights being fp16, HF's exporter keeps KV-cache
            # I/O declared fp32 (keep_io_types-style), with Cast nodes at the boundary.
            dtype = onnx.helper.tensor_dtype_to_np_dtype(i.type.tensor_type.elem_type)
            feeds[i.name] = (rng.standard_normal(shape) * 0.1).astype(dtype)
    return feeds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, required=True, help="new tokens this call (1 = decode step)")
    ap.add_argument("--past", type=int, required=True, help="KV cache length already present (0 = prefill)")
    ap.add_argument("--tag", default=None, help="output file stem; default derived from --seq/--past")
    args = ap.parse_args(argv)

    tag = args.tag or (f"decode_p{args.past}" if args.seq == 1 else f"prefill_s{args.seq}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model} ...", flush=True)
    t0 = time.perf_counter()
    src = onnx.load(args.model)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s, {len(src.graph.node)} nodes")

    vocab = src.graph.output[0].type.tensor_type.shape.dim[-1].dim_value

    t0 = time.perf_counter()
    raw = fix_shapes(src, args.batch, args.seq, args.past)
    print(f"fix_shapes: {time.perf_counter() - t0:.1f}s")
    onnx.save(raw, out / f"{tag}.raw.onnx")

    t0 = time.perf_counter()
    sim, ok = onnxsim.simplify(raw)
    assert ok, "onnxsim correctness check failed"
    print(f"simplify: {len(raw.graph.node)} -> {len(sim.graph.node)} nodes "
          f"({time.perf_counter() - t0:.1f}s)")
    onnx.save(sim, out / f"{tag}.sim.onnx")

    feeds = make_feeds(raw, args.batch, args.seq, args.past, vocab)
    np.savez(out / f"{tag}.feeds.npz", **feeds)

    import collections

    for name, m in (("raw", raw), ("sim", sim)):
        c = collections.Counter(n.op_type for n in m.graph.node)
        interesting = {k: c[k] for k in
                      ("Pow", "ReduceMean", "Sqrt", "Div", "RMSNormalization",
                       "Neg", "Concat", "Expand", "Softmax", "MatMul", "Cast")
                      if c.get(k)}
        print(f"  {name}: {interesting}")


if __name__ == "__main__":
    sys.exit(main())
