#!/usr/bin/env python3
"""Interactive chat demo on real AX650N hardware, measuring real
tokens/second: compiles a HF checkpoint once via
``pulsar2_docker.build_from_hf_checkpoint()``, then repeatedly runs the
SAME compiled ``.axmodel`` to generate one token at a time.

**No real KV cache** -- see ``build_from_hf_checkpoint()``'s docstring:
this generic ONNX-ingestion path has no incremental decode support (that's
Pulsar2's own closed-source ``pulsar2 llm_build`` + ``ax-llm`` runtime, see
``pulsar2_docker.llm_build()``). Instead, every generation step re-runs the
*entire* compiled forward pass over a fixed ``--max-seq-len`` window: the
already-generated prefix is written into ``input_ids``, everything after
it is left as token id 0, and ``position_ids`` is always
``arange(max_seq_len)``.

**This is still numerically correct, not an approximation** -- causal
masking in ``onnxsim.reconstruct_hf_graph()``'s graph is a static
per-position constant (independent of the actual token ids), so position
``i``'s output can never depend on any position ``j > i``. The token-id-0
padding beyond the real prefix has zero effect on the logits used to pick
the next real token. The real tradeoff is compute, not correctness: this
recomputes ``O(max_seq_len)`` work on *every* step regardless of how many
tokens are actually filled in, whereas a real KV-cache decode (Pulsar2's
own production path) does ``O(1)`` per step. So the tokens/sec this script
reports measures onnxsim's own integration path's real device behavior,
not Axera's production LLM serving numbers.

**Confirmed real, and the dominant cost by far**: each generation step
shells out to ``axcl_run_model`` fresh (see
``pulsar2_docker.run_on_device_with_inputs()``), and that CLI's own
per-invocation process-spawn + device-context + model-reload overhead is
~700ms -- confirmed by comparing a single bare call against a
``repeat=5, warmup=2`` call on the *same* process: both take ~0.72s
wall-clock, while `axcl_run_model`'s own reported per-inference average is
~0.6-0.8ms. That's about a 1000x gap between wall-clock tokens/sec (~1.4
tok/s measured against a tiny synthetic checkpoint) and the NPU's actual
compute-only latency. This script reports both numbers each turn --
neither this demo nor onnxsim implements a persistent-process/session API
that would amortize the reload cost away; that would need the real AXCL
SDK's C API (a loaded-once device/model handle reused across calls), not
just the `axcl_run_model` CLI tool this repo's tooling wraps.

Needs ``pip install tokenizers`` (not ``transformers`` -- lighter, and
sufficient to load a checkpoint's own ``tokenizer.json`` directly).

Usage:
    python scripts/axera/demo_hf_llm_chat.py --hf-dir ./tiny-random-mistral
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import onnx

_AXERA_DIR = os.path.dirname(os.path.abspath(__file__))
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import pulsar2_docker  # noqa: E402


def _read_config(hf_dir: str) -> dict:
    with open(os.path.join(hf_dir, "config.json")) as f:
        return json.load(f)


def _pack_int32(arr: np.ndarray) -> bytes:
    return arr.astype("<i4").tobytes()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-dir", required=True, help="HF checkpoint dir (config.json + safetensors)"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=32,
        help="fixed compiled window; prompt + generated tokens must fit in this",
    )
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--work-dir", default=None, help="default: a temp dir")
    parser.add_argument("--target-hardware", default="AX650")
    parser.add_argument("--image", default=pulsar2_docker.DEFAULT_IMAGE)
    args = parser.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("error: pip install tokenizers", file=sys.stderr)
        return 1

    tokenizer_path = os.path.join(args.hf_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"error: no tokenizer.json in {args.hf_dir}", file=sys.stderr)
        return 1
    tok = Tokenizer.from_file(tokenizer_path)

    config = _read_config(args.hf_dir)
    eos_token_id = config.get("eos_token_id")

    if not pulsar2_docker.docker_image_available(args.image):
        print(f"error: pulsar2 Docker image not loaded: {args.image}", file=sys.stderr)
        return 1
    if not pulsar2_docker.axcl_available():
        print("error: no AXCL device found", file=sys.stderr)
        return 1

    work_dir = args.work_dir or tempfile.mkdtemp(prefix="onnxsim_hf_llm_chat_")
    os.makedirs(work_dir, exist_ok=True)
    print(f"work dir: {work_dir}")
    print(f"compiling once for max_seq_len={args.max_seq_len} ...")

    result = pulsar2_docker.build_from_hf_checkpoint(
        args.hf_dir,
        work_dir,
        "output",
        batch_size=1,
        seq_len=args.max_seq_len,
        target_hardware=args.target_hardware,
        image=args.image,
    )
    if not result.success:
        print(f"build failed:\n{result.error}", file=sys.stderr)
        return 1
    print(f"compiled: {result.axmodel_path}")
    for phase, seconds in result.phase_timings.items():
        print(f"  {phase}: {seconds:.2f}s")

    compiled = onnx.load(result.axmodel_path)
    output_shape = tuple(
        d.dim_value for d in compiled.graph.output[0].type.tensor_type.shape.dim
    )

    print()
    print("Chat ready (Ctrl-D to quit).")
    while True:
        try:
            prompt = input("> ")
        except EOFError:
            print()
            break
        if not prompt.strip():
            continue

        generated = tok.encode(prompt).ids
        if len(generated) >= args.max_seq_len:
            print(
                f"[prompt is {len(generated)} tokens, >= max_seq_len={args.max_seq_len}]"
            )
            continue

        print("assistant: ", end="", flush=True)
        latencies = []
        device_ms = []
        n_generated = 0
        for _ in range(args.max_new_tokens):
            if len(generated) >= args.max_seq_len:
                break
            input_ids = np.zeros((1, args.max_seq_len), dtype=np.int64)
            input_ids[0, : len(generated)] = generated
            position_ids = np.arange(args.max_seq_len, dtype=np.int64)[None, :]

            t0 = time.perf_counter()
            run = pulsar2_docker.run_on_device_with_inputs(
                result.axmodel_path,
                {
                    "input_ids": _pack_int32(input_ids),
                    "position_ids": _pack_int32(position_ids),
                },
            )
            latencies.append(time.perf_counter() - t0)
            if run.outputs is None:
                print(f"\n[device run failed: {run.error}]", file=sys.stderr)
                break
            if run.avg_ms is not None:
                device_ms.append(run.avg_ms)

            logits = np.frombuffer(run.outputs[0], dtype="<f4").reshape(output_shape)
            next_id = int(np.argmax(logits[0, len(generated) - 1]))
            generated.append(next_id)
            n_generated += 1
            print(tok.decode([next_id]), end="", flush=True)
            if eos_token_id is not None and next_id == eos_token_id:
                break

        print()
        if latencies:
            total = sum(latencies)
            print(
                f"[wall-clock: {n_generated} tokens in {total:.3f}s -> "
                f"{n_generated / total:.2f} tok/s, avg step "
                f"{1000 * total / len(latencies):.1f}ms -- this includes "
                f"axcl_run_model's own per-invocation process/model-reload "
                f"cost, confirmed ~700ms fixed regardless of NPU compute]"
            )
            if device_ms:
                avg_device_ms = sum(device_ms) / len(device_ms)
                print(
                    f"[NPU compute only (axcl_run_model's own reported avg): "
                    f"{avg_device_ms:.3f}ms/step -> "
                    f"{1000 / avg_device_ms:.0f} tok/s if model-reload "
                    f"overhead were amortized away, e.g. by a persistent "
                    f"serving process -- not something this demo, or "
                    f"onnxsim, implements]"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
