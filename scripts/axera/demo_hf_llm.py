#!/usr/bin/env python3
"""End-to-end demo: HF checkpoint (config.json + safetensors) -> onnx (via
onnxsim.reconstruct_hf_graph()) -> real pulsar2 build (Docker) -> run on a
real AX650N (AXCL) -> decode the predicted next token.

See scripts/axera/README.md's "An alternative LLM path that does give
onnxsim a hook" section and pulsar2_docker.build_from_hf_checkpoint()'s
docstring for the full background; tests/test_pulsar2_hf_to_axmodel.py is
the automated version of what this script does interactively.

**Confirmed real gotcha, not just a style choice**: reconstruct_hf_graph()
declares its `input_ids`/`position_ids` graph inputs as ONNX INT64, but a
real compiled `.axmodel` re-declares them as INT32 (elem_type 6) --
confirmed by inspecting a real `compiled.axmodel`'s own `graph.input`.
Feed the device INT64 bytes and `axcl_run_model` will either reject them or
silently misread the buffer; this script writes INT32.

Usage:
    python scripts/axera/demo_hf_llm.py --hf-dir /path/to/hf_checkpoint

Only a checkpoint's architecture (llama/mistral/qwen2/qwen3 --
see onnxsim.reconstruct_hf_graph()) and shape need to be small enough to
compile in reasonable time: a full-size multi-billion-parameter checkpoint
through this *generic* ONNX ingestion path (as opposed to Pulsar2's own
`pulsar2 llm_build`, see `pulsar2_docker.llm_build()`) has not been
verified and may be slow or fail outright -- this generic path was only
confirmed against a tiny (2-layer, hidden_size=16) synthetic checkpoint.
For real multi-token generation on real hardware, `llm_build()` (Pulsar2's
own closed-source LLM ingestion, with real incremental KV-cache decode) is
the confirmed, production path -- this script demonstrates onnxsim's own
graph-construction hook instead, which only does a single-shot forward
pass (no KV cache): each run recomputes the whole prompt from scratch and
predicts one next token, it does not generate a whole continuation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

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
    parser.add_argument("--work-dir", default=None, help="default: a temp dir")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--calibration-size", type=int, default=4)
    parser.add_argument("--target-hardware", default="AX650")
    parser.add_argument("--image", default=pulsar2_docker.DEFAULT_IMAGE)
    parser.add_argument(
        "--prompt-ids",
        default=None,
        help="comma-separated token ids, length seq_len (default: random)",
    )
    parser.add_argument(
        "--skip-device", action="store_true", help="only build, don't run on hardware"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "pass --compiler.npu_perf --debug.dump_frontend_graph to pulsar2 "
            "build, for a per-op NPU trace.json (load at chrome://tracing) "
            "plus optimized_quant_axmodel.onnx (open in Netron) -- see "
            "README.md's 'Real NPU profiling' section"
        ),
    )
    args = parser.parse_args()

    if not pulsar2_docker.docker_image_available(args.image):
        print(f"error: pulsar2 Docker image not loaded: {args.image}", file=sys.stderr)
        return 1

    work_dir = args.work_dir or tempfile.mkdtemp(prefix="onnxsim_hf_llm_demo_")
    os.makedirs(work_dir, exist_ok=True)
    print(f"work dir: {work_dir}")

    print("building onnx graph + compiling with pulsar2 build ...")
    result = pulsar2_docker.build_from_hf_checkpoint(
        args.hf_dir,
        work_dir,
        "output",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        calibration_size=args.calibration_size,
        target_hardware=args.target_hardware,
        image=args.image,
        profile=args.profile,
    )
    if not result.success:
        print(f"build failed:\n{result.error}", file=sys.stderr)
        return 1
    print(f"compiled: {result.axmodel_path}")
    print(f"max_cycle={result.max_cycle} fused_subgraphs={result.fused_subgraphs}")
    print("phase timings:")
    for phase, seconds in result.phase_timings.items():
        print(f"  {phase}: {seconds:.2f}s")
    if args.profile:
        if result.trace_path is not None:
            print(f"NPU trace (open at chrome://tracing): {result.trace_path}")
        else:
            print(
                f"no trace.json found; build log tail:\n{result.stdout_tail}",
                file=sys.stderr,
            )
        if result.frontend_graph_path is not None:
            print(
                f"optimized quantized graph (open in Netron): {result.frontend_graph_path}"
            )

    if args.skip_device:
        return 0

    if not pulsar2_docker.axcl_available():
        print("no AXCL device found -- skipping on-device run")
        return 0

    config = _read_config(args.hf_dir)
    vocab_size = int(config.get("vocab_size", 32000))

    rng = np.random.default_rng(0)
    if args.prompt_ids:
        ids = [int(x) for x in args.prompt_ids.split(",")]
        if len(ids) != args.seq_len:
            print(
                f"error: --prompt-ids has {len(ids)} ids, need --seq-len={args.seq_len}",
                file=sys.stderr,
            )
            return 1
        input_ids = np.array([ids], dtype=np.int64)
    else:
        input_ids = rng.integers(
            0, vocab_size, size=(args.batch_size, args.seq_len)
        ).astype(np.int64)
    position_ids = np.arange(args.seq_len, dtype=np.int64)[None, :].repeat(
        args.batch_size, axis=0
    )
    print(f"prompt input_ids: {input_ids.tolist()}")

    import onnx

    compiled = onnx.load(result.axmodel_path)
    output_name = compiled.graph.output[0].name
    output_shape = tuple(
        d.dim_value for d in compiled.graph.output[0].type.tensor_type.shape.dim
    )

    # repeat=5, warmup=2: also benchmarks, with real in-range input data --
    # see run_on_device_with_inputs()'s docstring for why run_on_device()'s
    # own random-fill benchmarking mode is unsafe for a model with a
    # bounded-range input like a token id feeding an embedding Gather.
    run = pulsar2_docker.run_on_device_with_inputs(
        result.axmodel_path,
        {
            "input_ids": _pack_int32(input_ids),
            "position_ids": _pack_int32(position_ids),
        },
        repeat=5,
        warmup=2,
    )
    if run.outputs is None:
        print(f"on-device run with real input failed: {run.error}", file=sys.stderr)
        return 1
    if run.avg_ms is not None:
        print(
            f"on-device latency: min={run.min_ms}ms max={run.max_ms}ms "
            f"avg={run.avg_ms}ms"
        )

    logits = np.frombuffer(run.outputs[0], dtype="<f4").reshape(output_shape)
    next_token_logits = logits[0, -1]
    top5 = np.argsort(next_token_logits)[::-1][:5]
    print(f"output tensor: {output_name}, shape {output_shape}")
    print("top-5 predicted next tokens (id: logit):")
    for tok_id in top5:
        print(f"  {tok_id}: {next_token_logits[tok_id]:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
