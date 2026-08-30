#!/usr/bin/env python3
"""Greedy-decode a prompt through a Core ML LLM and measure decode throughput.

The counterpart to `export_llm_to_coreml.py`: loads the `.mlpackage` that
script produced (plus the tokenizer files it copied alongside it), greedily
generates tokens, and reports decode tokens/second and peak resident memory --
the same two axes DeviceMark's methodology measures on-device (see
https://devicemark.github.io/ and its METHODOLOGY.md), computed the same way
(wall-clock decode time on the actual runtime, RSS sampled during generation).

This only runs where Core ML actually executes models: macOS (loading the
model calls into Apple's Core ML framework, which isn't part of coremltools
itself -- see `onnxsim/coreml_export.py`'s module docstring). It has nothing
to do with WebGPU or any browser.

Decode strategy: the exported model has no growing KV cache -- every call
reprocesses the model's whole fixed-size context window from scratch (see
export_llm_to_coreml.py's docstring for why). So the tok/s this reports is
this recompute strategy's throughput, not what a real KV-cache-backed
deployment of the same model would achieve; treat it as a benchmark of this
pipeline, not a device capability number, until export_llm_to_coreml.py grows
real KV-cache (dynamic-shape) support.

Usage:
    python run_llm_decode_benchmark.py smollm2.mlpackage \\
        --prompt "The capital of France is" --max-new-tokens 20
"""

from __future__ import annotations

import argparse
import platform
import resource
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.proto import FeatureTypes_pb2 as _ft

_ARRAY_DTYPE = _ft.ArrayFeatureType.ArrayDataType
_DATATYPE_TO_NUMPY = {
    _ARRAY_DTYPE.FLOAT32: np.float32,
    _ARRAY_DTYPE.FLOAT16: np.float16,
    _ARRAY_DTYPE.INT32: np.int32,
    _ARRAY_DTYPE.DOUBLE: np.float64,
}


def _peak_rss_mb() -> float:
    """Peak resident set size of this process so far, in MB.

    `ru_maxrss` is bytes on macOS, KiB on Linux; there's no portable unit.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if platform.system() == "Darwin" else peak / 1e3


class CoreMLDecoder:
    """Wraps a fixed-context, empty-KV-cache Core ML decoder (see
    export_llm_to_coreml.py) for greedy generation.
    """

    def __init__(self, mlpackage_path: str, compute_units: str = "ALL"):
        self.mlmodel = ct.models.MLModel(
            mlpackage_path, compute_units=getattr(ct.ComputeUnit, compute_units)
        )
        spec = self.mlmodel.get_spec()
        self._input_shapes = {}
        self._input_dtypes = {}
        for inp in spec.description.input:
            arr = inp.type.multiArrayType
            self._input_shapes[inp.name] = list(arr.shape)
            self._input_dtypes[inp.name] = _DATATYPE_TO_NUMPY.get(
                arr.dataType, np.float32
            )
        (self._logits_name,) = [
            o.name for o in spec.description.output if o.name.startswith("logits")
        ]

        self.max_length = self._input_shapes["input_ids"][-1]
        self._past_kv_feeds = {
            name: np.zeros(shape, dtype=self._input_dtypes[name])
            for name, shape in self._input_shapes.items()
            if name.startswith("past_key_values")
        }

    def _forward(self, input_ids: np.ndarray, num_real: int) -> np.ndarray:
        """Run one forward pass over the padded window; return logits at the last
        real position."""
        attention_mask = np.zeros(
            (1, self.max_length), dtype=self._input_dtypes["attention_mask"]
        )
        attention_mask[0, :num_real] = 1
        position_ids = np.arange(
            self.max_length, dtype=self._input_dtypes["position_ids"]
        )[None, :]
        feeds = {
            "input_ids": input_ids.astype(self._input_dtypes["input_ids"]),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **self._past_kv_feeds,
        }
        out = self.mlmodel.predict(feeds)
        return out[self._logits_name][0, num_real - 1]

    def generate(
        self, prompt_ids: list[int], max_new_tokens: int, eos_token_id: int | None
    ):
        """Greedily generate up to `max_new_tokens` tokens after `prompt_ids`.

        Yields (token_id, seconds_for_this_step) per generated token. Stops early
        on `eos_token_id`, or once the fixed context window is full.
        """
        if len(prompt_ids) >= self.max_length:
            raise ValueError(
                f"prompt has {len(prompt_ids)} tokens, but this model's fixed "
                f"context window is only {self.max_length} (see --max-length in "
                "export_llm_to_coreml.py)"
            )
        input_ids = np.zeros((1, self.max_length), dtype=np.int64)
        input_ids[0, : len(prompt_ids)] = prompt_ids
        num_real = len(prompt_ids)

        while num_real < self.max_length and (
            max_new_tokens is None or max_new_tokens > 0
        ):
            t0 = time.perf_counter()
            logits = self._forward(input_ids, num_real)
            next_id = int(np.argmax(logits))
            dt = time.perf_counter() - t0

            yield next_id, dt
            if eos_token_id is not None and next_id == eos_token_id:
                return
            input_ids[0, num_real] = next_id
            num_real += 1
            if max_new_tokens is not None:
                max_new_tokens -= 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "mlpackage", help="Path to the .mlpackage produced by export_llm_to_coreml.py"
    )
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument(
        "--compute-units",
        default="ALL",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
    )
    args = ap.parse_args()

    model_dir = Path(args.mlpackage).parent
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {model_dir} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    print(
        f"Loading {args.mlpackage} (compute_units={args.compute_units}) ...", flush=True
    )
    decoder = CoreMLDecoder(args.mlpackage, compute_units=args.compute_units)
    print(f"Fixed context window: {decoder.max_length} tokens", flush=True)

    prompt_ids = tokenizer(args.prompt, return_tensors="np")["input_ids"][0].tolist()
    print(f"Prompt: {args.prompt!r} ({len(prompt_ids)} tokens)", flush=True)

    generated = []
    step_times = []
    for token_id, dt in decoder.generate(
        prompt_ids, args.max_new_tokens, tokenizer.eos_token_id
    ):
        generated.append(token_id)
        step_times.append(dt)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\nGenerated ({len(generated)} tokens): {text!r}", flush=True)

    if step_times:
        total = sum(step_times)
        print(f"\ndecode tok/s: {len(step_times) / total:.2f}", flush=True)
        print(f"mean step latency: {1000 * total / len(step_times):.1f} ms", flush=True)
    print(f"peak RSS: {_peak_rss_mb():.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
