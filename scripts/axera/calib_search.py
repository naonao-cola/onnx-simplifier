#!/usr/bin/env python3
"""Search Pulsar2 calibration parameters offline, without a card.

`pulsar2 build` turns a calibration method (`MinMax`, `Percentile`, `MSE`,
`KL`), a calibration size, and calibration data into per-tensor
scales/zero-points in `<output>/quant/quant_axmodel.json`. `replay.py`
reproduces the card's numerics from exactly that table (validated to a
fraction of a dB against real AX650N output), so the whole
method x size grid can be ranked by replay SNR right here -- no device,
no Docker beyond the builds themselves.

What the grid below found (three single-purpose probes -- a group-32
depthwise conv, a last-axis LayerNorm, a MatMul/Softmax/MatMul attention
fragment -- Pulsar2 7.0-lite, replay SNR in dB, mean (min) over 5 inputs):

======================  ============  ============  ============  ============
probe                   MinMax 8/32   Percentile 8/32  MSE/KL (both sizes)
======================  ============  ============  ============  ============
depthwise               36.0 (35.7) / 35.6 (35.3)   36.9 (35.7) / 36.5 (34.7)   == MinMax
layernorm               40.5 (29.0) / 41.5 (40.0)   26.1 (22.5) / 30.3 (23.8)   == MinMax
attn                    36.2 (32.1) / 34.8 (32.3)   32.9 (29.6) / 34.2 (30.1)   == MinMax
======================  ============  ============  ============  ============

Three findings, all actionable:

1. `MSE` and `KL` fall back to `MinMax` on these graphs: byte-identical
   scales at every size. Do not trust the method name -- diff the scales
   (`scripts/axera/README.md`'s calibration section has the one-liner).
2. LayerNorm wants `MinMax` and enough samples: 8 samples leaves the range
   underestimated badly enough that one input in five clips to 29 dB, while
   32 hold every seed above 40 dB. `Percentile` is actively harmful there
   (~26 dB): it trims exactly the tails a normaliser needs.
3. Everywhere else the choice barely matters (conv/attention saturate by 8
   samples; method differences are ~1 dB noise). The default (`MinMax`,
   size 32) is the right default.

Usage::

    calib_search.py --model model.onnx --input x --shape 1,4,8 \\
        --methods MinMax,Percentile --sizes 8,32 --seeds 5 --target AX650
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DEFAULT_IMAGE = "pulsar2:7.0-lite"
CALIB_SEED = 1000


def make_calibration_tar(
    out_path: str, shape: Sequence[int], n: int, seed: int = CALIB_SEED
) -> str:
    """`n` standard-normal samples of `shape` as a `.tar` of `.npy` files."""
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(n):
            p = os.path.join(td, f"{i}.npy")
            np.save(p, rng.standard_normal(tuple(shape)).astype(np.float32))
            paths.append(p)
        with tarfile.open(out_path, "w") as tf:
            for p in paths:
                tf.add(p, arcname=os.path.basename(p))
    return out_path


def write_config(
    path: str, input_name: str, calib_rel: str, method: str, size: int
) -> None:
    """A minimal Pulsar2 quant config sweeping one method and size."""
    with open(path, "w") as f:
        json.dump(
            {
                "model_type": "ONNX",
                "npu_mode": "NPU1",
                "quant": {
                    "input_configs": [
                        {
                            "tensor_name": input_name,
                            "calibration_dataset": "./" + calib_rel,
                            "calibration_format": "Numpy",
                            "calibration_size": size,
                        }
                    ],
                    "calibration_method": method,
                    "precision_analysis": False,
                },
                "compiler": {"check": 0},
            },
            f,
        )


def docker_rm(path: str, image: str = DEFAULT_IMAGE) -> None:
    """Delete a host path the builder container owns (root-written files)."""
    subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "rm", image, "-rf", path],
        capture_output=True,
        timeout=120,
    )


def build_cell(
    work_dir: str,
    model_name: str,
    config_name: str,
    output_name: str,
    target: str = "AX650",
    image: str = DEFAULT_IMAGE,
    timeout: int = 1200,
) -> Tuple[Optional[str], Optional[str]]:
    """One `pulsar2 build`; returns `(output_dir, None)` or `(None, error)`."""
    outdir = os.path.join(work_dir, output_name)
    if os.path.exists(outdir):
        shutil.rmtree(outdir, ignore_errors=True)
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{work_dir}:/data",
                "--entrypoint",
                "rm",
                image,
                "-rf",
                f"/data/{output_name}",
            ],
            capture_output=True,
            timeout=120,
        )
    cname = f"calib-search-{os.getpid()}-{int(time.time() * 1000)}"
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        cname,
        "-v",
        f"{work_dir}:/data",
        image,
        "pulsar2",
        "build",
        "--target_hardware",
        target,
        "--input",
        model_name,
        "--output_dir",
        output_name,
        "--config",
        config_name,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        log = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", cname], capture_output=True, timeout=30)
        return None, "TIMEOUT"
    axmodel = os.path.join(outdir, "compiled.axmodel")
    if proc.returncode != 0 or not os.path.exists(axmodel):
        return None, log[-1200:]
    return outdir, None


def score_cell(
    model_path: str, build_dir: str, input_name: str, shape: Sequence[int], seeds: int
) -> List[float]:
    """Replay SNR (dB) of one build over `seeds` random inputs."""
    import onnx
    import onnxruntime as ort
    import replay

    model = onnx.load(model_path)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    out = []
    for seed in range(seeds):
        x = np.random.default_rng(seed).standard_normal(tuple(shape)).astype(np.float32)
        ref = sess.run(None, {input_name: x})[0]
        got = replay.replay(model, build_dir, {input_name: x})[0]
        out.append(replay.snr_db(ref, got))
    return out


def rank_configs(scores: Dict[str, List[float]]) -> List[Tuple[str, float, float]]:
    """`(config, mean, min)` sorted best-first. Mean ranks, min breaks ties
    toward configs that do not collapse on any input (the layernorm-8
    failure mode: fine on average, 29 dB on one seed)."""
    ranked = [(k, statistics.mean(v), min(v)) for k, v in scores.items()]
    return sorted(ranked, key=lambda kv: (-kv[1], -kv[2]))


def tables_equal(build_a: str, build_b: str) -> bool:
    """Whether two builds' quant tables agree on every scale (catches a
    calibration method silently falling back to another, as `MSE`/`KL` do
    to `MinMax` on small graphs in 7.0-lite)."""
    import json as _json

    def sig(build_dir):
        with open(os.path.join(build_dir, "quant", "quant_axmodel.json")) as f:
            doc = _json.load(f)
        out = []
        for cfg in doc["tensor_configs"].values():
            for tensor, entry in cfg.items():
                value = doc["values"].get(str(entry.get("hash")))
                if value and value.get("scale"):
                    out.append((tensor, round(float(value["scale"][0]), 6)))
        return tuple(sorted(out))

    return sig(build_a) == sig(build_b)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument("--model", required=True, help="forward .onnx model")
    argp.add_argument("--input", required=True, help="model input tensor name")
    argp.add_argument("--shape", required=True, help="input shape, e.g. 1,4,8")
    argp.add_argument("--methods", default="MinMax,Percentile,MSE,KL")
    argp.add_argument("--sizes", default="8,32")
    argp.add_argument("--seeds", type=int, default=5)
    argp.add_argument("--calib-n", type=int, default=32)
    argp.add_argument("--target", default="AX650")
    argp.add_argument("--image", default=DEFAULT_IMAGE)
    argp.add_argument("--work-dir", default="calib_search_work")
    args = argp.parse_args(argv)

    shape = [int(x) for x in args.shape.split(",")]
    methods = [m for m in args.methods.split(",") if m]
    sizes = [int(s) for s in args.sizes.split(",") if s]
    work = os.path.abspath(args.work_dir)
    os.makedirs(os.path.join(work, "dataset"), exist_ok=True)
    calib_tar = make_calibration_tar(
        os.path.join(work, "dataset", "calib.tar"), shape, max(args.calib_n, max(sizes))
    )
    calib_rel = os.path.relpath(calib_tar, work)
    import shutil as _shutil

    _shutil.copy(args.model, os.path.join(work, "model.onnx"))

    baseline = None
    for method, size in itertools.product(methods, sizes):
        tag = f"{method.lower()}_s{size}"
        write_config(
            os.path.join(work, f"config_{tag}.json"),
            args.input,
            calib_rel,
            method,
            size,
        )
        outdir, error = build_cell(
            work,
            "model.onnx",
            f"config_{tag}.json",
            f"output_{tag}",
            target=args.target,
            image=args.image,
        )
        if outdir is None:
            print(f"{tag}: BUILD FAILED\n{error}")
            continue
        if baseline is None:
            baseline = outdir
        elif tables_equal(baseline, outdir):
            print(
                f"{tag}: scales identical to {os.path.basename(baseline)} -- method fell back"
            )
        scores = score_cell(
            os.path.join(work, "model.onnx"), outdir, args.input, shape, args.seeds
        )
        print(
            f"{tag}: SNR {statistics.mean(scores):.1f} dB "
            f"(min {min(scores):.1f} over {args.seeds} seeds)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
