#!/usr/bin/env python3
"""Real `pulsar2 build` wrapper -- Docker + (optionally) a real AXCL device.

Unlike `pulsar2_backend.py`/`pulsar2_simulator.py` (static/estimated, no
Docker or device needed -- see their docstrings for why), this module
actually shells out to a real Pulsar2 Docker image and, optionally, a real
device via `axcl_run_model`. It exists because a real toolchain + AX650N
*was* available in the session that produced the confirmed data in
`pulsar2_ops.py`'s docstring: `resnet18d_Opset18` built cleanly and matched
bit-for-bit on real hardware before/after onnxsim simplification;
`googlenet-6` hard-failed on `LRN`. This module is what did that manually,
turned into a reusable wrapper.

**Requires a local Docker image already loaded** (`docker load -i
ax_pulsar2_<version>_lite.tar.gz`, from
https://huggingface.co/AXERA-TECH/Pulsar2 -- match the version
`axcl-smi`/`axcl_run_model`'s `pulsar2 ver:` line reports for your device)
and, for `run_on_device()`, a connected AXCL device (`axcl-smi` to check).
Neither is available in ordinary CI -- this is a manual/local-only tool, not
wired into `run_pulsar2_compat.py`.

**Profiling** (`profile=True` on `build()`): passes Pulsar2's own
`--compiler.npu_perf --debug.dump_frontend_graph` flags. Confirmed real: the
build then writes a standard Chrome Trace Event Format `trace.json` under
`<output_dir>/compiler/debug/subgraph_npu_0/b1/trace.json` (load it at
`chrome://tracing`) plus a flat `op_profile.csv` next to it, and dumps the
optimized quantized graph to `<output_dir>/frontend/
optimized_quant_axmodel.onnx` (open in Netron) so trace task labels can be
matched back to op names. See `README.md`'s "Real NPU profiling" section.

**`llm_build()`**: wraps `pulsar2 llm_build`, confirmed real end-to-end
against `Qwen/Qwen3-0.6B` (a real HF checkpoint) on this same toolchain +
AX650N. Unlike `build()`, this does **not** take an ONNX model at all --
`--input_path` is a raw HuggingFace checkpoint directory (`*.safetensors` or
`pytorch_model.bin` + `config.json`), and there is no ONNX step anywhere in
the pipeline: the public `ax-llm-build` project
(https://github.com/AXERA-TECH/ax-llm-build) that Pulsar2's own docs point
to for this workflow contains no model-tracing/export code at all, only
per-architecture config JSONs and small pre/post-processing shell/Python
helpers around the actual (closed-source) `pulsar2 llm_build` call. So
**onnxsim has no direct integration point in this ingestion path** -- there
is no ONNX graph to simplify or quantize before Pulsar2 ever sees it.

Also note the CLI surface differs from Pulsar2's own V7.0 docs, which
document a `llm_build2` subcommand with `--max_context`/
`--prefill_step_size`/`--decode_step_size`: the `pulsar2:6.0-lite` image
this was verified against only has `llm_build` (no "2"), with
`--kv_cache_len` instead of those three flags -- `llm_build()`'s defaults
match this confirmed real v6.0 surface, not the newer docs.

Confirmed real output shape (`Qwen3-0.6B`, `--chip AX650 --prefill_len 512`):
one `<name>_p<prefill_len>_l<N>_together.axmodel` per transformer layer (28
for this model) plus one `<name>_post.axmodel` (the LM head). Each per-layer
file is **two** `neu mode` nodes in one graph, not one -- `subgraph_npu_0`
(decode, batch-1 shapes) and `subgraph_npu_1` (prefill, `prefill_len`-batch
shapes), both sharing the same `npu_params` initializer and each with
explicit `K_cache`/`V_cache` graph inputs *and* `K_cache_out`/`V_cache_out`
outputs -- the KV cache is passed as ordinary graph tensors the host runtime
(`ax-llm`/`axllm`) persists between calls, not hidden inside the compiled
blob. `pulsar2_ops.has_out_of_band_npu_data()`/`missing_npu_data()` (see
that module's docstring) already generalize correctly to this two-node
shape with no changes needed -- verified: `onnxsim.simplify()` corrupts a
real per-layer `.axmodel` the same way it does the CNN case (3 initializers
-> 0). Both a decode-shaped layer and the post model ran successfully on the
real AX650N via `axcl_run_model` (~1.5ms and ~9ms respectively).
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

DEFAULT_IMAGE = "pulsar2:6.0-lite"

_MAX_CYCLE_RE = re.compile(r"max_cycle\s*=\s*([\d,]+)")
_FUSE_RE = re.compile(r"fuse (\d+) subgraph\(s\)")


def docker_image_available(image: str = DEFAULT_IMAGE) -> bool:
    proc = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True
    )
    return proc.returncode == 0


def force_rmtree(path: str, work_dir: str, image: str) -> None:
    """Remove a directory `pulsar2 build` wrote into.

    The container runs as root (confirmed: `-u $(id -u):$(id -g)` breaks the
    image -- it needs root-owned `/root/*.hasplm`/`*.v2c` license files, and
    a non-existent-in-container uid breaks `getpass.getuser()` deep in a
    torchvision import), so files/dirs it wrote are root-owned and a plain
    `shutil.rmtree` as an ordinary host user raises `PermissionError`. Fall
    back to removing it from inside a container (same uid that created it).
    """
    try:
        shutil.rmtree(path)
        return
    except PermissionError:
        pass
    rel = os.path.relpath(path, work_dir)
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{work_dir}:/data",
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            f"rm -rf /data/{rel}",
        ],
        capture_output=True,
        text=True,
    )


@dataclass
class BuildResult:
    success: bool
    axmodel_path: Optional[str] = None
    max_cycle: Optional[int] = None
    fused_subgraphs: Optional[int] = None
    trace_path: Optional[str] = None
    frontend_graph_path: Optional[str] = None
    error: Optional[str] = None
    stdout_tail: str = ""


def _image_classifier_config(
    tensor_name: str,
    mean: Sequence[float],
    std: Sequence[float],
    *,
    tensor_format: str = "RGB",
    calibration_dataset: str = "./dataset/calib.tar",
    calibration_size: int = 16,
) -> dict:
    """The config.json shape confirmed against Pulsar2's own quick-start docs
    and the real `resnet18d`/`googlenet-6` builds -- see `pulsar2_ops.py`'s
    docstring. Only fits a single-image-input classifier; a model with a
    different input shape (NLP token ids, multiple inputs, ...) needs its
    own config -- build one by hand and call `build()` with
    `config_path=...` instead of `tensor_name`/`mean`/`std`.
    """
    return {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": [
                {
                    "tensor_name": tensor_name,
                    "calibration_dataset": calibration_dataset,
                    "calibration_size": calibration_size,
                    "calibration_mean": list(mean),
                    "calibration_std": list(std),
                }
            ],
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "input_processors": [
            {
                "tensor_name": tensor_name,
                "tensor_format": tensor_format,
                "src_format": tensor_format,
                "src_dtype": "U8",
                "src_layout": "NHWC",
                "csc_mode": "NoCSC",
            }
        ],
        "compiler": {"check": 0},
    }


def make_synthetic_calibration_tar(
    out_path: str, shape: Sequence[int] = (224, 224, 3), count: int = 16, seed: int = 0
) -> str:
    """A tar of `count` random uint8 images -- no accuracy claim, just enough
    for PTQ calibration to run (see `pulsar2_ops.py`'s docstring for why a
    compatibility check doesn't need a real, representative dataset the way
    a deployment accuracy check would)."""
    import numpy as np
    from PIL import Image

    rng = np.random.RandomState(seed)
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(count):
            arr = rng.randint(0, 256, shape, dtype=np.uint8)
            p = os.path.join(td, f"img_{i:02d}.jpg")
            Image.fromarray(arr).save(p, quality=90)
            paths.append(p)
        with tarfile.open(out_path, "w") as tf:
            for p in paths:
                tf.add(p, arcname=os.path.basename(p))
    return out_path


def build(
    work_dir: str,
    onnx_rel_path: str,
    output_rel_dir: str,
    *,
    tensor_name: Optional[str] = None,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    tensor_format: str = "RGB",
    calibration_dataset_rel_path: str = "dataset/calib.tar",
    calibration_size: int = 16,
    config_path: Optional[str] = None,
    target_hardware: str = "AX650",
    image: str = DEFAULT_IMAGE,
    profile: bool = False,
    timeout: int = 900,
) -> BuildResult:
    """Run a real `pulsar2 build` in Docker.

    `work_dir` is mounted at `/data` in the container -- `onnx_rel_path`,
    `output_rel_dir`, and `calibration_dataset_rel_path` are all relative to
    it, matching Pulsar2's own quick-start layout (see `pulsar2_ops.py`'s
    docstring). Either pass `tensor_name`/`mean`/`std` for the confirmed
    single-image-classifier config shape (auto-written under
    `work_dir/config/`), or `config_path` (relative to `work_dir`) for a
    hand-written one. With `profile=True`, adds `--compiler.npu_perf
    --debug.dump_frontend_graph` and locates the resulting `trace.json` /
    `optimized_quant_axmodel.onnx`.
    """
    if not docker_image_available(image):
        return BuildResult(success=False, error=f"docker image not loaded: {image}")

    if config_path is None:
        if tensor_name is None or mean is None or std is None:
            return BuildResult(
                success=False,
                error="pass tensor_name/mean/std, or an explicit config_path",
            )
        config_dir = os.path.join(work_dir, "config")
        os.makedirs(config_dir, exist_ok=True)
        config_rel_path = f"config/_auto_{os.path.basename(output_rel_dir)}.json"
        with open(os.path.join(work_dir, config_rel_path), "w") as f:
            json.dump(
                _image_classifier_config(
                    tensor_name,
                    mean,
                    std,
                    tensor_format=tensor_format,
                    calibration_dataset=f"./{calibration_dataset_rel_path}",
                    calibration_size=calibration_size,
                ),
                f,
            )
    else:
        config_rel_path = config_path

    output_abs_dir = os.path.join(work_dir, output_rel_dir)
    if os.path.exists(output_abs_dir):
        force_rmtree(output_abs_dir, work_dir, image)

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{work_dir}:/data",
        image,
        "pulsar2",
        "build",
        "--target_hardware",
        target_hardware,
        "--input",
        onnx_rel_path,
        "--output_dir",
        output_rel_dir,
        "--config",
        config_rel_path,
    ]
    if profile:
        cmd += ["--compiler.npu_perf", "--debug.dump_frontend_graph"]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return BuildResult(
            success=False, error=f"pulsar2 build timed out after {timeout}s"
        )

    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return BuildResult(success=False, error=log[-2000:], stdout_tail=log[-2000:])

    axmodel_path = os.path.join(output_abs_dir, "compiled.axmodel")
    max_cycle_match = _MAX_CYCLE_RE.search(log)
    fuse_match = _FUSE_RE.search(log)

    trace_path = None
    frontend_graph_path = None
    if profile:
        trace_hits = glob.glob(
            os.path.join(output_abs_dir, "compiler", "debug", "**", "trace.json"),
            recursive=True,
        )
        trace_path = trace_hits[0] if trace_hits else None
        candidate = os.path.join(
            output_abs_dir, "frontend", "optimized_quant_axmodel.onnx"
        )
        frontend_graph_path = candidate if os.path.exists(candidate) else None

    return BuildResult(
        success=os.path.exists(axmodel_path),
        axmodel_path=axmodel_path if os.path.exists(axmodel_path) else None,
        max_cycle=(
            int(max_cycle_match.group(1).replace(",", "")) if max_cycle_match else None
        ),
        fused_subgraphs=int(fuse_match.group(1)) if fuse_match else None,
        trace_path=trace_path,
        frontend_graph_path=frontend_graph_path,
        stdout_tail=log[-2000:],
    )


@dataclass
class LLMBuildResult:
    success: bool
    output_dir: Optional[str] = None
    layer_axmodels: List[str] = field(default_factory=list)
    embed_files: List[str] = field(default_factory=list)
    post_axmodel: Optional[str] = None
    error: Optional[str] = None
    stdout_tail: str = ""


def llm_build(
    work_dir: str,
    hf_checkpoint_rel_path: str,
    output_rel_dir: str,
    *,
    hidden_state_type: str = "bf16",
    weight_type: str = "s8",
    prefill_len: int = 512,
    kv_cache_len: int = 1023,
    chip: str = "AX650",
    parallel: int = 8,
    image: str = DEFAULT_IMAGE,
    timeout: int = 3600,
) -> LLMBuildResult:
    """Run a real `pulsar2 llm_build` in Docker on a HuggingFace checkpoint.

    See this module's docstring for the confirmed real facts this wraps:
    `hf_checkpoint_rel_path` (relative to `work_dir`, mounted at `/data`) is
    a raw HF checkpoint directory, not an ONNX model or a `pulsar2 build`
    config -- there is no config.json equivalent here, Pulsar2 reads the
    checkpoint's own `config.json` directly. `weight_type="s8"` matches
    Pulsar2's own default (its built-in quantization, not
    `pulsar2_quantizer.py` -- onnxsim has no role in this path).

    Confirmed real timing: ~7-8 minutes for a 0.6B model with `parallel=8`
    on a 32-core host.
    """
    if not docker_image_available(image):
        return LLMBuildResult(success=False, error=f"docker image not loaded: {image}")

    output_abs_dir = os.path.join(work_dir, output_rel_dir)
    if os.path.exists(output_abs_dir):
        force_rmtree(output_abs_dir, work_dir, image)

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{work_dir}:/data",
        image,
        "pulsar2",
        "llm_build",
        "--input_path",
        hf_checkpoint_rel_path,
        "--output_path",
        output_rel_dir,
        "--hidden_state_type",
        hidden_state_type,
        "--weight_type",
        weight_type,
        "--prefill_len",
        str(prefill_len),
        "--kv_cache_len",
        str(kv_cache_len),
        "--chip",
        chip,
        "--parallel",
        str(parallel),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return LLMBuildResult(
            success=False, error=f"pulsar2 llm_build timed out after {timeout}s"
        )

    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return LLMBuildResult(success=False, error=log[-2000:], stdout_tail=log[-2000:])

    layer_axmodels = sorted(
        glob.glob(os.path.join(output_abs_dir, "*_l*_together.axmodel"))
    )
    post_hits = glob.glob(os.path.join(output_abs_dir, "*_post.axmodel"))
    embed_files = sorted(glob.glob(os.path.join(output_abs_dir, "*embed_tokens*")))

    return LLMBuildResult(
        success=bool(layer_axmodels) and bool(post_hits),
        output_dir=output_abs_dir,
        layer_axmodels=layer_axmodels,
        embed_files=embed_files,
        post_axmodel=post_hits[0] if post_hits else None,
        stdout_tail=log[-2000:],
    )


def axcl_available(binary: str = "/usr/bin/axcl/axcl_run_model") -> bool:
    return os.path.exists(binary)


def run_on_device(
    axmodel_path: str,
    *,
    repeat: int = 5,
    warmup: int = 2,
    binary: str = "/usr/bin/axcl/axcl_run_model",
    timeout: int = 120,
) -> Dict[str, Optional[float]]:
    """Run a compiled `.axmodel` on a real AXCL device, return latency stats.

    Returns `{"min_ms": ..., "max_ms": ..., "avg_ms": ...}`, or all `None`
    plus `"error"` if the binary/device isn't available or the run failed.
    """
    if not axcl_available(binary):
        return {
            "min_ms": None,
            "max_ms": None,
            "avg_ms": None,
            "error": "axcl_run_model not found",
        }
    try:
        proc = subprocess.run(
            [binary, "-m", axmodel_path, "-r", str(repeat), "-w", str(warmup)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"min_ms": None, "max_ms": None, "avg_ms": None, "error": "timeout"}
    log = proc.stdout + proc.stderr
    m = re.search(
        r"min\s*=\s*([\d.]+)\s*ms\s*max\s*=\s*([\d.]+)\s*ms\s*avg\s*=\s*([\d.]+)\s*ms",
        log,
    )
    if not m:
        return {"min_ms": None, "max_ms": None, "avg_ms": None, "error": log[-500:]}
    return {
        "min_ms": float(m.group(1)),
        "max_ms": float(m.group(2)),
        "avg_ms": float(m.group(3)),
        "error": None,
    }


def run_on_device_with_input(
    axmodel_path: str,
    input_tensor_name: str,
    input_bytes: bytes,
    *,
    binary: str = "/usr/bin/axcl/axcl_run_model",
    timeout: int = 120,
) -> Optional[List[bytes]]:
    """Feed exactly `input_bytes` to `input_tensor_name` on a real device and
    return the raw output tensor bytes (one per output, model's declared
    order). Uses the confirmed real `-i/-o/-l` folder layout: `<in>/0/
    <tensor_name>.bin`, `list.txt` containing `0`, outputs land at `<out>/0/
    <output_name>.bin`. **The input filename must exactly match the tensor
    name** -- confirmed by trial: `axcl_run_model` errors with "Stimulus
    file ... is not exist" naming the *tensor's* name specifically, not an
    arbitrary/first-found file.
    """
    if not axcl_available(binary):
        return None
    with tempfile.TemporaryDirectory() as td:
        in_dir = os.path.join(td, "in", "0")
        out_dir = os.path.join(td, "out")
        os.makedirs(in_dir)
        os.makedirs(out_dir)
        with open(os.path.join(in_dir, f"{input_tensor_name}.bin"), "wb") as f:
            f.write(input_bytes)
        list_path = os.path.join(td, "list.txt")
        with open(list_path, "w") as f:
            f.write("0\n")
        try:
            subprocess.run(
                [
                    binary,
                    "-m",
                    axmodel_path,
                    "-i",
                    os.path.join(td, "in"),
                    "-o",
                    out_dir,
                    "-l",
                    list_path,
                    "-r",
                    "1",
                    "-w",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None
        outputs = []
        for p in sorted(glob.glob(os.path.join(out_dir, "0", "*.bin"))):
            with open(p, "rb") as f:
                outputs.append(f.read())
        return outputs or None
