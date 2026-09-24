#!/usr/bin/env python3
"""Benchmark a staged ONNX deployment across Core ML and tinygrad Metal.

The JSON manifest keeps task-specific preprocessing outside the runtime. Each
stage is an ONNX model with an optional NPZ file containing its non-connected
inputs. Connections route named outputs from an earlier stage to named inputs
of a later stage, so the same runner can cover encoder/head, detector/backbone,
and other multi-graph pipelines.

Manifest example::

  {
    "stages": [
      {"name": "encoder", "model": "encoder.onnx", "feeds": "image.npz"},
      {"name": "head", "model": "head.onnx", "feeds": "prompts.npz"}
    ],
    "connections": [
      {"from": ["encoder", "features"], "to": ["head", "features"]}
    ],
    "backends": {"encoder": "coreml", "head": "tinygrad_metal_jit"}
  }

NPZ files can be created with ``numpy.savez(path, input_name=array, ...)``.
Without a feeds file, static inputs receive deterministic seeded values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from benchmark_sam_hybrid import (
    _coreml_runner,
    _inputs,
    _measure,
    _MetalRunner,
    _quality,
)

BACKENDS = ("coreml", "tinygrad_metal", "tinygrad_metal_jit")


def _load_feeds(stage: dict, model_path: Path, seed: int) -> dict[str, np.ndarray]:
    feed_path = stage.get("feeds")
    if feed_path is None:
        return _inputs(onnx.load(model_path), seed)
    with np.load(model_path.parent / feed_path) as data:
        return {name: np.ascontiguousarray(data[name]) for name in data.files}


def _runner(
    path: Path, backend: str, package: Path, compute_units: str, compute_precision: str
):
    if backend == "coreml":
        return _coreml_runner(path, compute_units, compute_precision, package)
    return _MetalRunner(path, jit=backend == "tinygrad_metal_jit")


def _ort(path: Path, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [x.name for x in session.get_outputs()]
    return dict(zip(names, session.run(None, feeds)))


def run_pipeline(
    manifest_path: Path,
    output_path: Path,
    compute_units: str,
    compute_precision: str,
    warmup: int,
    repeats: int,
) -> dict:
    spec = json.loads(manifest_path.read_text())
    stages = spec["stages"]
    if not stages:
        raise ValueError("pipeline manifest must contain at least one stage")
    names = [stage["name"] for stage in stages]
    if len(set(names)) != len(names):
        raise ValueError("stage names must be unique")
    paths = {
        stage["name"]: (manifest_path.parent / stage["model"]).resolve()
        for stage in stages
    }
    output_names = {
        name: [value.name for value in onnx.load(path).graph.output]
        for name, path in paths.items()
    }
    feeds = {
        stage["name"]: _load_feeds(stage, paths[stage["name"]], i + 1)
        for i, stage in enumerate(stages)
    }
    connections = spec.get("connections", [])
    backend_map = spec.get("backends", {})
    for stage_name, backend in backend_map.items():
        if stage_name not in names or backend not in BACKENDS:
            raise ValueError(f"invalid backend assignment {stage_name}: {backend}")

    refs: dict[str, dict[str, np.ndarray]] = {}
    stage_records = {}
    package_dir = output_path.with_suffix("").with_name(output_path.stem + "_coreml")
    package_dir.mkdir(parents=True, exist_ok=True)
    runners = {}
    for stage in stages:
        name, path = stage["name"], paths[stage["name"]]
        stage_feeds = dict(feeds[name])
        for edge in connections:
            if edge["to"][0] == name:
                source_name, source_output = edge["from"]
                if source_name not in refs:
                    raise ValueError("connections must follow stage order")
                stage_feeds[edge["to"][1]] = refs[source_name][source_output]
        refs[name] = _ort(path, stage_feeds)
        record = {
            "input_shapes": {k: list(v.shape) for k, v in stage_feeds.items()},
            "backends": {},
        }
        for backend in BACKENDS:
            try:
                runner = _runner(
                    path,
                    backend,
                    package_dir / f"{name}.mlpackage",
                    compute_units,
                    compute_precision,
                )
                runners[name, backend] = runner
                out, timing = _measure(runner, stage_feeds, warmup, repeats)
                reference = list(refs[name].values())
                record["backends"][backend] = {
                    **timing,
                    "vs_ort": _quality(out, reference, output_names[name]),
                }
            except Exception as exc:
                record["backends"][backend] = {"error": f"{type(exc).__name__}: {exc}"}
        stage_records[name] = record

    selected = {name: backend_map.get(name, "coreml") for name in names}
    if any(backend not in BACKENDS for backend in selected.values()):
        raise ValueError(f"unsupported backend in {selected}")
    pipeline_result = None
    try:
        unavailable = [
            f"{name}={backend}: {stage_records[name]['backends'].get(backend, {}).get('error', 'runner unavailable')}"
            for name, backend in selected.items()
            if (name, backend) not in runners
        ]
        if unavailable:
            raise RuntimeError(
                "selected backend unavailable: " + "; ".join(unavailable)
            )
        selected_runners = {
            stage["name"]: runners[stage["name"], selected[stage["name"]]]
            for stage in stages
        }

        def execute(_):
            values: dict[str, dict[str, np.ndarray]] = {}
            for stage in stages:
                name = stage["name"]
                stage_feeds = dict(feeds[name])
                for edge in connections:
                    if edge["to"][0] == name:
                        stage_feeds[edge["to"][1]] = values[edge["from"][0]][
                            edge["from"][1]
                        ]
                result = selected_runners[name](stage_feeds)
                values[name] = dict(zip(output_names[name], result))
            last = stages[-1]["name"]
            return list(values[last].values())

        out, timing = _measure(execute, {}, warmup, repeats)
        last_name = stages[-1]["name"]
        pipeline_result = {
            "status": "ok",
            "backends": selected,
            **timing,
            "vs_ort": _quality(
                out,
                list(refs[last_name].values()),
                output_names[last_name],
            ),
        }
    except Exception as exc:
        pipeline_result = {
            "status": "error",
            "backends": selected,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "pipeline": spec.get("name", manifest_path.stem),
        "compute_units": compute_units,
        "compute_precision": compute_precision,
        "stages": stage_records,
        "end_to_end": pipeline_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON ONNX pipeline manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compute-units",
        choices=["ALL", "CPU_AND_GPU", "CPU_AND_NE"],
        default="CPU_AND_NE",
    )
    parser.add_argument(
        "--compute-precision",
        choices=["DEFAULT", "FLOAT16", "FLOAT32"],
        default="DEFAULT",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    result = run_pipeline(
        args.manifest,
        args.output,
        args.compute_units,
        args.compute_precision,
        args.warmup,
        args.repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
