#!/usr/bin/env python3
"""Validate on-device training (step-graph loops) on AMD ROCm hardware.

Exercises :func:`onnxsim.compile_training_loop` -- the same compiled step
graph (forward + ``graph_grad`` backward + Adam, all inside
``qat_graph.EP_FRIENDLY_OPS``) that block-wise QAT, block finetune and the
adaround/adaquant/autoround ``step_providers=`` loops all run -- on each GPU
execution provider this host offers (``ROCMExecutionProvider``,
``MIGraphXExecutionProvider``), with ``CPUExecutionProvider`` last as the
fallback for ops the accelerator cannot run.

For each provider the script reports:

- whether a session can be built at all (the probe -- presence in
  ``get_available_providers()`` alone is not proof for MIGraphX, whose wheel
  bundles the provider library with or without a ROCm device answering),
- the loss before/after a short training run (must decrease substantially),
- where the trained-parameter/optimizer state lives between steps
  (device-resident ``OrtValue`` vs. host round-trip),
- whether ``Runner.bind_loop`` (the ``IOBinding`` path ``run_step_graph``
  uses) binds on that provider.

A provider that cannot build a session is ``skipped``, never failed. A
training run whose loss does not fall is ``failed``. With no GPU provider at
all the run passes (nothing to test) unless ``--require-gpu`` is given.

On Strix Halo (gfx1151) the prebuilt ``onnxruntime-rocm`` wheel carries no
native kernels, so export ``HSA_OVERRIDE_GFX_VERSION=11.0.0`` to run what the
spoofed gfx1100 target supports; rocBLAS-backed ops (``Transpose``,
``MatMul``) still fail there, which reads as ``failed`` with a HIPBLAS
error -- that is the wheel's coverage, not the step graph. See
``scripts/amd/README.md`` ("Hardware notes").

Usage:
    python scripts/amd/run_training_compat.py
    python scripts/amd/run_training_compat.py --require-gpu   # fail without a GPU EP
    python scripts/amd/run_training_compat.py --steps 200 --lr 0.05
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import onnxsim
    from onnxsim import backend
except ImportError:
    print(
        "SKIP: onnxsim is not installed; run `pip install .` from the "
        "onnxsim checkout first, then re-run this script"
    )
    sys.exit(1 if "--require-gpu" in sys.argv else 0)

try:
    import onnxruntime as ort
except ImportError:
    print("SKIP: onnxruntime is not installed; nothing to validate")
    sys.exit(0)

try:
    from onnx import parser

    _HAS_ONNX_PARSER = True
except ImportError:
    _HAS_ONNX_PARSER = False

_HEADER = '<ir_version: 8, opset_import: ["": 17]>'

# Providers worth probing, in the order to try them. CUDA is included as a
# reference when this script happens to run on an NVIDIA box -- the point of
# the script is the AMD entries.
_CANDIDATES = (
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "CUDAExecutionProvider",
)


def _linear_model(rows: int = 8, k: int = 3, n: int = 2, seed: int = 0):
    """``loss = mean((x @ w^T - y) ** 2)`` with ``w`` trained -- the same
    well-determined regression ``tests/test_compile_training.py`` fits."""
    import onnx
    import onnx.numpy_helper

    rng = np.random.default_rng(seed)
    model = parser.parse_model(
        f"""{_HEADER}
        agraph (float[{rows},{k}] x, float[{rows},{n}] y) => (float loss)
        {{
            wt = Transpose<perm=[1,0]>(w)
            y_hat = MatMul(x, wt)
            diff = Sub(y_hat, y)
            sq = Mul(diff, diff)
            loss = ReduceMean<keepdims=0>(sq)
        }}
        """
    )
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(
            (rng.standard_normal((n, k)) * 0.1).astype(np.float32), "w"
        )
    )
    onnx.checker.check_model(model)
    w_true = rng.standard_normal((n, k)).astype(np.float32)
    x = rng.standard_normal((rows, k)).astype(np.float32)
    y = x @ w_true.T
    return model, x, y


def _probe(provider: str) -> tuple[bool, str]:
    """Whether ``provider`` builds a session on this host, with the reason.

    Presence in ``get_available_providers()`` is not proof (the MIGraphX
    wheel bundles its provider library with or without a ROCm device
    answering), and neither is a successful build alone: onnxruntime
    silently falls back to CPU when a provider's shared library fails to
    load, so the built session must keep the provider in
    ``get_providers()``.
    """
    if provider not in ort.get_available_providers():
        return False, "not in get_available_providers()"
    try:
        probe = parser.parse_model(
            f"""{_HEADER}
            agraph (float[1] x) => (float[1] y)
            {{
                y = Identity(x)
            }}
            """
        )
        sess = ort.InferenceSession(probe.SerializeToString(), providers=[provider])
        kept = sess.get_providers()
        if provider not in kept:
            return False, f"session fell back to {kept}"
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 -- the reason is the report
        return False, f"{type(exc).__name__}: {exc}"


def _check_provider(provider: str, steps: int, lr: float) -> dict:
    """Train the probe model on ``provider``; return the report row."""
    row: dict = {"provider": provider}
    model, x, y = _linear_model()
    try:
        loop = onnxsim.compile_training_loop(
            model, "loss", ("w",), providers=[provider, "CPUExecutionProvider"]
        )
    except Exception as exc:  # noqa: BLE001 -- session creation failed
        row["status"] = "failed"
        row["detail"] = f"session creation: {type(exc).__name__}: {exc}"
        return row
    try:
        losses = [loop({"x": x, "y": y}, lr=lr) for _ in range(steps)]
    except Exception as exc:  # noqa: BLE001 -- the step itself failed
        row["status"] = "failed"
        row["detail"] = f"step {type(exc).__name__}: {exc}"
        return row
    ratio = losses[-1] / losses[0] if losses[0] else float("nan")
    devices = sorted(
        {
            v.device_name() if hasattr(v, "device_name") else "numpy"
            for v in loop._state.values()
        }
    )
    bound = loop.step_graph is not None and _bind_probe(loop) is not None
    row["status"] = "passed" if ratio < 0.2 else "failed"
    row["detail"] = (
        f"loss {losses[0]:.4g} -> {losses[-1]:.4g} (x{ratio:.3g}); "
        f"state on {devices}; iobinding={'yes' if bound else 'no'}"
    )
    return row


def _bind_probe(loop) -> object:
    """Best-effort ``bind_loop`` on the compiled step graph -- report-only."""
    try:
        runner = backend.Runner(
            loop.step_graph.model,
            output_names=list(loop.step_graph.state.values())
            + [loop.step_graph.loss_name],
            providers=loop.providers,
        )
        state = loop.initial_state
        spec = loop.step_graph
        return runner.bind_loop(
            {},
            {k: (v, state[k]) for k, v in spec.state.items()},
        )
    except Exception:  # noqa: BLE001 -- binding is a pure optimization
        return None


def main() -> int:
    argp = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argp.add_argument("--steps", type=int, default=200)
    argp.add_argument("--lr", type=float, default=5e-2)
    argp.add_argument(
        "--require-gpu",
        action="store_true",
        help="exit non-zero when no GPU provider validates",
    )
    args = argp.parse_args()

    if not _HAS_ONNX_PARSER:
        print("SKIP: installed onnx has no onnx.parser; cannot build the probe model")
        return 0

    print(f"available providers: {ort.get_available_providers()}")
    rows = []
    for provider in _CANDIDATES:
        ok, reason = _probe(provider)
        if not ok:
            rows.append(
                {"provider": provider, "status": "skipped", "detail": reason}
            )
            continue
        rows.append(_check_provider(provider, args.steps, args.lr))

    width = max(len(r["provider"]) for r in rows)
    failed = 0
    for row in rows:
        print(f"{row['provider']:<{width}}  {row['status']:7}  {row['detail']}")
        if row["status"] == "failed":
            failed += 1
    tested = [r for r in rows if r["status"] != "skipped"]
    if not tested:
        print("no GPU provider usable on this host; nothing validated")
        return 1 if args.require_gpu else 0
    print(f"{len(tested) - failed}/{len(tested)} provider(s) validated")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
