#!/usr/bin/env python3
"""Emits ``webgpu_hf_demo_adversarial_x.json``: one 64-float input vector
that reproduces, deterministically and offline, the exact failure mode a
live CI run of ``webgpu_hf_demo.test.mjs`` hit (see that file's own comment
on its "loss decreased meaningfully" check, and
``webgpu_hf_demo_loss_check.test.mjs``, which replays this vector).

**The bug this documents:** ``step_qat_hf_demo.onnx``'s baked initial
weights (``w1``/``b1``/``w2``/``b2`` in ``step_qat_hf_demo.json``, generated
by ``build_qat_hf_demo``) define one fixed function of the input. For *some*
inputs -- entirely plausible among real, randomly-sampled photos, since
``fetchSampleImageBytes()`` fetches a fresh unseeded one every CI run -- that
function's untrained output already lands extremely close to the target
(0), making ``losses[0]`` a tiny, near-coincidental outlier. Adam's
bias-corrected first update is roughly ``lr * sign(gradient)`` *regardless*
of how small the gradient already is (see ``build_qat_hf_demo``'s own
comment), so training then necessarily jumps the loss up by orders of
magnitude in *relative* terms on step 1, even though it proceeds completely
normally afterward -- which is exactly why the "loss decreased meaningfully"
check now compares windowed averages instead of raw ``losses[0]``/
``losses[-1]`` endpoints.

**How this vector was found:** a plain random search over 200,000 vectors
uniform in ``[-1, 1]^64`` (``normalizePixels``' own output range for
grayscale input, the range any real photo actually arrives in), keeping the
one whose forward pass through the baked weights lands closest to zero. Not
an adversarial construction against the network internals -- literally
"try a lot of plausible photos and see how close one gets by chance" -- which
is the point: this really can happen to an ordinary real photo, not just a
hand-crafted worst case.

Regenerate (only ``numpy``, ``onnx``, and ``onnxruntime`` are needed -- reads
the already-committed ``step_qat_hf_demo.json`` for the baked weights rather
than rebuilding the graph, so no onnxsim import at all)::

    python3 make_webgpu_hf_demo_regression_fixture.py
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SEED = 1
NUM_CANDIDATES = 200_000


def _load_state():
    manifest = json.loads((HERE / "step_qat_hf_demo.json").read_text())
    state = manifest["state"]

    def tensor(name):
        spec = state[name]
        return np.asarray(spec["data"], dtype=np.float64).reshape(spec["dims"])

    return manifest, tensor("w1"), tensor("b1"), tensor("w2"), tensor("b2")


def _forward(x, w1, b1, w2, b2):
    h1 = x @ w1 + b1
    h2 = 1.0 / (1.0 + np.exp(-h1))
    return h2 @ w2 + b2


def main():
    manifest, w1, b1, w2, b2 = _load_state()
    in_dim = manifest["inputDim"]

    rng = np.random.default_rng(SEED)
    candidates = rng.uniform(-1.0, 1.0, size=(NUM_CANDIDATES, in_dim))
    outputs = np.array([_forward(x, w1, b1, w2, b2)[0] for x in candidates])
    best = int(np.argmin(np.abs(outputs)))
    x, y = candidates[best], outputs[best]

    fixture = {
        "note": (
            "One input vector whose forward pass through step_qat_hf_demo's "
            "baked initial weights lands within float32 precision of the "
            "target -- see this script's own docstring for why that "
            "reproduces webgpu_hf_demo.test.mjs's intermittent CI failure. "
            "Regenerate with make_webgpu_hf_demo_regression_fixture.py."
        ),
        "seed": SEED,
        "numCandidates": NUM_CANDIDATES,
        "initialOutput": float(y),
        "x": [float(v) for v in x],
    }
    out_path = HERE / "webgpu_hf_demo_adversarial_x.json"
    out_path.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {out_path.name}; initial forward output y={y:.6g} (loss={y**2:.6g})")


if __name__ == "__main__":
    main()
