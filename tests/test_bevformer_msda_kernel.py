"""Phone-free checks of BEVFormer-tiny's fused deformable-sampling kernel
(``scripts/android/vision_models/bevformer_tiny/msda_hvx``).

* ``split.msda_fused`` (the kernel's contract, torch) equals the model's own ``msda_rank5`` math for
  a TSA-shaped call (mean over the 2-frame queue) and an SCA-shaped one (visibility average).
* ``msda_kernel.h``'s plain-C path (``msda_host_check.c``, built with the host C compiler) matches
  ``msda_fused`` on synthetic cases that hit every edge: points outside the map, taps on the zero
  padding, coordinates exactly on pixel centers / borders, invisible (map, query) pairs, a query
  no camera sees.
* The HVX qf32 path (``msda_sim.c``) on ``hexagon-sim``, only when ``HEXAGON_TOOLS`` points at a
  Hexagon toolchain (qemu can't decode HVX float).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ANDROID = Path(__file__).resolve().parents[1] / "scripts" / "android"
MSDA = ANDROID / "vision_models" / "bevformer_tiny" / "msda_hvx"
for _p in (ANDROID, MSDA.parent, MSDA):
    sys.path.insert(0, str(_p))

import model as M  # noqa: E402
import split  # noqa: E402

# Don't leak BEVFormer's generic module names to the rest of the session: other tests import
# their own ``model`` (e.g. tests/test_nanochat.py's ``from model import GPT``), which would
# otherwise get this cached one. ``M``/``split`` keep their references; ANDROID stays on the
# path for the lazy ``hexagon_sim_harness`` import below.
for _name in ("model", "split"):
    sys.modules.pop(_name, None)
for _p in (MSDA.parent, MSDA):
    sys.path.remove(str(_p))


def _case(kind: str, q: int = 96, seed: int = 0):
    """Kernel args for a TSA- (NV=2, 50x50, NO=2, P=4) or SCA-shaped (NV=6, 15x25, NO=1, P=8) call."""
    g = torch.Generator().manual_seed(seed)
    nv, h, w, r, no, p = (2, 50, 50, 1, 2, 4) if kind == "tsa" else (6, 15, 25, 4, 1, 8)
    value = torch.randn(nv, h * w, 256, generator=g)
    ref = (
        torch.rand(nv, q, r, 2, generator=g) * 1.4 - 0.2
    )  # some reference points off the map
    off = torch.randn(q, 8, no, p, 2, generator=g) * 2.0
    # exact pixel centers / borders / half-pixels: the floor fix-up and the zero-padding edges
    ref[:, :4] = 0.0
    off[:4] = torch.tensor([-0.5, 0.0, 0.5, float(w) - 0.5])[:4].reshape(4, 1, 1, 1, 1)
    off[4, :, :, :, 0] = (
        float(w) + 0.5 - ref[0, 4, 0, 0] * w
    )  # x = w: just past the right edge
    attw = torch.softmax(torch.randn(q, 8, no, p, generator=g), -1)
    vis = None
    if kind == "sca":
        vis = (torch.rand(nv, q, generator=g) < 0.3).to(torch.uint8)
        vis[:, 5] = 0  # a query no camera sees -> zeros
        vis[:, 6] = 1  # one every camera sees
    return value, (h, w), ref, off, attw, vis


def test_msda_fused_matches_model_math():
    value, hw, ref, off, attw, vis = _case("tsa")
    q = off.shape[0]
    loc = ref.reshape(2, q, 1, 1, 2) + off.permute(2, 0, 1, 3, 4) / torch.tensor(
        [hw[1], hw[0]], dtype=off.dtype
    )
    ref_tsa = M.msda_rank5(
        value.reshape(2, -1, 8, 32), hw, loc, attw.permute(2, 0, 1, 3)
    ).mean(0)
    torch.testing.assert_close(
        split.msda_fused(value, hw, ref, off, attw), ref_tsa, rtol=1e-5, atol=1e-5
    )

    value, hw, ref, off, attw, vis = _case("sca")
    nv = value.shape[0]
    idx = torch.arange(off.shape[3]) % ref.shape[2]
    loc = ref[:, :, idx][:, :, None] + off[:, :, 0][None] / torch.tensor(
        [hw[1], hw[0]], dtype=off.dtype
    )
    per_cam = M.msda_rank5(
        value.reshape(nv, -1, 8, 32),
        hw,
        loc,
        attw[:, :, 0][None].expand(nv, -1, -1, -1),
    )
    v = vis.to(value.dtype)[..., None]
    ref_sca = (per_cam * v).sum(0) / v.sum(0).clamp(min=1.0)
    torch.testing.assert_close(
        split.msda_fused(value, hw, ref, off, attw, vis), ref_sca, rtol=1e-5, atol=1e-5
    )


def _dump(tmp_path: Path, kind: str) -> Path:
    args = _case(kind)
    split.save_kernel_case(tmp_path, kind, args, split.msda_fused(*args))
    return tmp_path / kind


def test_host_c_kernel(tmp_path):
    cc = shutil.which(os.environ.get("CC", "cc"))
    if cc is None:
        pytest.skip("no host C compiler")
    exe = tmp_path / "msda_host_check"
    subprocess.run(
        [cc, "-O2", "-o", str(exe), str(MSDA / "msda_host_check.c"), "-lm"], check=True
    )
    cases = [str(_dump(tmp_path, k)) for k in ("tsa", "sca")]
    out = subprocess.run([str(exe), *cases], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.rstrip().endswith("PASS"), (
        out.stdout + out.stderr
    )


def test_hvx_kernel_on_hexagon_sim(tmp_path):
    import hexagon_sim_harness as harness

    tools = harness.tools_dir()
    if tools is None:
        pytest.skip("HEXAGON_TOOLS does not point at a toolchain with hexagon-sim")
    elf = tmp_path / "msda_sim.elf"
    subprocess.run(
        [
            str(tools / "bin" / "hexagon-clang"),
            "-mv69",
            "-mhvx",
            "-mhvx-length=128B",
            "-O2",
            str(MSDA / "msda_sim.c"),
            "-o",
            str(elf),
            "-lm",
            "-lhexagon",
        ],
        check=True,
    )
    env = dict(os.environ)
    harness._ensure_ncurses5(env, tools / "bin" / "hexagon-sim", tmp_path)
    for kind in ("tsa", "sca"):
        case = _dump(tmp_path, kind)
        out = subprocess.run(
            [
                str(tools / "bin" / "hexagon-sim"),
                "-mv69",
                "--simulated_returnval",
                str(elf),
                "--",
                str(case),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        assert out.returncode == 0 and "\nPASS" in out.stdout, (
            out.stdout[-2000:] + out.stderr[-2000:]
        )
