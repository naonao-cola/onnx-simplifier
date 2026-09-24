"""Phone-free checks of StreamPETR's HVX cross-attention kernel
(``scripts/android/vision_models/streampetr/attn_hvx``): the C scalar body is bit-exact with the
numpy integer contract (``attn_contract.py``) on synthetic cases, and -- when ``HEXAGON_TOOLS`` points
at a toolchain with hexagon-sim -- so is the HVX body."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ANDROID = Path(__file__).resolve().parents[1] / "scripts" / "android"
ATTN = ANDROID / "vision_models" / "streampetr" / "attn_hvx"


def _contract():
    # a unique module name: nothing generic lands in sys.modules
    spec = importlib.util.spec_from_file_location(
        "streampetr_attn_contract", ATTN / "attn_contract.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cases(tmp_path):
    c = _contract()
    out = []
    for i, (lq, lk, step) in enumerate(
        [(20, 256, 0.06), (9, 384, 0.02), (5, 128, 0.2)]
    ):
        d = tmp_path / f"case{i}"
        c.synth_case(d, lq=lq, lk=lk, seed=i, step=step)
        out.append(str(d))
    return out


def test_contract_matches_float_softmax(tmp_path):
    np = pytest.importorskip("numpy")
    c = _contract()
    rng = np.random.default_rng(0)
    qu = rng.integers(90, 160, (6, 256)).astype(np.uint8)
    ku = rng.integers(60, 200, (128, 256)).astype(np.uint8)
    vu = rng.integers(0, 256, (128, 256)).astype(np.uint8)
    zq, zk, step = 120, 125, 0.05
    out = c.attn_u8(qu, ku, vu, zq, zk, *c.exp_params(step)).astype(np.float64)
    for h in range(8):
        sl = slice(32 * h, 32 * h + 32)
        s = (
            (qu[:, sl].astype(np.int64) - zq)
            @ (ku[:, sl].astype(np.int64) - zk).T
            * step
        )
        p = np.exp(s - s.max(1, keepdims=True))
        ref = (p / p.sum(1, keepdims=True)) @ vu[:, sl]
        assert (
            np.abs(out[:, sl] - ref).max() <= 2.0
        )  # uint8 probabilities + rounding: a couple of LSB


def _plain_env():
    # The sanitizer CI job runs pytest with LD_PRELOAD=libasan/LSan; a host `cc` (and the
    # checker it builds) inheriting that exits non-zero on LeakSanitizer's own reports.
    return {
        k: v
        for k, v in os.environ.items()
        if k not in ("LD_PRELOAD", "LSAN_OPTIONS", "ASAN_OPTIONS")
    }


def test_scalar_body_on_host(tmp_path):
    pytest.importorskip("numpy")
    cc = shutil.which(os.environ.get("CC", "cc"))
    if cc is None:
        pytest.skip("no host C compiler")
    exe = tmp_path / "attn_host_check"
    subprocess.run(
        [cc, "-O2", "-o", str(exe), str(ATTN / "attn_host_check.c")],
        check=True,
        env=_plain_env(),
    )
    out = subprocess.run(
        [str(exe), *_cases(tmp_path)], capture_output=True, text=True, env=_plain_env()
    )
    assert out.returncode == 0 and out.stdout.rstrip().endswith("PASS"), (
        out.stdout + out.stderr
    )


def test_hvx_body_on_hexagon_sim(tmp_path):
    pytest.importorskip("numpy")
    sys.path.insert(0, str(ANDROID))
    try:
        import hexagon_sim_harness as harness
    finally:
        sys.path.remove(str(ANDROID))
    tools = harness.tools_dir()
    if tools is None:
        pytest.skip("HEXAGON_TOOLS does not point at a toolchain with hexagon-sim")
    elf = tmp_path / "attn_sim.elf"
    subprocess.run(
        [
            str(tools / "bin" / "hexagon-clang"),
            "-mv69",
            "-mhvx",
            "-mhvx-length=128B",
            "-O2",
            str(ATTN / "attn_sim.c"),
            "-o",
            str(elf),
            "-lhexagon",
        ],
        check=True,
    )
    env = dict(os.environ)
    harness._ensure_ncurses5(env, tools / "bin" / "hexagon-sim", tmp_path)
    for case in _cases(tmp_path):
        cmd = [
            str(tools / "bin" / "hexagon-sim"),
            "-mv69",
            "--simulated_returnval",
            str(elf),
            "--",
            case,
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
        assert out.returncode == 0 and "\nPASS" in out.stdout, (
            out.stdout[-2000:] + out.stderr[-2000:]
        )
