"""Phone-free check of the HMX GEMM core (``scripts/android/hmx_gemm/hmx_gemm.h``) on hexagon-sim.

``hexagon-sim -mv69 --mhmx 1`` models the Hexagon matrix unit; ``sim/gemm_sim.c`` runs
``hmx_gemm_f16`` (HVX pack/unpack, chunked K, 256 KB VTCM windows) against a double reference.
Runs only when ``HEXAGON_TOOLS`` points at a Hexagon toolchain with hexagon-sim.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

ANDROID = Path(__file__).resolve().parents[1] / "scripts" / "android"
GEMM = ANDROID / "hmx_gemm"


def _harness():
    # loaded under a unique module name so no generic name leaks into sys.modules / sys.path
    spec = importlib.util.spec_from_file_location(
        "_hmx_gemm_sim_harness", ANDROID / "hexagon_sim_harness.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_sim(tmp_path, src, args):
    harness = _harness()
    tools = harness.tools_dir()
    if tools is None:
        pytest.skip("HEXAGON_TOOLS does not point at a toolchain with hexagon-sim")
    elf = tmp_path / (src.stem + ".elf")
    subprocess.run(
        [
            str(tools / "bin" / "hexagon-clang"),
            "-mv69",
            "-mhmx",
            "-mhvx",
            "-O2",
            str(src),
            "-o",
            str(elf),
            "-lm",
        ],
        check=True,
    )
    env = dict(os.environ)
    harness._ensure_ncurses5(env, tools / "bin" / "hexagon-sim", tmp_path)
    out = subprocess.run(
        [
            str(tools / "bin" / "hexagon-sim"),
            "-mv69",
            "--mhmx",
            "1",
            str(elf),
            "--",
            *args,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    return out.stdout + out.stderr


@pytest.mark.parametrize("shape", ["32 64 64 0", "45 576 128 1", "45 1056 128 1"])
def test_hmx_gemm_f16_on_hexagon_sim(tmp_path, shape):
    out = _run_sim(tmp_path, GEMM / "sim" / "gemm_sim.c", shape.split())
    assert "rc 0" in out and " 0 beyond fp16 rounding" in out, out[-2000:]


@pytest.mark.parametrize("ktiles", ["1", "9"])
def test_hmx_block_bit_exact_on_hexagon_sim(tmp_path, ktiles):
    # fp16 (rne of the exact sum + bias), int8 -> u16 and int8 -> u8 stores, bit-exact; 9 K-tiles also
    # exercise the split of an int8 K into two load pairs (8 + 1)
    out = _run_sim(tmp_path, GEMM / "sim" / "block_ref.c", [ktiles])
    assert "\nPASS" in out, out[-2000:]
