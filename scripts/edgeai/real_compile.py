#!/usr/bin/env python3
"""Real TIDL model-import/compile check, via the actual x86 PC toolchain.

Unlike every other module in this directory, this one *is* backed by a real
compiler: TI's own `onnxruntime_tidl` wheel (`TIDLCompilationProvider`) plus
the `tidl_tools` binaries, run in "PC emulation"/compile-only mode -- no
target device needed, per edgeai-tidl-tools' own `docs/model_compilation.md`
("Currently, TIDL model compilation is supported only through Python APIs").

This became possible only after this repository's own network policy
allowed `software-dl.ti.com` and, separately, the host its download links
302-redirect to, `downloads.ti.com` -- both were confirmed 403 earlier in
this project's history (see `scripts/edgeai/README.md`), and this module
exists because a later change to that policy unblocked them. If it is
blocked again, `find_tidl_python()`/`tidl_tools_path()` below return None
and the tests that need them skip cleanly rather than failing.

**Real, not simulated, but with two caveats:**

1. This runs the *compile/import* stage only ("PC emulation"), which is a
   genuine run of TI's own C7x-MMA layer-mapping and subgraph-partitioning
   logic -- not a mock. It does not run inference on real silicon; nothing
   here needs a TDA4x/AM6xA board.
2. `onnxruntime_tidl` ships only as a `cp310`-tagged wheel (confirmed: no
   `cp311`/`cp312` build exists at the same release path), so it needs its
   own Python 3.10 virtualenv with `numpy<2` (the pinned wheel predates
   NumPy 2's ABI) -- separate from whatever interpreter onnxsim's own tests
   run under. `find_tidl_python()` locates that venv's interpreter (via the
   `TIDL_PYTHON` env var) and every actual compile happens in a subprocess
   under it, mirroring the isolation `scripts/axera/worker.py` already uses
   for a different reason (crash containment there; interpreter/ABI
   isolation here).

See `README.md`'s "Running a real compile" section for how to set up that
venv and the `tidl_tools` directory this module also needs
(`TIDL_TOOLS_PATH`).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import Dict, Optional

# The setup.sh release tag this module was developed and verified against
# (scripts/setup/setup.sh's REL= at the time). TI does not publish a stable
# "latest" alias for these paths, so a version bump upstream means updating
# this constant, not a code change.
DEFAULT_RELEASE = "11_02_20_00"

# setup.sh's SOC name normalization (scripts/setup/setup_env.sh) collapses
# the AM68A/TDA4VL marketing name to this internal one; this is the SoC this
# module was verified against (docs/model_compilation.md's compile stage
# needs no device, so any SoC in the supported-devices table would do).
DEFAULT_SOC = "J721S2"

_TIDL_TOOLS_URL = (
    "https://software-dl.ti.com/jacinto7/esd/tidl-tools/{release}/TIDL_TOOLS/"
    "{soc}/tidl_tools.tar.gz"
)
_ONNXRUNTIME_TIDL_URL = (
    "https://software-dl.ti.com/jacinto7/esd/tidl-tools/{release}/OSRT_TOOLS/"
    "X86_64_LINUX/UBUNTU_22_04/onnxruntime_tidl-1.23.0-cp310-cp310-linux_x86_64.whl"
)


def download_tidl_tools(
    dest_dir: str, release: str = DEFAULT_RELEASE, soc: str = DEFAULT_SOC
) -> str:
    """Download and extract `tidl_tools` for `soc` into `dest_dir`.

    Returns the path to the extracted `tidl_tools` directory (what
    `TIDL_TOOLS_PATH` should point at). Requires network access to
    `software-dl.ti.com` and the `downloads.ti.com` host it redirects to.
    """
    import urllib.request

    os.makedirs(dest_dir, exist_ok=True)
    archive_path = os.path.join(dest_dir, "tidl_tools.tar.gz")
    url = _TIDL_TOOLS_URL.format(release=release, soc=soc)
    urllib.request.urlretrieve(url, archive_path)
    shutil.unpack_archive(archive_path, dest_dir)
    os.remove(archive_path)
    return os.path.join(dest_dir, "tidl_tools")


def download_onnxruntime_tidl_wheel(
    dest_dir: str, release: str = DEFAULT_RELEASE
) -> str:
    """Download the `cp310`-only `onnxruntime_tidl` wheel into `dest_dir`.

    Returns the downloaded `.whl` path. See this module's docstring for why
    it must be installed into a Python 3.10 virtualenv, not the interpreter
    running this function.
    """
    import urllib.request

    os.makedirs(dest_dir, exist_ok=True)
    url = _ONNXRUNTIME_TIDL_URL.format(release=release)
    dest_path = os.path.join(dest_dir, os.path.basename(url))
    urllib.request.urlretrieve(url, dest_path)
    return dest_path


def find_tidl_python() -> Optional[str]:
    """The `python` executable of a venv with a working `onnxruntime_tidl`.

    Reads the `TIDL_PYTHON` env var (expected to point at such a venv's
    interpreter, e.g. `/path/to/venv/bin/python`) and verifies it actually
    has `TIDLCompilationProvider` available -- returns None (rather than
    raising) on any failure, so callers can use this directly as a
    pytest skip condition.
    """
    python_exe = os.environ.get("TIDL_PYTHON")
    if not python_exe or not os.path.exists(python_exe):
        return None
    try:
        result = subprocess.run(
            [
                python_exe,
                "-c",
                "import onnxruntime as ort; print(ort.get_available_providers())",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return None
    if result.returncode != 0 or "TIDLCompilationProvider" not in result.stdout:
        return None
    return python_exe


def tidl_tools_path() -> Optional[str]:
    """The `TIDL_TOOLS_PATH` env var, if it points at an existing directory."""
    path = os.environ.get("TIDL_TOOLS_PATH")
    if path and os.path.isdir(path):
        return path
    return None


# Run in the TIDL_PYTHON subprocess (see `find_tidl_python`'s docstring for
# why this can't run in-process under whatever interpreter pytest uses).
_COMPILE_SCRIPT = r"""
import os, sys, shutil
import numpy as np
import onnxruntime as ort

model_path, artifacts_dir, tidl_tools_path = sys.argv[1], sys.argv[2], sys.argv[3]
if os.path.exists(artifacts_dir):
    shutil.rmtree(artifacts_dir)
os.makedirs(artifacts_dir)

so = ort.SessionOptions()
so.log_severity_level = 3
provider_options = {
    "tidl_tools_path": tidl_tools_path,
    "artifacts_folder": artifacts_dir,
    "tensor_bits": 8,
    "debug_level": 1,
    "advanced_options:calibration_frames": 1,
    "advanced_options:calibration_iterations": 1,
}
sess = ort.InferenceSession(
    model_path,
    sess_options=so,
    providers=["TIDLCompilationProvider", "CPUExecutionProvider"],
    provider_options=[provider_options, {}],
)
inp = sess.get_inputs()[0]
x = np.random.RandomState(0).rand(*inp.shape).astype(np.float32)
sess.run(None, {inp.name: x})
"""

_SUMMARY_ROW_RE = re.compile(r"\|\s*(C7x|CPU)\s*\|\s*(\d+)\s*\|\s*([\dx]+)\s*\|")


def compile_offload_summary(
    python_exe: str,
    model_path: str,
    tidl_tools_path: str,
    artifacts_dir: str,
    timeout: int = 300,
) -> Dict[str, object]:
    """Compile `model_path` for TIDL and parse the offload summary it prints.

    Returns
        ``{"c7x_nodes": int, "c7x_subgraphs": int, "cpu_nodes": int,
        "returncode": int, "stdout": str, "stderr": str}``.
    ``c7x_nodes``/``c7x_subgraphs`` are 0 if nothing was offloaded (or the
    compile failed -- check ``returncode``). Extra environment beyond
    ``TIDL_TOOLS_PATH`` (notably ``LD_LIBRARY_PATH`` including that same
    directory, which the native `.so`\\ s need to resolve their own
    dependencies) is set here rather than left to the caller.
    """
    env = dict(os.environ)
    env["TIDL_TOOLS_PATH"] = tidl_tools_path
    env["LD_LIBRARY_PATH"] = (
        tidl_tools_path + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    )

    proc = subprocess.run(
        [python_exe, "-c", _COMPILE_SCRIPT, model_path, artifacts_dir, tidl_tools_path],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    summary = {
        "c7x_nodes": 0,
        "c7x_subgraphs": 0,
        "cpu_nodes": 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    for match in _SUMMARY_ROW_RE.finditer(proc.stdout):
        core, nodes, subgraphs = match.group(1), int(match.group(2)), match.group(3)
        if core == "C7x":
            summary["c7x_nodes"] = nodes
            summary["c7x_subgraphs"] = int(subgraphs) if subgraphs != "x" else 0
        else:
            summary["cpu_nodes"] = nodes
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    setup_ap = sub.add_parser(
        "setup", help="download tidl_tools + the onnxruntime_tidl wheel"
    )
    setup_ap.add_argument("dest_dir")
    setup_ap.add_argument("--release", default=DEFAULT_RELEASE)
    setup_ap.add_argument("--soc", default=DEFAULT_SOC)

    compile_ap = sub.add_parser(
        "compile", help="compile one model and print the offload summary"
    )
    compile_ap.add_argument("model_path")
    compile_ap.add_argument("artifacts_dir")
    compile_ap.add_argument("--python", default=sys.executable)
    compile_ap.add_argument("--tidl-tools-path", default=tidl_tools_path())

    args = ap.parse_args()
    if args.cmd == "setup":
        tools_dir = download_tidl_tools(args.dest_dir, args.release, args.soc)
        wheel_path = download_onnxruntime_tidl_wheel(args.dest_dir, args.release)
        print(f"tidl_tools: {tools_dir}")
        print(f"onnxruntime_tidl wheel: {wheel_path}")
    elif args.cmd == "compile":
        if not args.tidl_tools_path:
            raise SystemExit("--tidl-tools-path (or TIDL_TOOLS_PATH) is required")
        result = compile_offload_summary(
            args.python, args.model_path, args.tidl_tools_path, args.artifacts_dir
        )
        print(result)
