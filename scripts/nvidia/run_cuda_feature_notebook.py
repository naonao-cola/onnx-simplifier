"""Run examples/cuda_feature_tests/cuda_feature_tests.ipynb's tests as a plain script.

The notebook is meant to be run by hand on a GPU (Colab) and never runs in CI. This runner
executes its code cells in one shared namespace without Jupyter, so the same tests can be
rerun on any GPU box -- e.g. a Jetson -- from a terminal:

    python run_cuda_feature_notebook.py [NOTEBOOK.ipynb] [--no-dlpack-check]

- The install cell (the one defining ``BUILD_FROM_SOURCE``) is skipped, as are ``!shell``
  lines and ``%magic`` lines: install onnxsim + a *GPU* onnxruntime (+ a CUDA torch for
  Test F) yourself first.
- Unlike the notebook, a failing cell does not stop the run: every remaining cell still
  executes so one report lists every PASS/FAIL. Exit status is non-zero if any test failed.
- After the notebook cells it runs one extra check the notebook lacks: whether
  ``as_ort_value`` really *aliases* a CUDA torch tensor's memory (same ``data_ptr``, and an
  in-place edit of the tensor is visible through the OrtValue), i.e. DLPack zero-copy.
- ``onnxsim`` (Test D, the CLI) is looked up next to this interpreter, so run it with the
  venv's own ``python``.

On JetPack 6 (CUDA 12.6) PyPI has no usable GPU onnxruntime for Python >= 3.11 (onnxruntime-gpu
>= 1.30 wheels need CUDA 13; older ones have no aarch64 wheel) -- see the recipe in the docs.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

DEFAULT_NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "cuda_feature_tests"
    / "cuda_feature_tests.ipynb"
)


def code_cells(path):
    nb = json.loads(Path(path).read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "BUILD_FROM_SOURCE" in src:  # the Colab install cell
            print(f"[skip] cell {i}: install cell")
            continue
        lines = [
            ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%"))
        ]
        yield i, "\n".join(lines)


def dlpack_aliasing_check(results):
    """Extra check: does ``as_ort_value`` alias a CUDA torch tensor's memory?"""
    import numpy as np
    import torch

    from onnxsim.backend import Runner, as_ort_value

    name = "DLPack aliasing (extra: ORT value shares the tensor's memory)"
    try:
        import onnx
        from onnx import parser

        model = parser.parse_model(
            '<ir_version: 10, opset_import: ["": 18]> g (float[2,2] x) => (float[2,2] y)'
            " { y = Identity(x) }"
        )
        onnx.checker.check_model(model)
        x = torch.arange(4, dtype=torch.float32).reshape(2, 2).cuda()
        ort_value = as_ort_value(x)
        same_ptr = ort_value.data_ptr() == x.data_ptr()
        runner = Runner(model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        before = runner.run_with_ort_values({"x": ort_value})["y"].numpy().copy()
        x.add_(10.0)  # in-place: only visible through the OrtValue if it aliases
        torch.cuda.synchronize()
        after = runner.run_with_ort_values({"x": ort_value})["y"].numpy()
        sees_edit = bool(np.allclose(after, before + 10.0))
        ok = same_ptr and sees_edit
        detail = f"same data_ptr={same_ptr}, in-place edit visible={sees_edit}"
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {e}"
    print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
    results[name] = ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("notebook", nargs="?", default=str(DEFAULT_NOTEBOOK))
    ap.add_argument("--no-dlpack-check", action="store_true")
    args = ap.parse_args(argv)

    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
    ns = {"__name__": "__main__"}
    failed_cells = []
    for i, src in code_cells(args.notebook):
        try:
            exec(compile(src, f"<notebook cell {i}>", "exec"), ns)  # noqa: S102
        except BaseException as e:  # noqa: BLE001 -- keep going, report everything
            failed_cells.append((i, e))
            print(f"[ERROR] cell {i}: {type(e).__name__}: {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc(limit=4)

    results = dict(ns.get("RESULTS", {}))
    if not args.no_dlpack_check:
        dlpack_aliasing_check(results)

    print("=" * 60)
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL':4}  {name}")
    for i, e in failed_cells:
        print(f"ERR   cell {i}: {type(e).__name__}: {str(e)[:100]}")
    bad = [n for n, ok in results.items() if not ok] or failed_cells
    print(f"\n{'FAILED' if bad else 'All passed'}: {len(results)} tests, "
          f"{len(failed_cells)} cell errors")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
