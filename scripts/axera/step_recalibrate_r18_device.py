"""Device check: ``step_recalibrate`` on the ResNet18 dense-head step.

This recalibrates ``r18_A1`` to ``r18_Bx2`` and compares the result against
the native ``r18_Bx2`` build on the AX8850. Both are #1836's Pulsar2 builds
of ``t8-r18head2``. They are not committed, because ``npu_params`` alone is
11 MB. The runs go in this order: native A1 (control), native Bx2,
recalibrated, then native A1 again (health). Each run holds the device lock
for that run only. See ``docs/axera-step-recalibrate-relayout.md``.

Usage::

    step_recalibrate_r18_device.py BUILDS_DIR OUT.json
    # BUILDS_DIR holds r18_A1/out/compiled.axmodel, r18_Bx2/out/compiled.axmodel
"""

from __future__ import annotations

import json
import os
import sys

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from short_unit_codec_device_check import _compare, inputs_for, run  # noqa: E402
from step_recalibrate import recalibrate  # noqa: E402


def main(builds: str, out_path: str) -> None:
    a = onnx.load(os.path.join(builds, "r18_A1", "out", "compiled.axmodel"))
    b = onnx.load(os.path.join(builds, "r18_Bx2", "out", "compiled.axmodel"))
    new, report = recalibrate(a, b)
    feeds = inputs_for(a)
    res: dict = {
        "report": {
            **report,
            "records": {str(k): v for k, v in report["records"].items()},
        },
        "inputs": {k: list(v.shape) for k, v in feeds.items()},
    }

    def save():
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)

    ctl = run(a, feeds)
    res["control_error"] = ctl["error"]
    save()
    if ctl["outputs"] is None:
        raise RuntimeError(f"control run failed: {ctl['error']}")
    nb = run(b, feeds)
    res["native_a_vs_native_b"] = {
        **_compare(ctl["outputs"], nb["outputs"]),
        "error": nb["error"],
    }
    save()
    got = run(new, feeds)
    res["recalibrated_vs_native_b"] = {
        **_compare(nb["outputs"], got["outputs"]),
        "error": got["error"],
    }
    save()
    res["health_after"] = run(a, feeds)["outputs"] == ctl["outputs"]
    save()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
