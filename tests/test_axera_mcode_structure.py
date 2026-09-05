"""Real-hardware structural analysis of the compiled `.axmodel` blobs this
project's own research identified as Axera's real "Wbt" (Weight Table --
the `npu_params` initializer) and "mcode" (the compiled command-queue
program -- the `<neu_key>`-named initializer) terms. See
scripts/axera/README.md's "Applying the new vocabulary to a real
`.axmodel`, and a real mcode-size finding" section for the full narrative
(a real 1-through-10-identical-Conv-layer sweep) this file locks a smaller
slice of in as a regression test.

Needs a loaded `pulsar2:*` Docker image -- skip-guarded like
tests/test_pulsar2_hf_to_axmodel.py.
"""

import json
import os
import sys

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import pulsar2_docker  # noqa: E402

pytestmark = pytest.mark.skipif(
    not pulsar2_docker.docker_image_available(),
    reason=f"pulsar2 Docker image not loaded: {pulsar2_docker.DEFAULT_IMAGE}",
)


def _n_conv_model(n):
    """`n` sequential, identically-shaped Conv layers -- same spatial size
    throughout (`pads=[1,1,1,1]`) so every layer is a truly identical unit,
    isolating per-op growth from shape-dependent effects."""
    rng = np.random.RandomState(0)
    nodes = []
    inits = []
    prev = "x"
    for i in range(n):
        w = (rng.randn(4, 4, 3, 3) * 0.1).astype(np.float32)
        wname = f"w{i}"
        inits.append(numpy_helper.from_array(w, name=wname))
        out = f"y{i}" if i < n - 1 else "y"
        nodes.append(helper.make_node("Conv", [prev, wname], [out], pads=[1, 1, 1, 1]))
        prev = out
    graph = helper.make_graph(
        nodes,
        f"g_{n}conv",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4, 16, 16])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4, 16, 16])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def _build_and_get_blob_sizes(tmp_path, n):
    work_dir = os.path.join(str(tmp_path), f"conv{n}")
    os.makedirs(work_dir, exist_ok=True)
    onnx.save(_n_conv_model(n), os.path.join(work_dir, "model.onnx"))

    rng = np.random.RandomState(0)
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    samples = [rng.randn(1, 4, 16, 16).astype(np.float32) for _ in range(4)]
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work_dir, "dataset", "calib_x.tar"), samples
    )
    cfg = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": [
                {
                    "tensor_name": "x",
                    "calibration_dataset": "./dataset/calib_x.tar",
                    "calibration_format": "Numpy",
                    "calibration_size": 4,
                }
            ],
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "compiler": {"check": 0},
    }
    os.makedirs(os.path.join(work_dir, "config"), exist_ok=True)
    with open(os.path.join(work_dir, "config", "cfg.json"), "w") as f:
        json.dump(cfg, f)

    result = pulsar2_docker.build(
        work_dir, "model.onnx", "output", config_path="config/cfg.json"
    )
    assert result.success, result.error

    compiled = onnx.load(result.axmodel_path)
    inits = {i.name: i for i in compiled.graph.initializer}
    neu_node = next(nd for nd in compiled.graph.node if nd.op_type == "neu mode")
    info = None
    for attr in neu_node.attribute:
        if attr.name == "npu_graph_info":
            info = json.loads(attr.s.decode())
    dotneu = info["dotneus"][0]
    wbt_key = dotneu["extra_inputs"][0]["const_data_key"]
    mcode_key = dotneu["neu_key"]
    return len(inits[wbt_key].raw_data), len(inits[mcode_key].raw_data)


def test_wbt_and_mcode_scale_differently_with_identical_ops(tmp_path):
    """Confirmed real (see the README's full 1-10 layer sweep): Axera's
    "Wbt" (Weight Table, the `npu_params` blob) grows by an *exact* constant
    number of bytes per added identical Conv layer -- a flat,
    one-record-per-op concatenation, no compression. Axera's "mcode" (the
    `<neu_key>` compiled command-queue blob) does NOT scale linearly, but
    every observed size delta is an exact multiple of 32 bytes -- consistent
    with a 32-byte-aligned command-queue allocation unit where a variable
    (not fixed) number of units get assigned per op instance.
    """
    sizes = {n: _build_and_get_blob_sizes(tmp_path, n) for n in (2, 3, 4)}
    wbt = {n: s[0] for n, s in sizes.items()}
    mcode = {n: s[1] for n, s in sizes.items()}

    wbt_delta_1 = wbt[3] - wbt[2]
    wbt_delta_2 = wbt[4] - wbt[3]
    assert wbt_delta_1 == wbt_delta_2 > 0, (wbt, "Wbt delta should be constant")

    mcode_delta_1 = mcode[3] - mcode[2]
    mcode_delta_2 = mcode[4] - mcode[3]
    assert mcode_delta_1 % 32 == 0, (mcode, "mcode delta should be a multiple of 32")
    assert mcode_delta_2 % 32 == 0, (mcode, "mcode delta should be a multiple of 32")
