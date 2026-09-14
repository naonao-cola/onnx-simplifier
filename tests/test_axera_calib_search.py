"""Searching Pulsar2 calibration parameters offline, without Docker or a card.

`scripts/axera/calib_search.py` rebuilds a model over a calibration
method x size grid and ranks the cells by replay SNR. The ranking and table
helpers are pure here; the Docker builds they wrap are not.
"""

import io
import json
import os
import sys
import tarfile

import numpy as np

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import calib_search  # noqa: E402


def test_rank_configs_prefers_mean_then_stability():
    scores = {
        "good": [40.0, 41.0, 40.5],
        "fine-on-average": [45.0, 45.0, 20.0],
        "bad": [20.0, 21.0, 19.0],
    }
    ranked = calib_search.rank_configs(scores)
    assert [k for k, _, _ in ranked] == ["good", "fine-on-average", "bad"]
    # same mean: the one that never collapses wins.
    tied = calib_search.rank_configs({"a": [30.0, 10.0], "b": [20.0, 20.0]})
    assert [k for k, _, _ in tied] == ["b", "a"]


def _fake_table(path, scales):
    quant = os.path.join(path, "quant")
    os.makedirs(quant, exist_ok=True)
    configs, values = {}, {}
    for i, (tensor, scale) in enumerate(scales.items()):
        h = str(1000 + i)
        values[h] = {"scale": [scale], "zero_point": [0.0]}
        configs.setdefault("op", {})[tensor] = {
            "bit_width": 8,
            "policy": {},
            "hash": int(h),
        }
    with open(os.path.join(quant, "quant_axmodel.json"), "w") as f:
        json.dump({"tensor_configs": configs, "values": values}, f)


def test_tables_equal_catches_a_silent_fallback(tmp_path):
    a = str(tmp_path / "a")
    b = str(tmp_path / "b")
    c = str(tmp_path / "c")
    _fake_table(a, {"x": 0.03, "y": 0.01})
    _fake_table(b, {"x": 0.03, "y": 0.01})
    _fake_table(c, {"x": 0.031, "y": 0.01})
    assert calib_search.tables_equal(a, b)
    assert not calib_search.tables_equal(a, c)


def test_write_config_names_method_and_size(tmp_path):
    path = str(tmp_path / "cfg.json")
    calib_search.write_config(path, "x", "dataset/calib.tar", "Percentile", 16)
    cfg = json.load(open(path))
    assert cfg["quant"]["calibration_method"] == "Percentile"
    assert cfg["quant"]["input_configs"][0]["calibration_size"] == 16
    assert cfg["quant"]["input_configs"][0]["tensor_name"] == "x"


def test_make_calibration_tar_holds_n_samples(tmp_path):
    path = str(tmp_path / "calib.tar")
    calib_search.make_calibration_tar(path, (1, 4), 6, seed=0)
    with tarfile.open(path) as tf:
        names = sorted(tf.getnames())
        assert names == [f"{i}.npy" for i in range(6)]
        arr = np.load(io.BytesIO(tf.extractfile("0.npy").read()))
        assert arr.shape == (1, 4) and arr.dtype == np.float32
