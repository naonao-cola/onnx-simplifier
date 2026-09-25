"""Offline tests for scripts/axera/make_training_calib.py.

No Docker, no card: every assertion inspects the generated work directory
(tars + config) for a tiny synthetic step-like graph.
"""

import io
import json
import os
import sys
import tarfile

import numpy as np
import onnx
import onnx.parser

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import make_training_calib  # noqa: E402


def _model():
    # Opset 11 so the reduction takes axes as an (absent, reduce-all)
    # attribute instead of an initializer input -- this test is about the
    # calibration directory layout, not the opset.
    return onnx.parser.parse_model(
        """
        <ir_version: 8, opset_import: ["" : 11]>
        agraph (float[4, 3] x, float[4, 2] y, float[3, 2] w, float[1] lr) => (float[1, 1] loss) {
            xm = MatMul(x, w)
            d = Sub(xm, y)
            sq = Mul(d, d)
            m = ReduceMean<keepdims = 1>(sq)
            loss = Mul(m, lr)
        }
        """
    )


def _save(model_path):
    onnx.save(_model(), model_path)
    return model_path


def _read_tar(work_dir, rel):
    out = []
    with tarfile.open(os.path.join(work_dir, rel)) as tf:
        for m in sorted(tf.getnames()):
            f = tf.extractfile(m)
            out.append(np.load(io.BytesIO(f.read())))
    return out


def test_layer_configs_passthrough_and_per_input_tars(tmp_path):
    wd = make_training_calib.make_work_dir(
        _save(str(tmp_path / "step.onnx")),
        str(tmp_path / "wd"),
        n=4,
        label_inputs=("y",),
        layer_configs=[{"op_types": ["Sub"], "data_type": "FP32"}],
    )
    cfg = json.load(open(os.path.join(wd, "config", "step.json")))
    assert cfg["quant"]["layer_configs"] == [{"op_types": ["Sub"], "data_type": "FP32"}]
    assert make_training_calib.DISTILL_FP32_LAYER_CONFIGS == [
        {
            "op_types": ["Mul", "Add", "Sub", "Div", "Sqrt", "MatMul"],
            "data_type": "FP32",
        }
    ]
    model = onnx.load(os.path.join(wd, "step.onnx"))
    assert {i.name for i in model.graph.input} == {"x", "y", "w", "lr"}
    expected = {"x": (4, 3), "y": (4, 2), "w": (3, 2), "lr": (1,)}
    for name, shape in expected.items():
        samples = _read_tar(wd, f"dataset/{name}.tar")
        assert len(samples) == 4
        for s in samples:
            assert tuple(s.shape) == shape, (name, s.shape)


def test_labels_onehot_per_row_and_lr_jitter(tmp_path):
    wd = make_training_calib.make_work_dir(
        _save(str(tmp_path / "step.onnx")),
        str(tmp_path / "wd"),
        n=4,
        label_inputs=("y",),
    )
    for s in _read_tar(wd, "dataset/y.tar"):
        assert s.shape == (4, 2)
        assert (s.sum(axis=1) == 1.0).all()
        assert ((s == 0.0) | (s == 1.0)).all()
    lrs = _read_tar(wd, "dataset/lr.tar")
    assert len({float(v.reshape(-1)[0]) for v in lrs}) == 4  # no zero-width pin


def test_trajectory_list_cycles(tmp_path):
    traj = [np.full((3, 2), fill_value=float(i), dtype=np.float32) for i in (7.0, 9.0)]
    wd = make_training_calib.make_work_dir(
        _save(str(tmp_path / "step.onnx")),
        str(tmp_path / "wd"),
        n=4,
        label_inputs=("y",),
        real_data={"w": traj},
    )
    got = [float(s.flat[0]) for s in _read_tar(wd, "dataset/w.tar")]
    assert got == [7.0, 9.0, 7.0, 9.0]


def test_scalar_trajectory_is_lifted_to_rank_one_input(tmp_path):
    wd = make_training_calib.make_work_dir(
        _save(str(tmp_path / "step.onnx")),
        str(tmp_path / "wd"),
        n=3,
        real_data={"lr": [0.1, 0.2]},
    )
    got = _read_tar(wd, "dataset/lr.tar")
    assert [tuple(sample.shape) for sample in got] == [(1,)] * 3
    np.testing.assert_allclose(
        [float(sample[0]) for sample in got], [0.1, 0.2, 0.1]
    )
