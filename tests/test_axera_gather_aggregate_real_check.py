"""Regression coverage for the real-scale Gather-into-aggregation check.

Does not require Docker or a device: it checks the committed compiled
fixtures' structure and `patch_indices_bytesafe`'s byte-level correctness.
The device numbers (retargeting clips against a narrow-calibration reference,
and a wide-calibration reference fixes it) are recorded in
`docs/axera-gather-aggregate-real.md` and are not re-asserted here.
"""

import gzip
import os
import sys

import numpy as np
import onnx
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

from gather_aggregate_real_check import (  # noqa: E402
    leading_indices,
    patch_indices_bytesafe,
)

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "gather_aggregate_real")


def _unzip(tmp_path, name):
    path = tmp_path / name.removesuffix(".gz")
    with gzip.open(os.path.join(_FIXTURES, name), "rb") as source:
        path.write_bytes(source.read())
    return str(path)


def _npu_params_len(path):
    model = onnx.load(path, load_external_data=False)
    return len(
        next(i.raw_data for i in model.graph.initializer if i.name == "npu_params")
    )


def test_fixture_table_is_not_word_aligned():
    # the real composed reference's npu_params is 10861 bytes -- one byte over
    # a whole number of uint32 words, which is why the byte-safe patcher exists.
    path = os.path.join(_FIXTURES, "a_reference_narrow.axmodel.gz")
    with gzip.open(path, "rb") as f:
        model = onnx.load_model_from_string(f.read())
    length = len(
        next(i.raw_data for i in model.graph.initializer if i.name == "npu_params")
    )
    assert length == 10861
    assert length % 4 != 0


def test_reference_and_native_hold_their_own_indices():
    real_idx = tuple(np.load(os.path.join(_FIXTURES, "real_idx.npy")).tolist())
    adv_idx = tuple(np.load(os.path.join(_FIXTURES, "adv_idx.npy")).tolist())
    assert len(real_idx) == len(adv_idx) == 441

    ref = os.path.join(_FIXTURES, "a_reference_narrow.axmodel.gz")
    native = os.path.join(_FIXTURES, "b_native_adversarial.axmodel.gz")
    with gzip.open(ref, "rb") as f:
        onnx.load_model_from_string(f.read()).SerializeToString()  # loads cleanly
    for gz, expected in ((ref, real_idx), (native, adv_idx)):
        with gzip.open(gz, "rb") as f:
            data = f.read()
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".axmodel") as tmp:
            tmp.write(data)
            tmp.flush()
            assert leading_indices(tmp.name, 441) == expected


def test_patch_indices_bytesafe_preserves_length_and_tail(tmp_path):
    reference = _unzip(tmp_path, "a_reference_narrow.axmodel.gz")
    before_len = _npu_params_len(reference)
    before_model = onnx.load(reference, load_external_data=False)
    before_table = next(
        i.raw_data for i in before_model.graph.initializer if i.name == "npu_params"
    )
    adv_idx = np.load(os.path.join(_FIXTURES, "adv_idx.npy")).tolist()

    out = str(tmp_path / "emitted.axmodel")
    patch_indices_bytesafe(reference, out, adv_idx)

    after_len = _npu_params_len(out)
    assert after_len == before_len == 10861
    assert leading_indices(out, 441) == tuple(adv_idx)
    # everything after the indices is untouched, including the odd trailing byte
    after_model = onnx.load(out, load_external_data=False)
    after_table = next(
        i.raw_data for i in after_model.graph.initializer if i.name == "npu_params"
    )
    assert bytes(before_table)[441 * 4 :] == bytes(after_table)[441 * 4 :]


def test_patch_indices_bytesafe_rejects_too_many_indices(tmp_path):
    reference = _unzip(tmp_path, "a_reference_narrow.axmodel.gz")
    with pytest.raises(ValueError):
        patch_indices_bytesafe(
            reference, str(tmp_path / "bad.axmodel"), list(range(100000))
        )
