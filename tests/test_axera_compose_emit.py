"""Regression coverage for Gather index retargeting inside composed AX650 graphs."""

import gzip
import json
import os
import struct
import sys

import onnx
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

from compose_emit import _CHAINS, emit_gather_in_graph, measured_chains  # noqa: E402

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures")
_ORACLE_B = {
    "gather_reshape_matmul": "compose_gather_reshape_matmul_oracle_b.axmodel.gz",
    "gather_reshape_matmul_transpose_add": (
        "compose_gather_reshape_matmul_transpose_add_oracle_b.axmodel.gz"
    ),
}
_INDICES_A = [0, 2, 4, 6, 8, 10, 12, 14]
_INDICES_B = [15, 3, 9, 0, 0, 7, 12, 1]


def _unzip(tmp_path, name):
    path = tmp_path / name.removesuffix(".gz")
    with gzip.open(os.path.join(_FIXTURES, name), "rb") as source:
        path.write_bytes(source.read())
    return str(path)


def _init(model, name):
    return next(item for item in model.graph.initializer if item.name == name)


def _words(model):
    table = bytes(_init(model, "npu_params").raw_data)
    return struct.unpack(f"<{len(table) // 4}I", table)


def _mcode(model):
    return bytes(
        next(i for i in model.graph.initializer if i.name.endswith("_neu")).raw_data
    )


def _load(path):
    return onnx.load(path, load_external_data=False)


def test_measured_chains_are_the_four_prefixes():
    assert measured_chains() == (
        "gather_reshape",
        "gather_reshape_matmul",
        "gather_reshape_matmul_transpose",
        "gather_reshape_matmul_transpose_add",
    )


@pytest.mark.parametrize("chain", sorted(_CHAINS))
def test_emit_retargets_only_the_index_words(tmp_path, chain):
    fixture = _load(_unzip(tmp_path, _CHAINS[chain][0]))
    target = tmp_path / "out.axmodel"

    emit_gather_in_graph(chain, str(target), indices=_INDICES_B)

    emitted = _load(str(target))
    assert _words(emitted)[:8] == tuple(_INDICES_B)
    assert _words(emitted)[8:] == _words(fixture)[8:]
    assert _mcode(emitted) == _mcode(fixture)
    out_shape = list(_CHAINS[chain][2])
    assert [
        d.dim_value for d in emitted.graph.output[0].type.tensor_type.shape.dim
    ] == out_shape
    attrs = {
        a.name: onnx.helper.get_attribute_value(a)
        for a in emitted.graph.node[0].attribute
    }
    assert json.loads(attrs["outputs_info"]) == {"y": ["FP32", out_shape]}


@pytest.mark.parametrize("chain", sorted(_CHAINS))
def test_index_words_sit_first_and_are_contiguous_in_the_fixture(tmp_path, chain):
    fixture = _load(_unzip(tmp_path, _CHAINS[chain][0]))
    assert _words(fixture)[:8] == tuple(_INDICES_A)


@pytest.mark.parametrize("chain", sorted(_ORACLE_B))
def test_compiler_built_variant_shares_index_words_but_not_the_scale_tail(
    tmp_path, chain
):
    # Pulsar2 calibrates the Gather output from the elements the indices select,
    # so a fresh build of the same graph with other indices has other scale
    # words after the index words. Retargeting the fixture keeps the reference
    # scales, which is why the emitted model is not a rebuild of the target.
    fixture = _load(_unzip(tmp_path, _CHAINS[chain][0]))
    oracle = _load(_unzip(tmp_path, _ORACLE_B[chain]))
    assert _words(oracle)[:8] == tuple(_INDICES_B)
    assert len(_words(oracle)) == len(_words(fixture))
    assert _words(oracle)[8:] != _words(fixture)[8:]
    assert len(_mcode(oracle)) == len(_mcode(fixture))


@pytest.mark.parametrize("chain", sorted(_ORACLE_B))
def test_emit_accepts_a_compiler_built_reference_and_keeps_its_scales(tmp_path, chain):
    oracle_path = _unzip(tmp_path, _ORACLE_B[chain])
    oracle = _load(oracle_path)
    target = tmp_path / "out.axmodel"

    emit_gather_in_graph(
        chain, str(target), indices=_INDICES_A, reference_path=oracle_path
    )

    emitted = _load(str(target))
    assert _words(emitted)[:8] == tuple(_INDICES_A)
    assert _words(emitted)[8:] == _words(oracle)[8:]
    # rebuild noise: the reference's own MCode is kept, not the fixture's
    assert _mcode(emitted) == _mcode(oracle)


def test_mcode_is_rewritten_not_extended_as_the_chain_grows(tmp_path):
    sizes = [
        len(_mcode(_load(_unzip(tmp_path, _CHAINS[c][0])))) for c in measured_chains()
    ]
    assert sizes == [2632, 3456, 3944, 4472]


@pytest.mark.parametrize(
    "indices",
    [
        _INDICES_A[:7],
        _INDICES_A + [1],
        [0, 2, 4, 6, 8, 10, 12, 16],
        [0, 2, 4, 6, 8, 10, 12, -1],
        [0, 2, 4, 6, 8, 10, 12, True],
        "01234567",
    ],
)
def test_emit_rejects_invalid_indices(tmp_path, indices):
    with pytest.raises(ValueError):
        emit_gather_in_graph(
            "gather_reshape_matmul", str(tmp_path / "bad.axmodel"), indices=indices
        )


def test_emit_rejects_unmeasured_chain(tmp_path):
    with pytest.raises(ValueError, match="unmeasured"):
        emit_gather_in_graph(
            "gather_only", str(tmp_path / "bad.axmodel"), indices=_INDICES_A
        )


def test_emit_rejects_reference_of_another_chain(tmp_path):
    other = _unzip(tmp_path, _CHAINS["gather_reshape_matmul_transpose"][0])
    with pytest.raises(ValueError, match="output must have shape"):
        emit_gather_in_graph(
            "gather_reshape_matmul",
            str(tmp_path / "bad.axmodel"),
            indices=_INDICES_A,
            reference_path=other,
        )


def test_emit_rejects_nonzero_padding_in_the_reference(tmp_path):
    reference = _unzip(tmp_path, _CHAINS["gather_reshape_matmul"][0])
    model = _load(reference)
    words = list(_words(model))
    words[-1] = 1
    _init(model, "npu_params").raw_data = struct.pack(f"<{len(words)}I", *words)
    onnx.save(model, reference)
    with pytest.raises(ValueError, match="padding"):
        emit_gather_in_graph(
            "gather_reshape_matmul",
            str(tmp_path / "bad.axmodel"),
            indices=_INDICES_A,
            reference_path=reference,
        )


def test_emit_rejects_reference_with_wrong_table_size(tmp_path):
    reference = _unzip(tmp_path, _CHAINS["gather_reshape_matmul"][0])
    model = _load(reference)
    _init(model, "npu_params").raw_data = bytes(_init(model, "npu_params").raw_data)[
        :-4
    ]
    onnx.save(model, reference)
    with pytest.raises(ValueError, match="npu_params size"):
        emit_gather_in_graph(
            "gather_reshape_matmul",
            str(tmp_path / "bad.axmodel"),
            indices=_INDICES_A,
            reference_path=reference,
        )


def test_emit_rejects_reference_with_missing_live_input(tmp_path):
    reference = _unzip(tmp_path, _CHAINS["gather_reshape_matmul"][0])
    model = _load(reference)
    del model.graph.input[1]
    onnx.save(model, reference)
    with pytest.raises(ValueError, match="inputs"):
        emit_gather_in_graph(
            "gather_reshape_matmul",
            str(tmp_path / "bad.axmodel"),
            indices=_INDICES_A,
            reference_path=reference,
        )
