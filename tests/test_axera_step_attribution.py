"""Tests for the whole-step attribution tool (synthetic traces + a real MCode fixture)."""

import gzip
import os
import sys

import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import step_attribution as sa  # noqa: E402
import step_compose_probe as scp  # noqa: E402

PROFILE = {
    "op_a:AxQuantizedConv_1_s0_gop": {"type": "AxQuantizedConv", "cycles": 900.0},
    "op_a:AxQuantizedConv_1_s1_gop": {"type": "AxQuantizedConv", "cycles": 900.0},
    "x_QuantizeLinear_s0_sp": {"type": "AxQuantizeLinear", "cycles": 100.0},
    "Add_5": {"type": "AxAdd", "cycles": 100.0},
    "ld:bias_rd_ocm": {"type": "AxSlice", "cycles": 10.0},
}


def _ev(name, tid, cname="", dur=1.0, inputs="", outputs="", dep_by=()):
    return {
        "name": name,
        "tid": tid,
        "dur": dur,
        "args": {
            "cname": cname,
            "inputs": inputs,
            "outputs": outputs,
            "dep_by": list(dep_by),
        },
    }


@pytest.mark.parametrize(
    "name,expected",
    [
        ("op_a:AxQuantizedConv_1_s0_gop_0_0", "op_a:AxQuantizedConv_1_s0_gop"),
        ("op_a:AxQuantizedConv_1_s1_gop_3_2", "op_a:AxQuantizedConv_1_s1_gop"),
        ("x_QuantizeLinear_s0_sp_2_0", "x_QuantizeLinear_s0_sp"),
        ("Add_5_0_0", "Add_5"),
        ("ld:bias_rd_ocm_0_0", "ld:bias_rd_ocm"),
        ("__rtv_data___0", None),
        ("st:unknown_out[0:1]3_s0_gop[0:1:1]_0_0", None),
    ],
)
def test_resolve_op_strips_tile_suffixes_and_prefixes(name, expected):
    assert sa.resolve_op(name, PROFILE) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("st:op_x:AxClip_out[0:1,:4]16_s3_gop[0:1:1]_0_0", "AxClip"),
        ("st:op_x:pre_transpose_out[1:2]16_s0_gop_0_0", "AxTranspose"),
        ("st:distill__sub_1191[0:1,:8]3_s0_gop_0_0", "AxSub"),
        ("st:plain_thing_0_0", None),
    ],
)
def test_embedded_type_reads_the_op_a_store_belongs_to(name, expected):
    assert sa.embedded_type(name) == expected


def test_event_type_load_takes_the_type_of_its_reader():
    load = _ev(
        "ld:xxh128:abcd_(64,)_U8_0_0",
        "sdma4",
        "ld:param",
        outputs="xxh128:abcd_(64,)_U8_0 - ocm_base:[0:64]",
    )
    reader = _ev(
        "op_a:AxQuantizedConv_1_s0_gop_0_0",
        "conv0",
        inputs="xxh128:abcd_(64,)_U8_0 - ocm_base:[0:64]",
    )
    events = [load, reader]
    readers = sa.build_readers(events)
    assert sa.event_type(load, PROFILE, readers) == "AxQuantizedConv"
    assert sa.event_type(load, PROFILE) == sa.UNATTRIBUTED
    # a runtime-value load is never given a type
    assert (
        sa.event_type(_ev("__rtv_data___0", "sdma4"), PROFILE, readers)
        == sa.UNATTRIBUTED
    )


def test_cycles_by_type_sums_per_type():
    out = sa.cycles_by_type(PROFILE)
    assert out["AxQuantizedConv"] == {"ops": 2, "cycles": 1800.0}
    assert out["AxAdd"]["ops"] == 1


def test_mcode_estimate_conserves_bytes_and_respects_weights():
    segs = [
        {"engine": "conv", "bytes": 100},
        {"engine": "conv", "bytes": 100},
        {"engine": "teng2", "bytes": 300},
        {"engine": "cv3", "bytes": 50},
        {"engine": "sdma4", "bytes": 400},
    ]
    events = [
        _ev("op_a:AxQuantizedConv_1_s0_gop_0_0", "conv0", dur=9.0),
        _ev("op_a:AxQuantizedConv_1_s1_gop_0_0", "conv1", dur=9.0),
        _ev("Add_5_0_0", "teng2", dur=1.0),
        _ev("x_QuantizeLinear_s0_sp_0_0", "teng2", dur=3.0),
        _ev("ld:bias_rd_ocm_0_0", "sdma4", "ld:param", dur=1.0),
        _ev("op_a:AxQuantizedConv_1_s0_gop_1_0", "cv3", dur=1.0),
    ]
    matrix = sa.task_matrix(events, PROFILE)
    by_tasks = sa.mcode_estimate(segs, matrix, "tasks")
    by_dur = sa.mcode_estimate(segs, matrix, "dur")
    total = sum(s["bytes"] for s in segs)
    assert sum(by_tasks.values()) == pytest.approx(total)
    assert sum(by_dur.values()) == pytest.approx(total)
    # teng2's 300 B split 1:1 by task count but 1:3 by duration
    assert by_tasks["AxAdd"] == pytest.approx(150.0)
    assert by_dur["AxAdd"] == pytest.approx(75.0)
    # the whole conv pair goes to the conv op type
    assert by_tasks["AxQuantizedConv"] == pytest.approx(200.0 + 50.0)


def test_mcode_estimate_reports_an_engine_with_no_tasks():
    segs = [{"engine": "conv", "bytes": 64}, {"engine": "conv", "bytes": 32}]
    out = sa.mcode_estimate(segs, {}, "tasks")
    assert out == {"(no tasks in engine)": 96.0}


def test_params_by_type_counts_overlapping_loads_once_and_uses_ocm_length():
    def load(name, start, lo, hi, tensor, dep=()):
        return _ev(
            name,
            "sdma4",
            "ld:param",
            inputs=f"{tensor}_ddr - params:[{start}:]",
            outputs=f"{tensor} - ocm_base:[{lo}:{hi}]",
            dep_by=dep,
        )

    events = [
        load("ld:w0_0_0", 0, 100, 164, "w0", dep=["op_a:AxQuantizedConv_1_s0_gop_0_0"]),
        # same params range loaded again (e.g. a second shard): counted once
        load(
            "ld:w0b_0_0", 0, 200, 264, "w0b", dep=["op_a:AxQuantizedConv_1_s1_gop_0_0"]
        ),
        load("ld:bias_rd_ocm_0_0", 64, 300, 316, "bias", dep=["Add_5_0_0"]),
        # a load nothing reads stays unattributed
        load("ld:orphan_0_0", 80, 400, 408, "orphan"),
    ]
    out = sa.params_by_type(events, PROFILE, params_len=200)
    assert out["by_type"]["AxQuantizedConv"] == 64
    assert out["by_type"]["AxAdd"] == 16
    # the orphan reads params[80:88), which no other load covers
    assert out["by_type"][sa.UNATTRIBUTED] == 8
    assert out["covered"] == 64 + 16 + 8


def test_coverage_report_fractions_sum_to_one_and_split_by_capability():
    shares = {
        "AxQuantizedConv": 50.0,
        "AxTranspose": 25.0,
        "AxReshape": 15.0,
        sa.UNATTRIBUTED: 10.0,
    }
    cov = sa.coverage_report(shares)
    assert sum(cov.values()) == pytest.approx(1.0)
    assert cov["retarget"] == pytest.approx(0.5)
    assert cov["none"] == pytest.approx(0.25)
    assert cov["metadata"] == pytest.approx(0.15)
    assert cov["unattributed"] == pytest.approx(0.10)


def test_check_segment_engines_flags_a_dma_task_count_mismatch():
    segs = [
        {"engine": "conv", "a3_verbs": 0},
        {"engine": "sdma4", "a3_verbs": 2},
    ]
    good = [_ev("a", "sdma4"), _ev("b", "sdma4")]
    bad = good + [_ev("c", "sdma4")]
    assert sa.check_segment_engines(segs, good) == []
    assert sa.check_segment_engines(segs, bad)


def test_layout_on_a_real_compiled_blob_tiles_the_stream():
    path = os.path.join(_AXERA_DIR, "fixtures", "piper_vocoder.mcode.gz")
    with gzip.open(path, "rb") as f:
        blob = f.read()
    segs = sa.layout(blob)
    assert [s["engine"] for s in segs] == list(sa.SEGMENT_ENGINES)
    # segments are contiguous and their words count matches their byte length
    for s in segs:
        assert s["bytes"] == 8 * s["words"]
    for prev, nxt in zip(segs, segs[1:]):
        assert prev["offset"] + prev["bytes"] == nxt["offset"]


def test_contained_counts_informative_windows_only():
    rng_bytes = bytes((i * 37 + 11) % 251 for i in range(200))
    other = bytes((i * 91 + 5) % 253 for i in range(200))
    # identical content is fully contained; unrelated content is not
    assert scp.contained(rng_bytes, [rng_bytes])[0] == 1.0
    assert scp.contained(rng_bytes, [other])[0] < 0.05
    # zero padding has no informative windows and is ignored, not counted as a hit
    frac, seen = scp.contained(bytes(300), [bytes(300)])
    assert seen == 0
    assert frac != frac  # NaN
    # a chain that embeds the reference plus new bytes is partly contained
    chain = rng_bytes + other
    frac, seen = scp.contained(chain, [rng_bytes])
    assert 0.4 < frac < 0.6
    assert seen == len(chain) - scp.WINDOW + 1


def test_probe_ladder_defines_every_case_the_compare_step_uses():
    names = set(scp.cases())
    for needed in (
        "conv1",
        "relu1",
        "conv_relu",
        "conv_relu_conv",
        "mm",
        "add1",
        "mm_add",
    ):
        assert needed in names
