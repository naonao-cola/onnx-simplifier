"""Unit tests for merging onnxruntime session traces into onnxsim's profile.

These exercise ``onnxsim.profile_merge`` directly with synthetic traces, so they
run without building onnxsim's native extension (the module is standard-library
only by design).
"""

import json
import os

from onnxsim import profile_merge


def _ort_span(name, ts, dur, tid=7):
    return {"name": name, "ph": "X", "ts": ts, "dur": dur, "pid": 1, "tid": tid}


def _profile_with_sessions(*spans):
    events = [
        {"name": "FoldConstant", "ph": "X", "ts": 0, "dur": 10_000, "pid": 1, "tid": 7}
    ]
    events.extend(spans)
    return {"displayTimeUnit": "ms", "traceEvents": events}


def _ort_trace(events):
    return [
        {"name": n, "ph": "X", "ts": ts, "dur": dur, "pid": 9, "tid": tid}
        for (n, ts, dur, tid) in events
    ]


def test_merge_offsets_ort_events_under_span():
    # One OrtSession span starting at ts=1000; the onnxruntime trace's own
    # timeline starts at 0 and must be shifted to line up under the span.
    trace = _profile_with_sessions(_ort_span("OrtSession", 1000, 500))
    ort = _ort_trace([("Conv", 0, 100, 1), ("Relu", 100, 50, 1)])

    profile_merge.merge_ort_traces(trace, [_write(ort)])

    ort_events = [e for e in trace["traceEvents"] if e.get("cat") == "onnxruntime"]
    assert {e["name"] for e in ort_events} == {"Conv", "Relu"}
    # Earliest onnxruntime event is shifted to the span start (offset = 1000 - 0).
    assert min(e["ts"] for e in ort_events) == 1000
    assert {e["ts"] for e in ort_events} == {1000, 1100}
    # onnxruntime events land on a dedicated track, not onnxsim's span thread.
    assert all(e["tid"] >= 1000 for e in ort_events)
    # The track is labelled.
    names = [
        e["args"]["name"]
        for e in trace["traceEvents"]
        if e.get("ph") == "M" and e.get("name") == "thread_name"
    ]
    assert any("onnxruntime session 0" == n for n in names)


def test_merge_matches_sessions_in_order_and_ignores_extra():
    # Two spans, three onnxruntime traces: the third (e.g. a correctness-check
    # run) has no span to attach to and must be dropped.
    trace = _profile_with_sessions(
        _ort_span("OrtSession", 1000, 200),
        _ort_span("OrtSession", 5000, 200),
    )
    t0 = _write(_ort_trace([("A", 0, 10, 1)]))
    t1 = _write(_ort_trace([("B", 0, 10, 1)]))
    t2 = _write(_ort_trace([("C", 0, 10, 1)]))

    profile_merge.merge_ort_traces(trace, [t0, t1, t2])

    ort = {
        e["name"]: e["ts"]
        for e in trace["traceEvents"]
        if e.get("cat") == "onnxruntime"
    }
    assert ort == {"A": 1000, "B": 5000}  # C dropped; A/B aligned to their spans
    # Distinct sessions use distinct tracks.
    tids = {e["tid"] for e in trace["traceEvents"] if e.get("cat") == "onnxruntime"}
    assert len(tids) == 2


def test_merge_no_ort_session_spans_is_noop():
    trace = {"traceEvents": [{"name": "FoldConstant", "ph": "X", "ts": 0, "dur": 1}]}
    before = json.dumps(trace, sort_keys=True)
    profile_merge.merge_ort_traces(trace, [_write(_ort_trace([("A", 0, 1, 1)]))])
    assert json.dumps(trace, sort_keys=True) == before


def test_merge_skips_unreadable_trace(tmp_path):
    trace = _profile_with_sessions(_ort_span("OrtSession", 1000, 200))
    bad = str(tmp_path / "bad.json")
    with open(bad, "w") as f:
        f.write("{ this is not valid json")
    # Must not raise; just no onnxruntime events added.
    profile_merge.merge_ort_traces(trace, [bad])
    assert not [e for e in trace["traceEvents"] if e.get("cat") == "onnxruntime"]


def test_merge_into_profile_reads_dir_in_mtime_order(tmp_path):
    profile_path = str(tmp_path / "profile.json")
    with open(profile_path, "w") as f:
        json.dump(
            _profile_with_sessions(
                _ort_span("OrtSession", 1000, 200),
                _ort_span("OrtSession", 5000, 200),
            ),
            f,
        )

    ort_dir = tmp_path / "ort"
    ort_dir.mkdir()
    # Write two onnxruntime traces with increasing mtime; first -> first span.
    first = ort_dir / "session_1.json"
    second = ort_dir / "session_2.json"
    with open(first, "w") as f:
        json.dump(_ort_trace([("First", 0, 10, 1)]), f)
    with open(second, "w") as f:
        json.dump(_ort_trace([("Second", 0, 10, 1)]), f)
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))

    profile_merge.merge_ort_traces_into_profile(profile_path, str(ort_dir))

    with open(profile_path) as f:
        merged = json.load(f)
    ort = {
        e["name"]: e["ts"]
        for e in merged["traceEvents"]
        if e.get("cat") == "onnxruntime"
    }
    assert ort == {"First": 1000, "Second": 5000}


def test_merge_into_profile_missing_profile_is_noop(tmp_path):
    # No profile file: nothing to do, and no error.
    ort_dir = tmp_path / "ort"
    ort_dir.mkdir()
    profile_merge.merge_ort_traces_into_profile(
        str(tmp_path / "absent.json"), str(ort_dir)
    )


_TMP_COUNTER = [0]


def _write(ort_events, _dir=[None]):
    """Write an onnxruntime trace to a fresh temp file and return its path."""
    import tempfile

    if _dir[0] is None:
        _dir[0] = tempfile.mkdtemp(prefix="ort_trace_")
    _TMP_COUNTER[0] += 1
    path = os.path.join(_dir[0], f"trace_{_TMP_COUNTER[0]}.json")
    with open(path, "w") as f:
        json.dump(ort_events, f)
    return path
