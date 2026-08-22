// Unit test for the "node reduction per loop" chart's pure data extraction
// (node_reduction_view.mjs's extractNodeCounts). DOM-free, so it runs under
// Node -- the rendering half (renderNodeReduction) is exercised manually in
// the browser via the converter page's "Node reduction per loop" section.

import assert from "node:assert/strict";
import { extractNodeCounts } from "../node_reduction_view.mjs";

// --- A trace shaped like what onnxsim's C++ profiler actually writes ---
{
  const trace = {
    displayTimeUnit: "ms",
    traceEvents: [
      { name: "process_name", ph: "M", pid: 1, tid: 0, args: { name: "onnxsim simplify" } },
      { name: "NodeCount", ph: "C", ts: 0, pid: 1, tid: 0, args: { node_count: 360, loop: "Initial" } },
      { name: "Simplify", ph: "X", ts: 0, dur: 100, pid: 1, tid: 1, args: {} }, // not a counter event
      { name: "NodeCount", ph: "C", ts: 10, pid: 1, tid: 0, args: { node_count: 600, loop: "Optimize" } },
      { name: "NodeCount", ph: "C", ts: 20, pid: 1, tid: 0, args: { node_count: 205, loop: "Optimize" } },
      { name: "NodeCount", ph: "C", ts: 30, pid: 1, tid: 0, args: { node_count: 160, loop: "Optimize" } },
      { name: "NodeCount", ph: "C", ts: 40, pid: 1, tid: 0, args: { node_count: 160, loop: "FoldConstant" } },
      { name: "RSS", ph: "C", ts: 5, pid: 1, tid: 0, args: { rss_mb: 12.3 } }, // a different counter track
    ],
  };

  const counts = extractNodeCounts(trace);
  assert.deepEqual([...counts.keys()], ["Initial", "Optimize", "FoldConstant"]);
  assert.deepEqual(counts.get("Initial"), [360]);
  assert.deepEqual(counts.get("Optimize"), [600, 205, 160]);
  assert.deepEqual(counts.get("FoldConstant"), [160]);
}

// --- A trace with no NodeCount events (e.g. profiling off, or an old trace) ---
{
  const trace = { displayTimeUnit: "ms", traceEvents: [] };
  const counts = extractNodeCounts(trace);
  assert.equal(counts.size, 0);
}

// --- Malformed/missing fields are skipped, not thrown ---
{
  const trace = {
    traceEvents: [
      { name: "NodeCount", ph: "C", args: {} }, // no node_count
      { name: "NodeCount", ph: "C", args: { node_count: "not a number", loop: "Optimize" } },
      { name: "NodeCount", ph: "C", args: { node_count: 42 } }, // no loop -> "?"
    ],
  };
  const counts = extractNodeCounts(trace);
  assert.deepEqual([...counts.keys()], ["?"]);
  assert.deepEqual(counts.get("?"), [42]);
}

console.log("PASS: node_reduction_view.mjs extracts NodeCount series correctly");
