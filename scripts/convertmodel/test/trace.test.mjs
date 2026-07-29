// Unit test for the onnxruntime-web trace assembler (trace_build.mjs).
//
// buildOrtTrace() is pure and DOM-free, so we exercise its Chrome Trace Event
// output directly under Node -- no browser, GPU or onnxruntime-web needed. This
// guards the shape the in-page flame-graph viewer (trace_viewer.mjs) and
// Perfetto rely on: an { displayTimeUnit, traceEvents } object with process /
// thread metadata and per-span "X" events at the right tracks and offsets.

import assert from "node:assert/strict";
import { buildOrtTrace, summarizeOrtTrace } from "../trace_build.mjs";

function findEvents(trace, pred) {
  return trace.traceEvents.filter(pred);
}

// --- WebGPU: wall iteration bars on tid 0, GPU kernels on tid 1 ---
{
  const runs = [
    { index: 0, startMs: 100, durMs: 4 },
    { index: 1, startMs: 110, durMs: 2 },
  ];
  const kernels = [
    // start/end are GPU timestamps in nanoseconds.
    { kernelType: "Conv", kernelName: "conv1", programName: "Conv", startTime: 1_000_000, endTime: 1_500_000 },
    { kernelType: "Relu", kernelName: "relu1", programName: "Relu", startTime: 1_500_000, endTime: 1_600_000 },
  ];
  const trace = buildOrtTrace({ version: "1.27.0", ep: "webgpu", runs, kernels });

  assert.equal(trace.displayTimeUnit, "ms");
  assert.ok(Array.isArray(trace.traceEvents));

  // Metadata present.
  assert.equal(findEvents(trace, (e) => e.ph === "M" && e.name === "process_name").length, 1);
  const threadNames = findEvents(trace, (e) => e.ph === "M" && e.name === "thread_name");
  assert.equal(threadNames.length, 2, "one thread_name per track");

  // Wall bars: tid 0, normalized so the first run starts at t=0.
  const wall = findEvents(trace, (e) => e.ph === "X" && e.tid === 0);
  assert.equal(wall.length, 2);
  assert.equal(wall[0].ts, 0, "first run at t=0");
  assert.equal(wall[0].dur, 4000, "4 ms -> 4000 us");
  assert.equal(wall[1].ts, 10000, "second run 10 ms later");

  // Kernel bars: tid 1, normalized to the first kernel, ns -> us.
  const gpu = findEvents(trace, (e) => e.ph === "X" && e.tid === 1);
  assert.equal(gpu.length, 2);
  assert.equal(gpu[0].ts, 0);
  assert.equal(gpu[0].dur, 500, "0.5 ms kernel -> 500 us");
  assert.equal(gpu[0].name, "conv1");
  assert.equal(gpu[1].ts, 500);

  const s = summarizeOrtTrace({ runs, kernels });
  assert.equal(s.iterations, 2);
  assert.equal(s.kernels, 2);
  assert.equal(s.wallMs, 6);
  assert.equal(s.avgWallMs, 3);
  assert.ok(Math.abs(s.gpuMs - 0.6) < 1e-9, "0.5 + 0.1 ms of GPU time");
}

// --- WASM (no kernels): only the wall track ---
{
  const runs = [{ index: 0, startMs: 0, durMs: 7 }];
  const trace = buildOrtTrace({ version: "1.27.0", ep: "wasm", runs, kernels: [] });
  const gpu = findEvents(trace, (e) => e.ph === "X" && e.tid === 1);
  assert.equal(gpu.length, 0, "no GPU track without kernels");
  const wall = findEvents(trace, (e) => e.ph === "X" && e.tid === 0);
  assert.equal(wall.length, 1);
  assert.equal(wall[0].dur, 7000);
}

// --- Empty run is still a valid trace object ---
{
  const trace = buildOrtTrace({ version: "?", ep: "wasm", runs: [], kernels: [] });
  assert.equal(trace.displayTimeUnit, "ms");
  assert.equal(findEvents(trace, (e) => e.ph === "X").length, 0);
}

console.log("PASS: trace_build.mjs assembles valid Chrome traces");
