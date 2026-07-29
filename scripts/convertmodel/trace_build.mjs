// Assemble a Chrome Trace Event Format object for an onnxruntime-web run.
//
// onnxruntime-web has no single public "give me a chrome trace" call: the
// native session profiler (`enableProfiling`) writes its JSON into the ORT wasm
// module's in-memory filesystem and then discards the filename, so it is not
// reachable from application code. What *is* public is
//   * `ort.env.webgpu.profiling.ondata`, a callback that hands us one record per
//     GPU kernel invocation (type, name, start/end GPU timestamps), and
//   * the per-iteration wall-clock timings we already measure around
//     `session.run` in inference_core.mjs.
// This module folds those two sources into a Chrome Trace Event object -- the
// exact format onnxsim's own C++ profiler emits -- so the same in-page viewer
// (and chrome://tracing / Perfetto) renders both traces.
//
// Kept dependency-free and DOM-free so it runs unchanged under Node for the
// unit test in test/trace.test.mjs.

// Chrome trace timestamps and durations are microseconds.
const US_PER_MS = 1000;
const US_PER_NS = 1 / 1000;

// A fresh, empty Chrome trace. `displayTimeUnit: "ms"` asks viewers to label the
// axis in milliseconds even though the numbers themselves are microseconds.
export function emptyTrace() {
  return { displayTimeUnit: "ms", traceEvents: [] };
}

function processName(trace, pid, name) {
  trace.traceEvents.push({
    name: "process_name",
    ph: "M",
    pid,
    tid: 0,
    args: { name },
  });
}

function threadName(trace, pid, tid, name) {
  trace.traceEvents.push({
    name: "thread_name",
    ph: "M",
    pid,
    tid,
    args: { name },
  });
}

// Build the trace from the pieces inference_core.mjs collects.
//
//   version      onnxruntime-web version string, for the process label.
//   ep           the execution provider that actually ran ("webgpu" | "wasm").
//   runs         [{ index, startMs, durMs }] -- one entry per session.run call,
//                `startMs` from a shared performance.now() epoch.
//   kernels      [{ kernelType, kernelName, programName, startTime, endTime }]
//                as delivered by ort.env.webgpu.profiling.ondata (start/end are
//                GPU timestamps in nanoseconds). Empty for the wasm EP.
//
// The two sources sit on independent clocks (wall vs. GPU), so each track is
// normalized to its own first event at t=0; durations within a track are exact
// and that is what a profiling flame graph is read for. Tracks:
//   tid 0  session.run (wall)     -- one bar per iteration
//   tid 1  onnxruntime GPU kernels -- one bar per kernel (WebGPU only)
export function buildOrtTrace({ version, ep, runs = [], kernels = [] } = {}) {
  const trace = emptyTrace();
  const pid = 1;
  processName(
    trace,
    pid,
    `onnxruntime-web ${version ?? "?"} (${ep ?? "?"})`,
  );

  // --- Wall-clock iteration bars ---
  threadName(trace, pid, 0, "session.run (wall)");
  if (runs.length > 0) {
    const wallZero = Math.min(...runs.map((r) => r.startMs));
    for (const r of runs) {
      trace.traceEvents.push({
        name: `run #${r.index}`,
        cat: "inference",
        ph: "X",
        ts: Math.round((r.startMs - wallZero) * US_PER_MS),
        dur: Math.round(r.durMs * US_PER_MS),
        pid,
        tid: 0,
        args: { iteration: r.index, wall_ms: Number(r.durMs.toFixed(3)) },
      });
    }
  }

  // --- Per-kernel GPU bars (WebGPU EP only) ---
  if (kernels.length > 0) {
    threadName(trace, pid, 1, "onnxruntime GPU kernels");
    const gpuZero = Math.min(...kernels.map((k) => k.startTime));
    for (const k of kernels) {
      const durNs = Math.max(0, k.endTime - k.startTime);
      trace.traceEvents.push({
        name: k.kernelName || k.kernelType || k.programName || "kernel",
        cat: "onnxruntime",
        ph: "X",
        ts: Math.round((k.startTime - gpuZero) * US_PER_NS),
        dur: Math.round(durNs * US_PER_NS),
        pid,
        tid: 1,
        args: {
          kernelType: k.kernelType,
          programName: k.programName,
          gpu_us: Number((durNs / 1000).toFixed(3)),
        },
      });
    }
  }

  return trace;
}

// Total wall time and, when available, summed GPU kernel time -- handy for a
// one-line status message next to the viewer.
export function summarizeOrtTrace({ runs = [], kernels = [] } = {}) {
  const wallMs = runs.reduce((a, r) => a + r.durMs, 0);
  const gpuMs =
    kernels.reduce((a, k) => a + Math.max(0, k.endTime - k.startTime), 0) /
    1e6;
  return {
    iterations: runs.length,
    kernels: kernels.length,
    wallMs,
    avgWallMs: runs.length ? wallMs / runs.length : 0,
    gpuMs,
  };
}
