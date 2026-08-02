// Shared inference core used by both the browser converter page and the Node
// smoke test, so the test exercises exactly the code the page runs.
//
// The caller passes in the already-configured `onnxruntime-web` module (its
// wasm paths / thread settings differ between Node and the browser), a model,
// an input tensor, and the list of execution providers to try. We create a
// session on the first provider that works (so a page can ask for WebGPU and
// transparently fall back to wasm), then run several iterations and report
// timings plus whether the output stayed identical across iterations.
//
// When `profile` is set the run also captures an onnxruntime-web profiling
// trace (see trace_build.mjs): per-iteration wall timings for every provider,
// plus per-kernel GPU timings on WebGPU via the profiling callback. The trace
// is returned as a ready-to-render Chrome Trace Event object.

import { buildOrtTrace } from "./trace_build.mjs";
import { providerLabel } from "./webnn.mjs";

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

// Try each execution provider in order; return the session created on the first
// one that succeeds, along with the provider name that won. Providers may be
// bare name strings ("wasm", "webgpu") or onnxruntime-web options objects (e.g.
// { name: "webnn", deviceType: "gpu" }); the returned/logged `ep` is always the
// readable label from providerLabel(). Throws only if every provider fails.
export async function createSessionWithFallback(ort, model, providers, onLog = () => {}) {
  const errors = [];
  for (const ep of providers) {
    const label = providerLabel(ep);
    try {
      const session = await ort.InferenceSession.create(model, {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });
      return { session, ep: label };
    } catch (e) {
      const msg = e && e.message ? e.message : String(e);
      onLog(`execution provider '${label}' unavailable: ${msg}`);
      errors.push(`${label}: ${msg}`);
    }
  }
  throw new Error(`no usable execution provider (${errors.join(" | ")})`);
}

// Turn on onnxruntime-web's WebGPU per-kernel profiling and collect the records
// the callback delivers. Returns { kernels, restore } where restore() puts
// env.webgpu.profiling back the way it was. A no-op-friendly wrapper: on builds
// without WebGPU profiling the array simply stays empty.
function startWebGpuProfiling(ort) {
  const kernels = [];
  const prev = ort?.env?.webgpu?.profiling;
  try {
    ort.env.webgpu.profiling = {
      mode: "default",
      ondata: (d) => kernels.push(d),
    };
  } catch {
    // env.webgpu may be absent (older builds); wall timings still get captured.
  }
  const restore = () => {
    try {
      ort.env.webgpu.profiling = prev ?? { mode: "off" };
    } catch {
      /* ignore */
    }
  };
  return { kernels, restore };
}

// Run `iterations` inference passes and verify determinism (every pass must
// equal the first) and, if `reference` is given, correctness within `tolerance`.
// Returns { ep, iterations, timings, avgMs, output, dims, trace? }.
//
// With `profile: true`, `trace` is a Chrome Trace Event object built from the
// per-iteration wall timings and (on WebGPU) the per-kernel GPU records.
export async function runInference(ort, {
  model,
  inputName,
  input,
  outputName,
  providers,
  iterations = 5,
  reference = null,
  tolerance = 1e-4,
  profile = false,
  onLog = () => {},
}) {
  // Arm WebGPU kernel profiling before the session is created so its programs
  // are built with timestamp queries enabled.
  const gpu = profile ? startWebGpuProfiling(ort) : null;

  let session;
  let ep;
  try {
    ({ session, ep } = await createSessionWithFallback(
      ort,
      model,
      providers,
      onLog,
    ));
  } catch (e) {
    gpu?.restore();
    throw e;
  }
  const resolvedInputName = inputName || session.inputNames[0];
  const resolvedOutputName = outputName || session.outputNames[0];

  let first = null;
  let lastDims = null;
  const timings = [];
  const runs = [];
  try {
    for (let i = 0; i < iterations; i++) {
      const t0 = performance.now();
      const results = await session.run({ [resolvedInputName]: input });
      const durMs = performance.now() - t0;
      timings.push(durMs);
      runs.push({ index: i, startMs: t0, durMs });

      const out = results[resolvedOutputName];
      if (!out) throw new Error(`iteration ${i}: missing output '${resolvedOutputName}'`);
      const data = out.data;
      lastDims = out.dims;

      if (reference) {
        const refDiff = maxAbsDiff(data, reference);
        if (refDiff > tolerance) {
          throw new Error(`iteration ${i}: output differs from reference by ${refDiff} (> ${tolerance})`);
        }
      }
      if (first === null) {
        first = data;
      } else if (maxAbsDiff(data, first) !== 0) {
        throw new Error(`iteration ${i}: non-deterministic across runs on '${ep}'`);
      }
      onLog(`iter ${i}: ${timings[i].toFixed(2)} ms`);
    }
  } finally {
    gpu?.restore();
  }

  const avgMs = timings.reduce((a, b) => a + b, 0) / timings.length;
  const result = { ep, iterations, timings, avgMs, output: first, dims: lastDims };
  if (profile) {
    const version = ort?.env?.versions?.web;
    result.kernels = gpu ? gpu.kernels : [];
    result.trace = buildOrtTrace({ version, ep, runs, kernels: result.kernels });
  }
  return result;
}

// Compare two runInference() results computed from the SAME input: how far the
// outputs diverge and how their speed relates. Pure (no ort), so the Node test
// drives the exact numeric logic the page shows. Returns
// { maxAbsDiff, dimsMatch, comparable, equivalent, speedup }:
//   - maxAbsDiff: largest |a - b| over the flat outputs, or null when the two
//     outputs can't be compared elementwise (missing or different length).
//   - dimsMatch:  the reported output dims are identical.
//   - comparable: both outputs were present and the same length.
//   - equivalent: comparable, dims match, and maxAbsDiff <= tolerance — i.e. the
//     graph change preserved the numerics.
//   - speedup:    a.avgMs / b.avgMs (>1 means b — the "after" model — is
//     faster), or null if a latency is non-positive.
export function compareOutputs(a, b, tolerance = 1e-3) {
  const dimsMatch = !!(
    a &&
    b &&
    Array.isArray(a.dims) &&
    Array.isArray(b.dims) &&
    a.dims.length === b.dims.length &&
    a.dims.every((d, i) => d === b.dims[i])
  );
  const comparable = !!(
    a &&
    b &&
    a.output &&
    b.output &&
    a.output.length === b.output.length
  );
  const maxDiff = comparable ? maxAbsDiff(a.output, b.output) : null;
  const equivalent =
    comparable && dimsMatch && maxDiff != null && maxDiff <= tolerance;
  const speedup =
    a && b && a.avgMs > 0 && b.avgMs > 0 ? a.avgMs / b.avgMs : null;
  return { maxAbsDiff: maxDiff, dimsMatch, comparable, equivalent, speedup };
}

// Run two models on the SAME input tensor and compare their outputs — the
// before/after check for a graph conversion (simplify/optimize). Feeding
// identical inputs isolates the effect of the graph change: a small maxAbsDiff
// means the conversion preserved the model's numerics, while the two avgMs let
// the caller report the speed difference.
//
// `before`/`after` are { model, label? }. Returns
// { before, after, ...compareOutputs(before, after) } where before/after are the
// full runInference results (so the caller can still read timings, trace, etc.).
export async function compareInference(ort, {
  before,
  after,
  inputName,
  input,
  outputName,
  providers,
  iterations = 5,
  tolerance = 1e-3,
  profile = false,
  onLog = () => {},
}) {
  const beforeLabel = before.label || "before";
  const afterLabel = after.label || "after";
  onLog(`running ${beforeLabel} (before conversion)…`);
  const a = await runInference(ort, {
    model: before.model,
    inputName,
    input,
    outputName,
    providers,
    iterations,
    profile,
    onLog,
  });
  onLog(`running ${afterLabel} (after conversion)…`);
  const b = await runInference(ort, {
    model: after.model,
    inputName,
    input,
    outputName,
    providers,
    iterations,
    profile,
    onLog,
  });
  return { before: a, after: b, ...compareOutputs(a, b, tolerance) };
}
