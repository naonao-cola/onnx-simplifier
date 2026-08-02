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
  feeds = null,
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
  // Feed every model input. A caller with a multi-input model passes the whole
  // `feeds` map ({ name: tensor }); the single-input callers (and the Node smoke
  // test) pass just `input`, which we bind to the first/only input name. Without
  // this, a model like MatMul(X, W) + Add(_, B) fails with
  // "input 'W' is missing in 'feeds'".
  const runFeeds = feeds || { [resolvedInputName]: input };

  let first = null;
  let lastDims = null;
  const timings = [];
  const runs = [];
  try {
    for (let i = 0; i < iterations; i++) {
      const t0 = performance.now();
      const results = await session.run(runFeeds);
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
