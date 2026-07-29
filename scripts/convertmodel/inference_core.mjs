// Shared inference core used by both the browser converter page and the Node
// smoke test, so the test exercises exactly the code the page runs.
//
// The caller passes in the already-configured `onnxruntime-web` module (its
// wasm paths / thread settings differ between Node and the browser), a model,
// an input tensor, and the list of execution providers to try. We create a
// session on the first provider that works (so a page can ask for WebGPU and
// transparently fall back to wasm), then run several iterations and report
// timings plus whether the output stayed identical across iterations.

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

// Try each execution provider in order; return the session created on the first
// one that succeeds, along with the provider name that won. Throws only if every
// provider fails.
export async function createSessionWithFallback(ort, model, providers, onLog = () => {}) {
  const errors = [];
  for (const ep of providers) {
    try {
      const session = await ort.InferenceSession.create(model, {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });
      return { session, ep };
    } catch (e) {
      const msg = e && e.message ? e.message : String(e);
      onLog(`execution provider '${ep}' unavailable: ${msg}`);
      errors.push(`${ep}: ${msg}`);
    }
  }
  throw new Error(`no usable execution provider (${errors.join(" | ")})`);
}

// Run `iterations` inference passes and verify determinism (every pass must
// equal the first) and, if `reference` is given, correctness within `tolerance`.
// Returns { ep, iterations, timings, avgMs, output, dims }.
export async function runInference(ort, {
  model,
  inputName,
  input,
  outputName,
  providers,
  iterations = 5,
  reference = null,
  tolerance = 1e-4,
  onLog = () => {},
}) {
  const { session, ep } = await createSessionWithFallback(ort, model, providers, onLog);
  const resolvedInputName = inputName || session.inputNames[0];
  const resolvedOutputName = outputName || session.outputNames[0];

  let first = null;
  let lastDims = null;
  const timings = [];
  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    const results = await session.run({ [resolvedInputName]: input });
    timings.push(performance.now() - t0);

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

  const avgMs = timings.reduce((a, b) => a + b, 0) / timings.length;
  return { ep, iterations, timings, avgMs, output: first, dims: lastDims };
}
