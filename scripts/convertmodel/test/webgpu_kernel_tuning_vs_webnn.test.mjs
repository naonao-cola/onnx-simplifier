// How much could tinygrad's own browser-tuned kernel (see
// webgpu_kernel_tuning.test.mjs / docs/webgpu-kernel-tuning.md) outperform
// onnxruntime-web's WebNN execution provider on the *same* op, on a *real*
// device? Both sides run the identical Conv2D
// (make_webgpu_kernel_tuning_fixture.py's fixture: x=(1,4,16,16),
// w=(4,4,3,3), pad=1) -- the tinygrad side via dispatchWebgpuProgram against
// every tuned candidate (fastest wins), the WebNN side via a plain
// onnxruntime-web session over webgpu_kernel_tuning_fixture.onnx (the same
// Conv2D as an ordinary standalone model) -- and both are timed the same
// way: wall-clock median over several warmed-up runs, since WebNN has no
// GPU-timestamp-query equivalent exposed through onnxruntime-web (unlike
// dispatchWebgpuProgram's own `profile: true`, which is GPU-time only and
// so isn't directly comparable to a JS-side session.run() call).
//
// Deliberately ATTEMPTED AND REPORTED, never required to pass CI -- same
// posture as webnn_reshape_placement.test.mjs, for the same reason: WebNN's
// browser support is still experimental (docs/webnn.md). Concretely, in this
// repo's own dev sandbox (headless Linux Chromium) `navigator.ml` is absent
// under plain `--enable-unsafe-webgpu`, but *does* appear -- and its "gpu"
// device type context actually builds and runs -- once
// `--enable-features=WebMachineLearningNeuralNetwork` is also passed (the
// same flag webnn_reshape_placement.test.mjs already uses). So availability
// here depends on that flag, not the platform alone; on a browser/platform
// where it's still absent, this still reports tinygrad's own fastest number
// (useful on its own) and skips only the comparison-specific checks, the
// same way webnn_reshape_placement.test.mjs skips instead of failing.
//
// A measured result from that sandbox (Conv2D, x=(1,4,16,16), w=(4,4,3,3)):
// WebNN's "gpu" device type came out ~1.7-2.4x *faster* than tinygrad's own
// best-tuned candidate across repeated runs -- the opposite direction from
// what might be assumed. Read that with real caution though: neither side is
// running on real hardware there -- WebGPU goes through SwiftShader's
// software rasterizer (see webgpu_hf_demo.test.mjs's own comment) and
// Chromium's WebNN "gpu" device type falls back to its own software ML
// backend when there's no real GPU/NPU init path in a headless Linux
// container -- so this doesn't say which one wins on an end user's actual
// GPU or NPU, only that neither is a safe default assumption. Re-run this on
// real hardware (a real macOS/Windows CI runner or a developer's own
// machine) before treating the direction, not just the magnitude, as settled.
//
// Requires the "playwright" package, a Chromium binary, and "onnxruntime-web"
// -- same requirements as webgpu_kernel_tuning.test.mjs and
// webnn_reshape_placement.test.mjs.
//
// Usage:
//   npx playwright install chromium   # once
//   pip install onnx numpy onnxruntime 'tinygrad==0.14.0'
//   python3 make_webgpu_kernel_tuning_fixture.py   # regenerate fixtures if needed
//   node test/webgpu_kernel_tuning_vs_webnn.test.mjs
//   ORT_REQUIRE_WEBNN=1 node test/webgpu_kernel_tuning_vs_webnn.test.mjs

import assert from "node:assert/strict";
import http from "node:http";
import fs, { readFileSync } from "node:fs";
import path, { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const MANIFEST = JSON.parse(readFileSync(join(HERE, "webgpu_kernel_tuning_fixture.json"), "utf8"));
const REQUIRE_WEBNN = !!process.env.ORT_REQUIRE_WEBNN;

// Warm-up runs discarded before timing, then this many timed runs per side,
// median taken -- smooths out one-off scheduling noise without needing many
// iterations (this is a comparison of magnitude, not a rigorous benchmark).
const WARMUP_RUNS = 3;
const TIMED_RUNS = 7;

let passed = 0;
let skipped = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}
function skip(name, reason) {
  if (REQUIRE_WEBNN) {
    throw new Error(`${name}: ${reason} (failing because ORT_REQUIRE_WEBNN=1)`);
  }
  skipped += 1;
  console.log("  skip -", name, "-", reason);
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Same minimal static file server as the other browser tests here.
function serveConvertmodelDir() {
  const server = http.createServer((req, res) => {
    const reqPath = decodeURIComponent(req.url.split("?")[0]);
    if (reqPath === "/") {
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end("<!doctype html><title>webgpu kernel tuning vs webnn</title>");
      return;
    }
    const filePath = join(ROOT, reqPath);
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end("not found: " + reqPath);
        return;
      }
      const type = {
        ".html": "text/html", ".mjs": "text/javascript", ".js": "text/javascript",
        ".json": "application/json", ".onnx": "application/octet-stream",
        ".wasm": "application/wasm",
      }[path.extname(filePath)] || "application/octet-stream";
      res.writeHead(200, { "Content-Type": type });
      res.end(data);
    });
  });
  return new Promise((resolve) => {
    server.listen(0, () => resolve(server));
  });
}

// Runs the fastest-of-N-candidates tinygrad kernel `TIMED_RUNS` times
// (plus warmup), wall-clock, and checks the output. dispatchWebgpuProgram
// itself awaits device.queue.onSubmittedWorkDone() before returning, so a
// plain performance.now() wrap around it is already a fair, GPU-complete
// wall-clock measurement.
// Everything here runs inside the browser page via page.evaluate, which
// serializes only the function body -- no access to this file's own
// top-level scope (WARMUP_RUNS/TIMED_RUNS/median included), so every value
// it needs travels through the single params object and raw per-run
// durations travel back out for median() (defined above, in Node) to reduce.
async function timeTinygradCandidates({ port, outputName, inputs, candidates, expected, warmupRuns, timedRuns }) {
  const base = `http://localhost:${port}`;
  const { dispatchWebgpuProgram, createStorageBuffer, readBackFloat32Buffer } = await import(
    `${base}/webgpu_kernel_dispatcher.mjs`
  );

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const runCandidate = async (candidate) => {
    const buffersByTensor = new Map();
    for (const [name, { data }] of Object.entries(inputs)) {
      buffersByTensor.set(name, createStorageBuffer(device, Float32Array.from(data)));
    }
    buffersByTensor.set(outputName, createStorageBuffer(device, new Float32Array(expected.length)));

    const t0 = performance.now();
    await dispatchWebgpuProgram(device, candidate.spec, buffersByTensor, {});
    const durationMs = performance.now() - t0;

    const actual = await readBackFloat32Buffer(device, buffersByTensor.get(outputName), expected.length);
    return { durationMs, data: Array.from(actual) };
  };

  const results = [];
  for (const candidate of candidates) {
    for (let i = 0; i < warmupRuns; i++) await runCandidate(candidate);
    const durationsMs = [];
    let lastData = null;
    for (let i = 0; i < timedRuns; i++) {
      const { durationMs, data } = await runCandidate(candidate);
      durationsMs.push(durationMs);
      lastData = data;
    }
    results.push({ appliedOpts: candidate.appliedOpts, durationsMs, data: lastData });
  }

  return { ok: true, results };
}

// Detects a usable WebNN device and, if one exists, times
// webgpu_kernel_tuning_fixture.onnx through onnxruntime-web's WebNN EP the
// same way: warmup, then `timedRuns` wall-clock session.run() calls.
async function timeWebnn({ port, outputName, inputs, expectedLength, warmupRuns, timedRuns }) {
  const base = `http://localhost:${port}`;
  const { detectWebnn, webnnProvider, WEBNN_DEVICE_TYPES } = await import(`${base}/webnn.mjs`);

  const report = await detectWebnn();
  const deviceType = WEBNN_DEVICE_TYPES.find((d) => report.devices[d]);
  if (!deviceType) return { available: false, report };

  const ortMod = await import(`${base}/node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs`);
  const ort = ortMod.default ?? ortMod;
  ort.env.wasm.wasmPaths = `${base}/node_modules/onnxruntime-web/dist/`;

  const modelBytes = await fetch(`${base}/test/webgpu_kernel_tuning_fixture.onnx`).then((r) => r.arrayBuffer());
  const session = await ort.InferenceSession.create(new Uint8Array(modelBytes), {
    executionProviders: [webnnProvider(deviceType), "wasm"],
    graphOptimizationLevel: "disabled",
  });

  // Only feed the model's actual graph inputs -- `w` in the fixture's inputs
  // map is a Conv weight baked into the model as an initializer (matching
  // generate_kernel_candidates' own named_tensors contract, which has no
  // such input/initializer distinction), so onnxruntime-web rejects it as an
  // unknown feed if passed here.
  const feeds = {};
  for (const name of session.inputNames) {
    const { shape, data } = inputs[name];
    feeds[name] = new ort.Tensor("float32", Float32Array.from(data), shape);
  }

  const runOnce = async () => {
    const t0 = performance.now();
    const out = await session.run(feeds);
    const durationMs = performance.now() - t0;
    return { durationMs, data: Array.from(out[outputName].data) };
  };

  for (let i = 0; i < warmupRuns; i++) await runOnce();
  const durationsMs = [];
  let lastData = null;
  for (let i = 0; i < timedRuns; i++) {
    const { durationMs, data } = await runOnce();
    durationsMs.push(durationMs);
    lastData = data;
  }
  await session.release?.();

  return { available: true, deviceType, durationsMs, data: lastData };
}

function maxAbsDiff(actual, expected) {
  let max = 0;
  for (let i = 0; i < expected.length; i++) max = Math.max(max, Math.abs(actual[i] - expected[i]));
  return max;
}

async function main() {
  console.log(
    `WebGPU kernel tuning vs WebNN check (best-effort) -- ${MANIFEST.candidates.length} tinygrad candidates\n`,
  );

  const server = await serveConvertmodelDir();
  const port = server.address().port;
  const browser = await chromium.launch({
    headless: true,
    // Both flags at once: WebGPU for the tinygrad-side dispatch (same as
    // webgpu_kernel_tuning.test.mjs), WebNN's flag for the comparison side
    // (same as webnn_reshape_placement.test.mjs) -- one page needs both APIs.
    args: ["--enable-unsafe-webgpu", "--enable-features=WebMachineLearningNeuralNetwork"],
  });

  const expected = MANIFEST.expectedOutput.data;
  let tinygrad;
  let webnn;
  try {
    const page = await browser.newPage();
    await page.goto(`http://localhost:${port}/`);

    tinygrad = await page.evaluate(timeTinygradCandidates, {
      port,
      outputName: MANIFEST.outputName,
      inputs: MANIFEST.inputs,
      candidates: MANIFEST.candidates,
      expected,
      warmupRuns: WARMUP_RUNS,
      timedRuns: TIMED_RUNS,
    });

    webnn = await page.evaluate(timeWebnn, {
      port,
      outputName: MANIFEST.outputName,
      inputs: MANIFEST.inputs,
      expectedLength: expected.length,
      warmupRuns: WARMUP_RUNS,
      timedRuns: TIMED_RUNS,
    });

    await page.close();
  } finally {
    await browser.close();
    server.close();
  }

  await check("every tinygrad-tuned candidate computes the correct output", () => {
    assert.ok(tinygrad.ok);
    for (const r of tinygrad.results) {
      const diff = maxAbsDiff(r.data, expected);
      assert.ok(diff < 1e-2, `candidate ${r.appliedOpts} max abs diff ${diff} too large`);
    }
  });

  const tinygradByMedian = tinygrad.results
    .map((r) => ({ ...r, medianMs: median(r.durationsMs) }))
    .sort((a, b) => a.medianMs - b.medianMs);
  const fastestTinygrad = tinygradByMedian[0];
  console.log(
    `    fastest tinygrad candidate: ${fastestTinygrad.appliedOpts} (${fastestTinygrad.medianMs.toFixed(3)}ms median of ${TIMED_RUNS} runs)`,
  );

  if (!webnn.available) {
    skip(
      "tinygrad vs WebNN comparison",
      `no WebNN device could be created in this browser (${webnn.report.message})`,
    );
    console.log(`\nwebgpu kernel tuning vs webnn: ${passed} checks passed, ${skipped} skipped`);
    return;
  }

  await check(`WebNN (${webnn.deviceType}) computes the correct output`, () => {
    const diff = maxAbsDiff(webnn.data, expected);
    assert.ok(diff < 1e-2, `WebNN output max abs diff ${diff} too large`);
  });

  const webnnMedianMs = median(webnn.durationsMs);
  const ratio = webnnMedianMs / fastestTinygrad.medianMs;
  console.log(`    WebNN (${webnn.deviceType}): ${webnnMedianMs.toFixed(3)}ms median of ${TIMED_RUNS} runs`);
  if (ratio >= 1) {
    console.log(`    tinygrad's tuned kernel is ~${ratio.toFixed(2)}x faster than WebNN here`);
  } else {
    console.log(`    WebNN is ~${(1 / ratio).toFixed(2)}x faster than tinygrad's tuned kernel here`);
  }

  console.log(`\nwebgpu kernel tuning vs webnn: ${passed} checks passed, ${skipped} skipped`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
