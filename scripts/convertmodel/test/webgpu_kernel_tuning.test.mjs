// Proves the actual point of onnxsim.webgpu_kernel_tuning.generate_kernel_candidates:
// several *alternative* WebGPU kernels for the exact same computation
// (different tinygrad Opt tunings -- see that module's own docstring for
// why tinygrad's own BEAM search can't be used directly here) can be
// dispatched on a real WebGPU device, timed with
// webgpu_kernel_dispatcher.mjs's own dispatchWebgpuProgram(..., {profile: true}),
// and the fastest one picked -- while every candidate still computes the
// right answer, not just the one that happens to get chosen.
//
// webgpu_kernel_tuning_fixture.json (see make_webgpu_kernel_tuning_fixture.py)
// carries several standalone WebgpuKernelSpec candidates for one real
// Conv2D (all sharing identical bindings -- only wgsl/entry_point/dispatch
// differ between them), the concrete x/w values used to generate them, and
// the expected output from onnx.reference.ReferenceEvaluator running the
// real node.
//
// Requires the "playwright" package and a Chromium binary -- same
// requirement as the other webgpu_*.test.mjs files here.
//
// Usage:
//   npx playwright install chromium   # once
//   pip install onnx numpy tinygrad==0.14.0
//   python3 make_webgpu_kernel_tuning_fixture.py   # regenerate fixtures if needed
//   node test/webgpu_kernel_tuning.test.mjs

import assert from "node:assert/strict";
import http from "node:http";
import fs, { readFileSync } from "node:fs";
import path, { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const MANIFEST = JSON.parse(readFileSync(join(HERE, "webgpu_kernel_tuning_fixture.json"), "utf8"));

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

// Same minimal static file server as the other browser tests here.
function serveConvertmodelDir() {
  const server = http.createServer((req, res) => {
    const reqPath = decodeURIComponent(req.url.split("?")[0]);
    if (reqPath === "/") {
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end("<!doctype html><title>webgpu kernel tuning</title>");
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

// Runs inside the real browser page via page.evaluate -- no access to this
// file's own scope, only what's passed in and what it can import()/fetch().
async function runInPage({ port, outputName, inputs, candidates, expectedLength }) {
  const base = `http://localhost:${port}`;
  const { dispatchWebgpuProgram, createStorageBuffer, readBackFloat32Buffer, supportsWebgpuProfiling } = await import(
    `${base}/webgpu_kernel_dispatcher.mjs`
  );

  const adapter = await navigator.gpu.requestAdapter();
  const wantTimestampQuery = adapter.features.has("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: wantTimestampQuery ? ["timestamp-query"] : [],
  });
  const canProfile = supportsWebgpuProfiling(device);

  const results = [];
  for (const candidate of candidates) {
    const buffersByTensor = new Map();
    for (const [name, { data }] of Object.entries(inputs)) {
      buffersByTensor.set(name, createStorageBuffer(device, Float32Array.from(data)));
    }
    buffersByTensor.set(outputName, createStorageBuffer(device, new Float32Array(expectedLength)));

    const { timings } = await dispatchWebgpuProgram(device, candidate.spec, buffersByTensor, { profile: true });
    const actual = await readBackFloat32Buffer(device, buffersByTensor.get(outputName), expectedLength);

    results.push({
      appliedOpts: candidate.appliedOpts,
      data: Array.from(actual),
      durationNs: timings ? timings.reduce((sum, t) => sum + t.durationNs, 0) : null,
    });
  }
  return { ok: true, canProfile, results };
}

async function main() {
  console.log(`WebGPU kernel tuning check -- ${MANIFEST.candidates.length} candidates\n`);

  const server = await serveConvertmodelDir();
  const port = server.address().port;
  const browser = await chromium.launch({
    headless: true,
    // Verified sufficient on its own in this repo's own dev sandbox (via
    // SwiftShader's software Vulkan path); see webgpu_hf_demo.test.mjs's own
    // comment. A real GPU, as CI runners with one have, needs nothing more.
    args: ["--enable-unsafe-webgpu"],
  });

  let result;
  try {
    const page = await browser.newPage();
    await page.goto(`http://localhost:${port}/`);
    result = await page.evaluate(runInPage, {
      port,
      outputName: MANIFEST.outputName,
      inputs: MANIFEST.inputs,
      candidates: MANIFEST.candidates,
      expectedLength: MANIFEST.expectedOutput.data.length,
    });
    await page.close();
  } finally {
    await browser.close();
    server.close();
  }

  await check("every candidate dispatched without throwing", () => {
    assert.ok(result.ok);
    assert.equal(result.results.length, MANIFEST.candidates.length);
  });

  const expected = MANIFEST.expectedOutput.data;
  await check("every candidate computes the correct output, not just the one that gets picked", () => {
    for (const r of result.results) {
      let maxAbsDiff = 0;
      for (let i = 0; i < expected.length; i++) {
        maxAbsDiff = Math.max(maxAbsDiff, Math.abs(r.data[i] - expected[i]));
      }
      assert.ok(
        maxAbsDiff < 1e-2,
        `candidate ${r.appliedOpts} max abs diff ${maxAbsDiff} too large`,
      );
    }
  });

  await check("real per-candidate GPU durations are reported when the device supports profiling", () => {
    if (!result.canProfile) {
      console.log('    (device lacks "timestamp-query" here -- skipping the timing assertion itself)');
      for (const r of result.results) assert.equal(r.durationNs, null);
      return;
    }
    for (const r of result.results) {
      assert.ok(
        Number.isFinite(r.durationNs) && r.durationNs >= 0,
        `expected a non-negative finite duration for ${r.appliedOpts}, got ${r.durationNs}`,
      );
    }
    const ranked = [...result.results].sort((a, b) => a.durationNs - b.durationNs);
    console.log(
      "    (ranked by GPU duration: " +
        ranked.map((r) => `${r.appliedOpts}=${r.durationNs}ns`).join(", ") +
        ")",
    );
    console.log(`    fastest candidate: ${ranked[0].appliedOpts} (${ranked[0].durationNs}ns)`);
  });

  console.log(`\nwebgpu kernel tuning: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
