// End-to-end check of the opt-in "Tune this kernel" feature in the actual
// converter page (not a synthetic harness page) -- webgpu_kernel_tuner.mjs
// driven live from webgpu_kernel_annotations_view.mjs's own "Custom WebGPU
// kernels" panel, exactly the way a real visitor would click through it, for
// two scenarios:
//
//   A. A fixture that already carries an attached kernel spec on its one
//      flagged Conv node (webgpu_tinygrad_conv3d.onnx) -- the button lives
//      inside that node's own already-shown entry.
//   B. A *plain* Conv model with NO kernel metadata attached at all
//      (webgpu_kernel_tuning_fixture.onnx) -- not already flagged by
//      onnxsim.webgpu_tinygrad_codegen's server-side gap detection, which is
//      nearly every real upload. This used to show no "Tune this kernel"
//      button whatsoever, because the button only ever lived inside an
//      already-attached annotation's own entry -- reported directly against
//      the deployed page. See webgpu_kernel_annotations_view.mjs's own
//      setSide/listConvNodeNames fix (the "Tune a Conv kernel" section).
//
// Each scenario, via the shared runScenario() below:
//
//   1. Load the fixture via `window.webgpuKernelsShowBefore`, the same hook
//      Hugging Face loads use.
//   2. Confirm the panel shows a "Tune this kernel…" button for the node.
//   3. Click it and wait for the real Pyodide+tinygrad candidate generation
//      and real WebGPU dispatch loop to finish -- needs real network access
//      (jsdelivr for Pyodide, PyPI for tinygrad's wheel), same requirement
//      as the pyodide_webgpu_*_codegen.test.mjs files, plus a real WebGPU
//      device (Playwright/Chromium). Pyodide/tinygrad load once and stay
//      cached (webgpu_kernel_tuner.mjs's own module-level singleton) as long
//      as the page isn't reloaded, so scenario B's own tuning run is fast.
//   4. Click "Export tuned model" and capture the real browser download.
//   5. Verify the downloaded bytes actually carry the winning candidate's
//      spec (via onnx_node_metadata.mjs's readWebgpuKernelSpecs, in Node --
//      the same reader the "before"/"after" panes themselves use) and that
//      dispatching *that* spec against a real device reproduces the
//      fixture's own ground-truth expectedOutput -- so this isn't just "a
//      button produced a file", the exported model's attached kernel is
//      proven to still compute the right answer.
//
// Requires real network access from inside the *browser page itself* (not
// just the Node test process, unlike every other pyodide_*.test.mjs file
// here) -- webgpu_kernel_tuner.mjs's Pyodide load and tinygrad wheel fetch
// both happen client-side, exactly like a real visitor's browser would do
// it. Verified in this repo's own dev sandbox by temporarily pointing
// webgpu_kernel_tuner.mjs's PYODIDE_CDN_URL at the already-installed
// node_modules/pyodide/pyodide.js (served by this test's own static server)
// instead of the real jsdelivr CDN, and reverted before committing --
// this sandbox's own egress policy denies cdn.jsdelivr.net outright
// (CONNECT refused with 403, confirmed via the agent-proxy status endpoint),
// which is a property of this one sandbox, not of a real visitor's browser
// or (expected) of the GitHub-hosted runner this job runs on. Every other
// piece (tinygrad wheel fetch from PyPI, the real WebGPU dispatch loop, the
// protobuf export, and the exported kernel's correctness) was verified for
// real, unmodified, end to end.
//
// Usage:
//   npx playwright install chromium   # once
//   node test/webgpu_kernel_tuner_ui.test.mjs

import assert from "node:assert/strict";
import http from "node:http";
import fs, { readFileSync } from "node:fs";
import path, { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const CODEGEN_MANIFEST = JSON.parse(readFileSync(join(HERE, "webgpu_tinygrad_codegen_fixture.json"), "utf8"));
const CONV3D_FIXTURE = CODEGEN_MANIFEST.conv3d;
const TUNING_MANIFEST = JSON.parse(readFileSync(join(HERE, "webgpu_kernel_tuning_fixture.json"), "utf8"));

const SCENARIOS = [
  {
    label: "already-annotated node (webgpu_tinygrad_conv3d.onnx)",
    file: CONV3D_FIXTURE.file,
    nodeName: CONV3D_FIXTURE.nodeName,
    outputName: CONV3D_FIXTURE.outputName,
    inputs: CONV3D_FIXTURE.inputs,
    expectedOutput: CONV3D_FIXTURE.expectedOutput.data,
  },
  {
    label: "plain Conv node with no attached kernel metadata (webgpu_kernel_tuning_fixture.onnx) -- regression",
    file: "webgpu_kernel_tuning_fixture.onnx",
    nodeName: "conv_node",
    outputName: TUNING_MANIFEST.outputName,
    inputs: TUNING_MANIFEST.inputs,
    expectedOutput: TUNING_MANIFEST.expectedOutput.data,
  },
];

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

function serveConvertmodelDir() {
  const server = http.createServer((req, res) => {
    const reqPath = decodeURIComponent(req.url.split("?")[0]);
    const filePath = reqPath === "/" ? join(ROOT, "index.html") : join(ROOT, reqPath);
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end("not found: " + reqPath);
        return;
      }
      const type = {
        ".html": "text/html", ".mjs": "text/javascript", ".js": "text/javascript",
        ".css": "text/css",
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

// Runs one full load/tune/export/verify cycle against the given page (which
// may already have a previous scenario's model loaded -- webgpuKernelsShowBefore
// replaces "before" wholesale, including its tuning state, see setSide's own
// comment) and the shared static server at `port`.
async function runScenario(page, port, scenario) {
  console.log(`\n-- scenario: ${scenario.label} --`);
  console.log(`Loading ${scenario.file} into the panel via window.webgpuKernelsShowBefore...`);
  const modelBytes = new Uint8Array(readFileSync(join(HERE, scenario.file)));
  await page.evaluate(
    ({ bytes, name }) => window.webgpuKernelsShowBefore(new Uint8Array(bytes), name),
    { bytes: Array.from(modelBytes), name: scenario.file },
  );

  const tuneButtonSelector = `#webgpu-kernels-content button[data-action="tune"][data-node="${scenario.nodeName}"]`;
  await check(`[${scenario.label}] the panel shows a 'Tune this kernel' button for the node`, async () => {
    await page.waitForSelector(tuneButtonSelector, { timeout: 10_000 });
  });

  console.log("Clicking 'Tune this kernel…' -- this loads Pyodide + tinygrad's wheel over the network (once), can take a while...");
  await page.click(tuneButtonSelector);

  // Real network + real Pyodide boot + real candidate dispatch loop --
  // generous timeout, same order of magnitude as the other pyodide_*
  // tests' own tinygrad-wheel-fetch-plus-dispatch budget. Cheap on the
  // second scenario since Pyodide/tinygrad are already loaded.
  await page.waitForSelector('#webgpu-kernels-content button[data-action="export"]', { timeout: 180_000 });

  const tuneResult = await page.evaluate(() => window.__wkLastTuneResult);

  await check(`[${scenario.label}] tuning produced more than one candidate`, () => {
    assert.ok(tuneResult, "window.__wkLastTuneResult was never set");
    assert.ok(tuneResult.result.results.length > 1, `expected multiple candidates, got ${tuneResult.result.results.length}`);
  });

  await check(`[${scenario.label}] every candidate's sampled output is finite (no NaN/broken bindings)`, () => {
    for (const r of tuneResult.result.results) {
      assert.ok(r.sampleFinite, `candidate ${r.appliedOpts} produced a non-finite sample`);
    }
  });

  await check(`[${scenario.label}] results are sorted fastest first and the winner is the fastest`, () => {
    const { results, winner } = tuneResult.result;
    for (let i = 1; i < results.length; i++) {
      assert.ok(results[i].medianMs >= results[i - 1].medianMs, "results not sorted ascending by medianMs");
    }
    assert.equal(winner, results[0]);
  });

  console.log("Clicking 'Export tuned model' and capturing the real browser download...");
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.click('#webgpu-kernels-content button[data-action="export"]'),
  ]);
  // Read the downloaded bytes into memory right away -- Playwright's own
  // temp download artifact isn't guaranteed to survive the page closing.
  const exportedBytes = new Uint8Array(readFileSync(await download.path()));

  await check(`[${scenario.label}] the exported file carries the winning candidate's spec`, async () => {
    const { readWebgpuKernelSpecs } = await import("../onnx_node_metadata.mjs");
    const specs = readWebgpuKernelSpecs(exportedBytes);
    const attached = specs.get(scenario.nodeName);
    assert.ok(attached, `no onnxsim.webgpu_kernel metadata found on ${scenario.nodeName} in the exported model`);
    assert.deepEqual(attached, tuneResult.result.winner.spec);
  });

  console.log("Dispatching the exported model's attached (winning) kernel against a real device...");
  const actual = await page.evaluate(
    async ({ port, exportedBytesArray, outputName, inputs, expectedLength }) => {
      const base = `http://localhost:${port}`;
      const { readWebgpuKernelSpecs } = await import(`${base}/onnx_node_metadata.mjs`);
      const { dispatchWebgpuProgram, createStorageBuffer, readBackFloat32Buffer } = await import(
        `${base}/webgpu_kernel_dispatcher.mjs`
      );
      const specs = readWebgpuKernelSpecs(new Uint8Array(exportedBytesArray));
      const spec = [...specs.values()][0];

      const adapter = await navigator.gpu.requestAdapter();
      const device = await adapter.requestDevice();
      const buffersByTensor = new Map();
      for (const [name, { data }] of Object.entries(inputs)) {
        buffersByTensor.set(name, createStorageBuffer(device, Float32Array.from(data)));
      }
      buffersByTensor.set(outputName, createStorageBuffer(device, new Float32Array(expectedLength)));
      await dispatchWebgpuProgram(device, spec, buffersByTensor, {});
      const result = await readBackFloat32Buffer(device, buffersByTensor.get(outputName), expectedLength);
      return Array.from(result);
    },
    {
      port,
      exportedBytesArray: Array.from(exportedBytes),
      outputName: scenario.outputName,
      inputs: scenario.inputs,
      expectedLength: scenario.expectedOutput.length,
    },
  );

  await check(`[${scenario.label}] the exported model's attached (tuned) kernel still computes the right answer`, () => {
    const expected = scenario.expectedOutput;
    let maxAbsDiff = 0;
    for (let i = 0; i < expected.length; i++) {
      maxAbsDiff = Math.max(maxAbsDiff, Math.abs(actual[i] - expected[i]));
    }
    assert.ok(maxAbsDiff < 1e-3, `max abs diff ${maxAbsDiff} too large`);
  });
}

async function main() {
  console.log("WebGPU kernel tuner UI check (real Pyodide + tinygrad + WebGPU)");

  const server = await serveConvertmodelDir();
  const port = server.address().port;
  const browser = await chromium.launch({
    headless: true,
    args: ["--enable-unsafe-webgpu"],
  });

  try {
    const page = await browser.newPage({ acceptDownloads: true });
    page.on("console", (msg) => {
      if (msg.type() === "error") console.log("    [page error]", msg.text());
    });
    await page.goto(`http://localhost:${port}/`);

    for (const scenario of SCENARIOS) {
      await runScenario(page, port, scenario);
    }

    await page.close();
  } finally {
    await browser.close();
    server.close();
  }

  console.log(`\nwebgpu kernel tuner UI: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
