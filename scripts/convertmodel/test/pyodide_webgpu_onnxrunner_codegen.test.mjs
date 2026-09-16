// Goes one step further than pyodide_webgpu_single_node_codegen.test.mjs:
// that test proved a single hand-picked op (Conv) could be regenerated live
// by re-deriving its shapes/attributes in JS and driving a hand-translated
// copy of onnxsim.webgpu_tinygrad_codegen.generate_conv_kernel. This test
// instead drives **tinygrad's own generic ONNX importer**
// (tinygrad.nn.onnx.OnnxRunner) directly on a whole multi-node model's raw
// bytes, inside Pyodide -- no onnxsim-specific per-op translation, and no
// JS-side shape/attribute reader at all: OnnxRunner has its own hand-rolled
// protobuf parser (no dependency on the "onnx" Python package) and drives
// tinygrad's own (much larger) op table generically, so this exercises a
// real multi-op graph (Conv3D -> Relu, fused by tinygrad's scheduler into
// ONE kernel) including an op (Relu) onnxsim has no bespoke codegen for.
//
// webgpu_onnxrunner_conv_relu.onnx / webgpu_onnxrunner_fixture.json (see
// make_webgpu_onnxrunner_fixture.py) carry the model, real x/w values, the
// expected output from onnx.reference.ReferenceEvaluator, and a "ground
// truth" WebGPU kernel spec generated the same way (OnnxRunner + the same
// _lower_tensor_program lowering), but offline in native Python with
// *different* random dummy data -- kernel generation only depends on
// shapes, never on tensor values, so an exact match against Pyodide's own
// live-generated spec (built from yet another random seed) is real evidence
// this isn't a coincidence of matching inputs.
//
// Two checks, same structure as pyodide_webgpu_single_node_codegen.test.mjs:
//   1. (Node only, no browser/GPU) the Pyodide-generated spec deepEquals the
//      fixture's offline "ground truth" spec.
//   2. (Playwright/Chromium, real WebGPU device) the *live* spec is
//      dispatched via webgpu_kernel_dispatcher.mjs against the fixture's
//      real x/w values, checked against ReferenceEvaluator's real output.
//
// A real device backend (WEBGPU or otherwise) is never touched inside
// Pyodide itself -- same as every other pyodide_*.test.mjs here -- so
// OnnxRunner's own op *translation* correctness (independent of whether
// WGSL rendering/dispatch is right) is checked separately, natively, in
// make_webgpu_onnxrunner_fixture.py (which compares its own tinygrad run,
// on tinygrad's default CPU-ish device, against ReferenceEvaluator before
// ever generating the "ground truth" spec at all).
//
// Usage:
//   npx playwright install chromium   # once
//   node test/pyodide_webgpu_onnxrunner_codegen.test.mjs

import assert from "node:assert/strict";
import http from "node:http";
import fs, { readFileSync } from "node:fs";
import path, { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { loadPyodide } from "pyodide";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const MANIFEST = JSON.parse(readFileSync(join(HERE, "webgpu_onnxrunner_fixture.json"), "utf8"));
const TINYGRAD_VERSION = "0.14.0";

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

const FETCH_TIMEOUT_MS = 60_000;
function fetchWithTimeout(url) {
  return fetch(url, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) });
}

async function fetchTinygradWheel() {
  const meta = await fetchWithTimeout(`https://pypi.org/pypi/tinygrad/${TINYGRAD_VERSION}/json`).then((r) =>
    r.json(),
  );
  const wheelInfo = meta.urls.find((u) => u.packagetype === "bdist_wheel");
  if (!wheelInfo) {
    throw new Error(`no wheel (bdist_wheel) found for tinygrad==${TINYGRAD_VERSION} on PyPI`);
  }
  return fetchWithTimeout(wheelInfo.url).then((r) => r.arrayBuffer());
}

// Same _lower_tensor_program translation as
// pyodide_webgpu_single_node_codegen.test.mjs's own embedded copy and
// make_webgpu_onnxrunner_fixture.py's -- see either's docstring for why
// each of these three copies exists independently rather than sharing one
// module (this one runs inside Pyodide, that one runs offline in native
// Python to produce the fixture, and onnxsim's own copy runs server-side).
const ONNXRUNNER_CODEGEN_PY = `
import json
import random
import re
from tinygrad.codegen import to_program
from tinygrad.helpers import Context, Target
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.uop.ops import Ops

_ANSI_RE = re.compile(r"\\x1b\\[[0-9;]*m")


def _rand(*shape):
    if len(shape) == 1:
        return [random.gauss(0, 1) for _ in range(shape[0])]
    return [_rand(*shape[1:]) for _ in range(shape[0])]


def _encode_float(v):
    if v != v:
        return "NaN"
    if v == float("inf"):
        return "Infinity"
    if v == float("-inf"):
        return "-Infinity"
    return v


def _base_buffer(t):
    for u in t.uop.toposort():
        if u.op is Ops.BUFFER:
            return u
    return None


def _lower_tensor_program(named_tensors, output_name):
    output = named_tensors[output_name]
    linear = output.schedule_linear()
    kernel_calls = [u for u in linear.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.SINK]
    if not kernel_calls:
        raise RuntimeError("tinygrad scheduled no compute kernel -- the graph may have constant-folded away")

    uop_to_name = {}
    for name, t in named_tensors.items():
        buf = _base_buffer(t)
        if buf is None:
            raise RuntimeError(f"tensor {name!r} has no underlying BUFFER after scheduling")
        uop_to_name[buf] = name

    renderer = WGSLRenderer(Target())
    steps = []
    intermediate_names = {}
    intermediate_bytes = {}
    next_intermediate = [0]

    def _name_for(buf_uop):
        name = uop_to_name.get(buf_uop)
        if name is not None:
            return "tensor", name
        name = intermediate_names.get(buf_uop)
        if name is None:
            name = f"_intermediate_{next_intermediate[0]}"
            next_intermediate[0] += 1
            intermediate_names[buf_uop] = name
            intermediate_bytes[name] = buf_uop.size()[0] * buf_uop.dtype.itemsize
        return "intermediate", name

    for call in kernel_calls:
        ast = call.src[0]
        buffer_uops = list(call.src[1:])
        prg = to_program(ast, renderer)
        info = prg.arg
        source_uop = next(s for s in prg.src if s.op is Ops.SOURCE)
        wgsl = source_uop.arg
        entry_point = _ANSI_RE.sub("", info.function_name)

        bindings = [{"group": 0, "binding": 0, "access": "uniform", "constant": [_encode_float(float("inf"))]}]
        for slot, buf_uop in enumerate(buffer_uops):
            kind, name = _name_for(buf_uop)
            if kind == "tensor":
                bindings.append({"group": 0, "binding": slot + 1, "access": "read_write", "tensor": name})
            else:
                bindings.append({"group": 0, "binding": slot + 1, "access": "read_write", "intermediate": name})

        padded_size = [int(x) for x in info.global_size] + [1, 1, 1]
        steps.append(
            {
                "wgsl": wgsl,
                "entry_point": entry_point,
                "dispatch": padded_size[:3],
                "bindings": bindings,
            }
        )

    return {"steps": steps, "intermediates": intermediate_bytes}


def generate_whole_graph_spec(model_path, input_name, input_shape, output_name):
    random.seed(2)  # deliberately different from make_webgpu_onnxrunner_fixture.py's seed
    with Context(DEV="WEBGPU"):
        # Even this import must happen inside the Context, not just the
        # tensors this function builds -- tinygrad.nn.onnx evaluates a
        # stray Tensor(0) default argument (for ConvInteger) at *module
        # import* time, which resolves tinygrad's device default like any
        # other tensor construction, and Pyodide has no usable tinygrad
        # device at all (no GPU, no native compiler for a CPU backend)
        # outside of this context -- see make_webgpu_onnxrunner_fixture.py's
        # own docstring for where this was first found.
        from tinygrad import Tensor
        from tinygrad.nn.onnx import OnnxRunner

        run = OnnxRunner(model_path)
        initializer_names = [k for k in run.graph_values if k != ""]
        outputs = run({input_name: Tensor(_rand(*input_shape))})
        named = {name: run.graph_values[name] for name in [*initializer_names, input_name]}
        named[output_name] = outputs[output_name]
        return _lower_tensor_program(named, output_name)


json.dumps(generate_whole_graph_spec(MODEL_PATH, INPUT_NAME, INPUT_SHAPE, OUTPUT_NAME))
`;

async function generateSpecInPyodide({ modelBytes, inputName, inputShape, outputName }) {
  const pyodide = await loadPyodide();
  const wheelBuffer = await fetchTinygradWheel();
  pyodide.unpackArchive(wheelBuffer, "zip", { extractDir: "/tinygrad_pkg" });
  await pyodide.runPythonAsync('import sys; sys.path.insert(0, "/tinygrad_pkg")');

  // OnnxRunner reads a model from a real (virtual) filesystem path -- unlike
  // the single-node test, there's no need to hand it shapes/attributes
  // separately, since it parses the whole file itself.
  pyodide.FS.writeFile("/model.onnx", modelBytes);

  pyodide.globals.set("MODEL_PATH", "/model.onnx");
  pyodide.globals.set("INPUT_NAME", inputName);
  pyodide.globals.set("INPUT_SHAPE", pyodide.toPy(inputShape));
  pyodide.globals.set("OUTPUT_NAME", outputName);
  const raw = await pyodide.runPythonAsync(ONNXRUNNER_CODEGEN_PY);
  return JSON.parse(raw);
}

// Same minimal static file server as the other browser tests here.
function serveConvertmodelDir() {
  const server = http.createServer((req, res) => {
    const reqPath = decodeURIComponent(req.url.split("?")[0]);
    if (reqPath === "/") {
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end("<!doctype html><title>pyodide onnxrunner webgpu codegen</title>");
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
async function dispatchInPage({ port, outputName, liveSpec, inputs, expectedLength }) {
  const base = `http://localhost:${port}`;
  const { dispatchWebgpuProgram, createStorageBuffer, readBackFloat32Buffer } = await import(
    `${base}/webgpu_kernel_dispatcher.mjs`
  );

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const buffersByTensor = new Map();
  for (const [name, { data }] of Object.entries(inputs)) {
    buffersByTensor.set(name, createStorageBuffer(device, Float32Array.from(data)));
  }
  buffersByTensor.set(outputName, createStorageBuffer(device, new Float32Array(expectedLength)));

  await dispatchWebgpuProgram(device, liveSpec, buffersByTensor);

  const actual = await readBackFloat32Buffer(device, buffersByTensor.get(outputName), expectedLength);
  return Array.from(actual);
}

async function main() {
  console.log("Whole-graph live WebGPU kernel generation (Pyodide + tinygrad's own ONNX importer) check\n");

  const modelBytes = new Uint8Array(readFileSync(join(HERE, MANIFEST.file)));
  const inputShape = MANIFEST.inputs[MANIFEST.inputName].shape;

  console.log("Generating the whole-graph kernel live, inside Pyodide, via tinygrad.nn.onnx.OnnxRunner...");
  const liveSpec = await generateSpecInPyodide({
    modelBytes,
    inputName: MANIFEST.inputName,
    inputShape,
    outputName: MANIFEST.outputName,
  });
  console.log(`  got ${liveSpec.steps.length} step(s)\n`);

  await check("live-generated spec has the same step count as the offline ground-truth spec", () => {
    assert.equal(liveSpec.steps.length, MANIFEST.groundTruthSpec.steps.length);
  });

  await check(
    "live-generated WGSL/entry_point/dispatch/bindings exactly match the offline ground-truth spec (different random data on each side)",
    () => {
      assert.deepEqual(liveSpec, MANIFEST.groundTruthSpec);
    },
  );

  console.log("\nDispatching the *live* spec against a real WebGPU device...");
  const server = await serveConvertmodelDir();
  const port = server.address().port;
  const browser = await chromium.launch({
    headless: true,
    // Verified sufficient on its own in this repo's own dev sandbox (via
    // SwiftShader's software Vulkan path); see webgpu_hf_demo.test.mjs's own
    // comment. A real GPU, as CI runners with one have, needs nothing more.
    args: ["--enable-unsafe-webgpu"],
  });

  let actual;
  try {
    const page = await browser.newPage();
    await page.goto(`http://localhost:${port}/`);
    actual = await page.evaluate(dispatchInPage, {
      port,
      outputName: MANIFEST.outputName,
      liveSpec,
      inputs: MANIFEST.inputs,
      expectedLength: MANIFEST.expectedOutput.data.length,
    });
    await page.close();
  } finally {
    await browser.close();
    server.close();
  }

  await check("the live-generated whole-graph kernel's real GPU output matches onnx.reference.ReferenceEvaluator", () => {
    const expected = MANIFEST.expectedOutput.data;
    let maxAbsDiff = 0;
    for (let i = 0; i < expected.length; i++) {
      maxAbsDiff = Math.max(maxAbsDiff, Math.abs(actual[i] - expected[i]));
    }
    assert.ok(
      maxAbsDiff < 1e-3,
      `max abs diff ${maxAbsDiff} too large -- actual ${JSON.stringify(actual)} vs expected ${JSON.stringify(expected)}`,
    );
  });

  console.log(`\npyodide onnxrunner whole-graph webgpu codegen: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
