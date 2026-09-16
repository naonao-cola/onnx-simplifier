// Proves the "single node, live" step beyond pyodide_tinygrad_codegen.test.mjs:
// not just that tinygrad's codegen pipeline *can* run inside Pyodide, but that
// it can *regenerate the same real kernel* onnxsim.webgpu_tinygrad_codegen's
// own generate_conv_kernel already produces server-side -- for one flagged
// node in a real (fixture) .onnx model, reading that node's own shapes and
// attributes straight out of the model bytes in JS (never touching Python's
// onnx package or numpy), driving tinygrad-in-Pyodide with them, and getting
// back a WebgpuKernelSpec-shaped program.
//
// Two checks, in order:
//
//   1. (Node only, no browser/GPU) The Pyodide-generated spec is compared,
//      field for field, against the spec onnxsim.webgpu_tinygrad_codegen
//      already attached to webgpu_tinygrad_conv3d.onnx (read back via
//      onnx_node_metadata.mjs's readWebgpuKernelSpecs, the exact same
//      committed fixture webgpu_tinygrad_codegen.test.mjs itself dispatches).
//      Kernel generation only depends on shapes/attributes, never on the
//      concrete tensor *values* used to build the tinygrad Tensor graph (see
//      onnxsim.webgpu_tinygrad_codegen's own module docstring) -- so an
//      exact match here (same WGSL text, same entry point, same dispatch,
//      same bindings) is a strong signal this is the identical kernel, not
//      just "some kernel that runs".
//
//   2. (Playwright/Chromium, real WebGPU device) The *live, Pyodide-generated*
//      spec -- not the one already embedded in the fixture -- is dispatched
//      via webgpu_kernel_dispatcher.mjs against the fixture's own concrete
//      x/w values, and the GPU output is checked against
//      webgpu_tinygrad_codegen_fixture.json's expectedOutput (from
//      onnx.reference.ReferenceEvaluator running the real ONNX node). This is
//      the real proof: it doesn't just look like the right kernel, it
//      computes the right answer on a real GPU.
//
// Together, these show the "generate a kernel for one flagged node, live,
// client-side" step onnxsim/webgpu_tinygrad_codegen.py's docstring calls out
// as not yet built is achievable for the simplest real case (Conv, all
// attributes at their defaults) without needing onnx-the-Python-package or
// numpy inside Pyodide at all -- see docs/webgpu-kernel-dispatch.md for what
// remains before this generalizes past one node/one op.
//
// Usage:
//   npx playwright install chromium   # once
//   node test/pyodide_webgpu_single_node_codegen.test.mjs

import assert from "node:assert/strict";
import http from "node:http";
import fs, { readFileSync } from "node:fs";
import path, { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { loadPyodide } from "pyodide";
import { readConvNodeInfo } from "../onnx_conv_node_reader.mjs";
import { readWebgpuKernelSpecs } from "../onnx_node_metadata.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const CODEGEN_MANIFEST = JSON.parse(readFileSync(join(HERE, "webgpu_tinygrad_codegen_fixture.json"), "utf8"));
const CONV_FIXTURE = CODEGEN_MANIFEST.conv3d;
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

// A numpy-free, onnx-free re-implementation of
// onnxsim.webgpu_tinygrad_codegen.generate_conv_kernel + its own
// _lower_tensor_program helper -- see that module's own source for the
// original this was translated from line by line. It never imports numpy or
// onnx (the caller supplies shapes/attributes already extracted from the
// model in JS -- see onnx_conv_node_reader.mjs), so it runs unmodified
// inside Pyodide, matching pyodide_tinygrad_codegen.test.mjs's own
// numpy-avoidance technique (plain nested Python lists instead of numpy
// arrays as Tensor leaves).
const CONV_CODEGEN_PY = `
import json
import random
import re
from tinygrad import Tensor
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
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


def _rename_tensor_bindings(spec, rename):
    for step in spec["steps"]:
        for b in step["bindings"]:
            if "tensor" in b and b["tensor"] in rename:
                b["tensor"] = rename[b["tensor"]]
    return spec


def generate_conv_kernel_spec(node_info):
    random.seed(0)
    x_shape = node_info["xShape"]
    w_shape = node_info["wShape"]
    spatial_rank = len(w_shape) - 2
    strides = node_info["strides"]
    dilations = node_info["dilations"]
    group = node_info["group"]
    pads = node_info["pads"]
    pads_begin, pads_end = pads[:spatial_rank], pads[spatial_rank:]

    x = Tensor(_rand(*x_shape), device="WEBGPU")
    w = Tensor(_rand(*w_shape), device="WEBGPU")
    named = {"__x": x, "__w": w}
    if pads_begin != [0] * spatial_rank or pads_end != [0] * spatial_rank:
        pad_pairs = [None, None] + [(b, e) for b, e in zip(pads_begin, pads_end)]
        x = x.pad(pad_pairs)

    b_shape = node_info.get("bShape")
    if b_shape:
        b = Tensor(_rand(*b_shape), device="WEBGPU")
        named["__b"] = b
    else:
        b = None

    y = x.conv2d(w, bias=b, groups=group, stride=strides, dilation=dilations, padding=0)
    named["__y"] = y

    spec = _lower_tensor_program(named, "__y")
    rename = {"__x": node_info["xName"], "__w": node_info["wName"], "__y": node_info["outputName"]}
    if node_info.get("bName"):
        rename["__b"] = node_info["bName"]
    return _rename_tensor_bindings(spec, rename)


json.dumps(generate_conv_kernel_spec(json.loads(NODE_INFO_JSON)))
`;

async function generateSpecInPyodide(nodeInfo) {
  const pyodide = await loadPyodide();
  const wheelBuffer = await fetchTinygradWheel();
  pyodide.unpackArchive(wheelBuffer, "zip", { extractDir: "/tinygrad_pkg" });
  await pyodide.runPythonAsync('import sys; sys.path.insert(0, "/tinygrad_pkg")');
  pyodide.globals.set("NODE_INFO_JSON", JSON.stringify(nodeInfo));
  const raw = await pyodide.runPythonAsync(CONV_CODEGEN_PY);
  return JSON.parse(raw);
}

// Same minimal static file server as the other browser tests here.
function serveConvertmodelDir() {
  const server = http.createServer((req, res) => {
    const reqPath = decodeURIComponent(req.url.split("?")[0]);
    if (reqPath === "/") {
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end("<!doctype html><title>pyodide single-node webgpu codegen</title>");
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
async function dispatchInPage({ port, file, outputName, liveSpec, inputs, expectedLength }) {
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
  console.log("Single-node live WebGPU kernel generation (Pyodide + tinygrad) check\n");

  console.log(`Reading node ${JSON.stringify(CONV_FIXTURE.nodeName)}'s shapes/attributes straight from ${CONV_FIXTURE.file} (JS, no onnx package)...`);
  const modelBytes = new Uint8Array(readFileSync(join(HERE, CONV_FIXTURE.file)));
  const nodeInfo = readConvNodeInfo(modelBytes, CONV_FIXTURE.nodeName);
  console.log(`  ${JSON.stringify(nodeInfo)}\n`);

  console.log("Generating the kernel live, inside Pyodide...");
  const liveSpec = await generateSpecInPyodide(nodeInfo);
  console.log(`  got ${liveSpec.steps.length} step(s)\n`);

  const offlineSpecs = readWebgpuKernelSpecs(modelBytes);
  const offlineSpec = offlineSpecs.get(CONV_FIXTURE.nodeName);

  await check("a kernel spec for the node exists in the fixture to compare against", () => {
    assert.ok(offlineSpec, `no onnxsim.webgpu_kernel metadata found on node ${CONV_FIXTURE.nodeName}`);
  });

  await check("live-generated spec has the same step count as the offline one", () => {
    assert.equal(liveSpec.steps.length, offlineSpec.steps.length);
  });

  await check("live-generated WGSL/entry_point/dispatch/bindings exactly match the offline (Python-generated) spec", () => {
    assert.deepEqual(liveSpec, offlineSpec);
  });

  console.log("\nDispatching the *live* spec (not the fixture's own) against a real WebGPU device...");
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
      file: CONV_FIXTURE.file,
      outputName: CONV_FIXTURE.outputName,
      liveSpec,
      inputs: CONV_FIXTURE.inputs,
      expectedLength: CONV_FIXTURE.expectedOutput.data.length,
    });
    await page.close();
  } finally {
    await browser.close();
    server.close();
  }

  await check("the live-generated kernel's real GPU output matches onnx.reference.ReferenceEvaluator", () => {
    const expected = CONV_FIXTURE.expectedOutput.data;
    let maxAbsDiff = 0;
    for (let i = 0; i < expected.length; i++) {
      maxAbsDiff = Math.max(maxAbsDiff, Math.abs(actual[i] - expected[i]));
    }
    assert.ok(
      maxAbsDiff < 1e-3,
      `max abs diff ${maxAbsDiff} too large -- actual ${JSON.stringify(actual)} vs expected ${JSON.stringify(expected)}`,
    );
  });

  console.log(`\npyodide single-node webgpu codegen: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
