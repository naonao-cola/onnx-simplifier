// Proves tinygrad's own Tensor -> UOp -> WGSL codegen pipeline (the same
// one onnxsim.webgpu_tinygrad_codegen drives server-side) actually runs
// inside Pyodide -- a real Python-in-WebAssembly runtime, the same kind
// this repo's own WASM converter UI is -- answering the open question of
// whether tinygrad could ever run client-side rather than only as an
// offline codegen step.
//
// No browser/GPU needed here at all: Pyodide runs standalone via its own
// bundled WASM build (the "pyodide" npm package), so this is plain Node,
// unlike every other webgpu_*.test.mjs file in this directory.
//
// Deliberately avoids numpy and micropip entirely, fetching tinygrad's own
// wheel straight from PyPI (files.pythonhosted.org) and unpacking it into
// Pyodide's virtual filesystem by hand:
//   - numpy has no pure-Python wheel on PyPI (it needs a real wasm32 build,
//     which only Pyodide's own package index carries) -- but tinygrad
//     itself doesn't actually need numpy for this: building a Tensor from
//     plain (possibly nested) Python lists instead of a numpy array, and
//     never calling .numpy()/.realize(), keeps numpy out of sys.modules
//     for the whole codegen path, verified directly below.
//   - micropip (Pyodide's own installer) would otherwise fetch itself from
//     Pyodide's package CDN, an extra network dependency this test doesn't
//     need just to install one pure-Python wheel.
// The result: this only ever needs pypi.org/files.pythonhosted.org
// reachable, the same as any other pip install -- no Pyodide-CDN
// dependency, which also makes it robust to environments (like sandboxes
// with restrictive egress allowlists) that block that CDN specifically.
//
// This does NOT prove onnxsim.webgpu_tinygrad_codegen's own generate_*
// functions run unmodified inside Pyodide -- those still import numpy at
// module level (for convenience: random dummy data, reading a real
// initializer's bytes via onnx.numpy_helper) and have not themselves been
// run here. What this proves is the underlying tinygrad codegen machinery
// those functions build on -- Tensor construction, schedule_linear,
// to_program, WGSLRenderer -- has no fundamental barrier to running
// client-side, for both an elementwise case and the harder Conv3D case
// (the gap onnxsim.webgpu_target.check_webgpu_conv3d_support flags).
//
// Usage:
//   node test/pyodide_tinygrad_codegen.test.mjs

import assert from "node:assert/strict";
import { loadPyodide } from "pyodide";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

const TINYGRAD_VERSION = "0.14.0";

// node's fetch() has no default timeout -- a stalled connection (proxy
// hiccup, PyPI slow to respond) would otherwise hang this test forever
// instead of failing with a clear error.
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

// Runs the shared setup (build a WEBGPU-tagged Tensor graph from plain
// Python lists, schedule it, render every resulting kernel to WGSL) for one
// case, returning a JSON-serializable summary. `pythonExpr` builds the
// output Tensor from `x`/`w` (already-bound Tensors); kept as a Python
// expression string, evaluated inside Pyodide, so each case only differs in
// this one line.
async function runCodegenCase(pyodide, { xShape, wShape, outputExpr }) {
  const script = `
import sys, random, json
from tinygrad import Tensor
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.uop.ops import Ops

def rand(*shape):
    if len(shape) == 1:
        return [random.gauss(0, 1) for _ in range(shape[0])]
    return [rand(*shape[1:]) for _ in range(shape[0])]

random.seed(0)
x = Tensor(rand(*${JSON.stringify(xShape)}), device="WEBGPU")
w = Tensor(rand(*${JSON.stringify(wShape)}), device="WEBGPU") if ${wShape ? "True" : "False"} else None
y = ${outputExpr}

linear = y.schedule_linear()
kernel_calls = [u for u in linear.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.SINK]
if not kernel_calls:
    raise RuntimeError("no compute kernel scheduled -- the graph constant-folded away")

renderer = WGSLRenderer(Target())
wgsl_sources = []
for call in kernel_calls:
    prg = to_program(call.src[0], renderer)
    source_uop = next(s for s in prg.src if s.op is Ops.SOURCE)
    wgsl_sources.append(source_uop.arg)

json.dumps({
    "kernel_count": len(kernel_calls),
    "wgsl_sources": wgsl_sources,
    "numpy_imported": "numpy" in sys.modules,
})
`;
  const raw = await pyodide.runPythonAsync(script);
  return JSON.parse(raw);
}

async function main() {
  console.log(`Pyodide + tinygrad WGSL codegen check (tinygrad==${TINYGRAD_VERSION})\n`);

  console.log("Loading Pyodide (bundled stdlib, no network needed for this step)...");
  const pyodide = await loadPyodide();
  console.log(`  Pyodide ${pyodide.version} ready`);

  console.log("Fetching tinygrad's wheel from PyPI and unpacking it into Pyodide's own filesystem...");
  const wheelBuffer = await fetchTinygradWheel();
  pyodide.unpackArchive(wheelBuffer, "zip", { extractDir: "/tinygrad_pkg" });
  await pyodide.runPythonAsync('import sys; sys.path.insert(0, "/tinygrad_pkg")');
  console.log(`  unpacked ${wheelBuffer.byteLength} bytes\n`);

  console.log("Elementwise Add (the simplest real case)...");
  const addResult = await runCodegenCase(pyodide, {
    xShape: [4],
    wShape: [4],
    outputExpr: "x + w",
  });
  check("schedules to exactly one kernel", () => {
    assert.equal(addResult.kernel_count, 1);
  });
  check("renders real WGSL with a @compute entry point", () => {
    assert.ok(addResult.wgsl_sources[0].includes("@compute"));
  });
  check("never imports numpy", () => {
    assert.equal(addResult.numpy_imported, false);
  });

  console.log("\nConv (3-D spatial rank -- the gap onnxsim.webgpu_target.check_webgpu_conv3d_support flags)...");
  const convResult = await runCodegenCase(pyodide, {
    xShape: [1, 2, 4, 4, 4],
    wShape: [2, 2, 3, 3, 3],
    outputExpr: "x.conv2d(w)", // despite the name, generalizes to any spatial rank -- see onnxsim/webgpu_tinygrad_codegen.py's own docstring
  });
  check("Conv3D schedules to a real kernel too", () => {
    assert.ok(convResult.kernel_count >= 1);
    assert.ok(convResult.wgsl_sources[0].includes("@compute"));
  });
  check("Conv3D codegen also never imports numpy", () => {
    assert.equal(convResult.numpy_imported, false);
  });

  console.log(`\npyodide tinygrad codegen: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
