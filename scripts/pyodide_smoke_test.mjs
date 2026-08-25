#!/usr/bin/env node
// Loads the wasm32-emscripten `onnxsim_cpp2py_export.abi3.so` produced by
// `build_wasm_pyodide.sh` inside a real Pyodide runtime (via the `pyodide`
// npm package, running under Node) and confirms it actually imports and
// runs, THEN goes further: installs the real `onnx` PyPI wasm wheel via
// micropip and runs a full `onnxsim.simplify()` end to end. The second part
// only works because this workflow specifically targets Pyodide 0.29.4
// (ABI epoch 2025_0) -- the same epoch PyPI's only `onnx` wasm wheel is
// tagged for. See docs/wasm_pyodide.md for how that was found; earlier
// versions of this script (and the toolchain it ran against) could only
// reach the first phase below, because the newer Pyodide release then in
// use (314.0.5, epoch 2026_0) built cleanly but produced a module ABI-
// incompatible with onnx's only published wheel.
//
// Usage: node pyodide_smoke_test.mjs <path-to-onnxsim_cpp2py_export.abi3.so>
import { loadPyodide } from "pyodide";
import { readFileSync, existsSync, readdirSync, statSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function copyDirIntoFS(pyodide, src, dst) {
  pyodide.FS.mkdirTree(dst);
  for (const entry of readdirSync(src)) {
    const s = join(src, entry);
    const d = `${dst}/${entry}`;
    if (statSync(s).isDirectory()) {
      copyDirIntoFS(pyodide, s, d);
    } else if (entry.endsWith(".py")) {
      pyodide.FS.writeFile(d, readFileSync(s));
    }
  }
}

const MODULE_NAME = "onnxsim_cpp2py_export";

function fail(message) {
  console.error(`\nFAIL: ${message}\n`);
  process.exit(1);
}

async function main() {
  const soPathArg = process.argv[2];
  if (!soPathArg) {
    fail(`usage: node pyodide_smoke_test.mjs <path-to-${MODULE_NAME}.abi3.so>`);
  }
  const soPath = resolve(soPathArg);
  if (!existsSync(soPath)) {
    fail(`built module not found at ${soPath}`);
  }
  const soBytes = readFileSync(soPath);
  console.log(`read ${soBytes.length} bytes from ${soPath}`);

  const pyodide = await loadPyodide();
  console.log(`Pyodide ${pyodide.version} ready (loaded via the 'pyodide' npm package)`);

  // Find the real site-packages directory rather than assuming a path --
  // it depends on the exact Pyodide release's Python minor version, and
  // guessing it wrong would fail for a reason unrelated to what's actually
  // being tested.
  const sitePackages = pyodide.runPython(`
import site
_candidates = [p for p in site.getsitepackages() if p.endswith("site-packages")]
_candidates[0] if _candidates else None
`);
  if (!sitePackages) {
    fail("could not find a site-packages directory in this Pyodide's sys.path (site.getsitepackages() returned nothing usable)");
  }
  console.log(`site-packages: ${sitePackages}`);

  const destPath = `${sitePackages}/${MODULE_NAME}.abi3.so`;
  try {
    pyodide.FS.writeFile(destPath, soBytes);
  } catch (e) {
    fail(`writing ${destPath} into Pyodide's virtual FS failed: ${e.stack || e}`);
  }
  console.log(`wrote built module to ${destPath} inside Pyodide's virtual FS`);

  // Extension-module dlopen under wasm32-emscripten needs the async loading
  // path (Pyodide may need to fetch/instantiate the side module's own wasm
  // and resolve its GOT entries against the main Pyodide module), so this
  // must run through runPythonAsync, not the synchronous runPython.
  const pythonCode = `
import sys
import ${MODULE_NAME} as m

print("sys.path:", sys.path, file=sys.stderr)
print("imported module:", m, file=sys.stderr)

# Exercise one real, argument-free binding -- not just "did it import" --
# to confirm the module is actually functional, not just loadable. This is
# a plain module-level function (nanobind's m.def(...) chains directly off
# the module object here, not off a class), returning the list of
# fuse/elimination passes onnxsim's statically-linked onnx-optimizer
# registers. A non-empty result means the extension initialized and linked
# correctly against its statically-linked onnx-optimizer/onnx/protobuf/
# abseil dependencies, not merely that the .so file loaded.
optimizers = m._list_optimizers()
assert isinstance(optimizers, list), f"_list_optimizers() returned {optimizers!r}, expected a list"
assert len(optimizers) > 0, "_list_optimizers() returned an empty list -- module loaded but onnx-optimizer's passes did not register"
print(f"OK: {len(optimizers)} optimizer passes registered, e.g. {optimizers[:3]}", file=sys.stderr)
`;
  try {
    await pyodide.runPythonAsync(pythonCode);
  } catch (e) {
    fail(`import ${MODULE_NAME} (or calling _list_optimizers()) failed under Pyodide:\n${e.stack || e}`);
  }

  console.log(`\nPASS: ${MODULE_NAME} imports and runs under Pyodide ${pyodide.version}\n`);

  // --- Phase 2: the real ABI-compatibility proof --------------------------
  // Install onnx from PyPI for real (micropip resolves its numpy/protobuf/
  // ml_dtypes dependencies from Pyodide's own package repo automatically),
  // add onnxsim's pure-Python package on top of the extension already
  // loaded above, and run a real onnxsim.simplify() call.
  const abiVersion = pyodide.runPython(
    `import sysconfig; sysconfig.get_config_var("PYODIDE_ABI_VERSION")`
  );
  console.log(`Pyodide ABI epoch: ${abiVersion}`);

  await pyodide.loadPackage("micropip");
  try {
    await pyodide.runPythonAsync(`
import micropip
await micropip.install(["onnx", "rich"])
`);
  } catch (e) {
    fail(
      `micropip.install(["onnx", "rich"]) failed under Pyodide ${pyodide.version} ` +
        `(ABI epoch ${abiVersion}) -- if this Pyodide version's epoch no longer matches ` +
        `PyPI's onnx wheel, see docs/wasm_pyodide.md for how to find the version whose ` +
        `epoch does:\n${e.stack || e}`
    );
  }
  console.log(`micropip.install(["onnx", "rich"]): OK`);

  copyDirIntoFS(pyodide, join(REPO_ROOT, "onnxsim"), `${sitePackages}/onnxsim`);
  console.log("copied onnxsim's pure-Python package into site-packages");

  const simplifyCode = `
import onnx
from onnx import helper, TensorProto

node = helper.make_node("Identity", ["x"], ["y"])
graph = helper.make_graph(
    [node],
    "smoke-test-graph",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8

import onnxsim
simplified, check_ok = onnxsim.simplify(model)
assert check_ok, "onnxsim.simplify() reported check_ok=False"
print(f"OK: onnxsim.simplify() ran, {len(simplified.graph.node)} node(s) in the result")
`;
  try {
    await pyodide.runPythonAsync(simplifyCode);
  } catch (e) {
    fail(`full onnxsim.simplify() failed under Pyodide:\n${e.stack || e}`);
  }

  console.log(
    `\nPASS: full onnxsim.simplify() (with real onnx) runs under Pyodide ${pyodide.version}\n`
  );
}

main().catch((e) => {
  fail(`unexpected error:\n${e.stack || e}`);
});
