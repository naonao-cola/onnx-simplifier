#!/usr/bin/env node
// Loads the wasm32-emscripten `onnxsim_cpp2py_export.abi3.so` produced by
// `build_wasm_pyodide.sh` inside a real Pyodide runtime (via the `pyodide`
// npm package, running under Node) and confirms it actually imports and
// runs -- the integration coverage docs/wasm_pyodide.md's "What's next"
// section calls out as missing (only structural checks -- `file`, a
// `dylink.0` section, the exported `PyInit_...` symbol -- had been done
// before this script existed).
//
// Deliberately narrow in scope: this imports only the low-level nanobind
// extension module (`onnxsim_cpp2py_export`), NOT `onnxsim` itself (the
// pure-Python package). Importing `onnxsim` would additionally require
// `onnx` and `rich` to be present inside the Pyodide environment, which is
// a separate concern from what this build step produces and needs verified
// here: does the compiled extension module actually load and initialize
// under Pyodide's dlopen.
//
// Usage: node pyodide_smoke_test.mjs <path-to-onnxsim_cpp2py_export.abi3.so>
import { loadPyodide } from "pyodide";
import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";

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
}

main().catch((e) => {
  fail(`unexpected error:\n${e.stack || e}`);
});
