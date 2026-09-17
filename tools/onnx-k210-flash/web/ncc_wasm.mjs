// In-browser ONNX -> kmodel compiler, driving nncase's own `ncc` CLI
// (onnxsim/nncase's `wasm-k210-build` branch, cross-compiled to wasm32 --
// see ./ncc/README.md) exactly the way its own build script documents:
// write the input into Emscripten's virtual filesystem with
// `Module.FS.writeFile`, run the real CLI via `Module.callMain([...])`,
// read the kmodel back with `Module.FS.readFile`. No NODERAWFS, no host
// filesystem access -- this is the same interface whether the module is
// loaded in a browser tab or under Node (see ./ncc/README.md's own test
// notes for the Node-side verification this was checked against first).
//
// `ncc.js` is loaded via a classic (non-module) <script src="./ncc/ncc.js">
// tag in index.html, matching how scripts/convertmodel/index.html loads
// onnxsim.js: Emscripten's UMD-style output isn't valid ES module syntax
// (its trailing `if (typeof exports === "object" ...)` block is a silent
// no-op under `import`, since `exports`/`module` are never bound in module
// scope), so `window.create_ncc` has to be a plain global, not an import.

/**
 * Compile an ONNX model to a K210 kmodel entirely in-browser.
 *
 * @param {Uint8Array} onnxBytes - ONNX model bytes (run shape inference
 *   first if the model lacks it -- nncase's importer needs `value_info` on
 *   intermediate tensors; the onnxsim converter page already does this).
 * @param {{target?: string, onLog?: (line: string) => void}} [opts]
 *   target: "k210" (default) or "cpu". onLog: called with each line of
 *   ncc's own stdout/stderr, in order, for a live progress log.
 * @returns {Promise<Uint8Array>} the compiled kmodel's bytes.
 */
export async function compileToKmodel(onnxBytes, { target = "k210", onLog } = {}) {
  if (typeof window.create_ncc !== "function") {
    throw new Error('ncc.js not loaded -- add <script src="./ncc/ncc.js"></script> before this module runs');
  }

  let stderr = "";
  const Module = await window.create_ncc({
    noInitialRun: true,
    print: (line) => onLog?.(line),
    printErr: (line) => {
      stderr += line + "\n";
      onLog?.(line);
    },
  });

  Module.FS.writeFile("/input.onnx", onnxBytes);
  const args = ["compile", "-i", "onnx", "-t", target, "/input.onnx", "/output.kmodel"];

  let exitCode;
  try {
    exitCode = Module.callMain(args);
  } catch (e) {
    // An uncaught C++ exception (e.g. a malformed/unshaped ONNX input)
    // surfaces here as a bare pointer integer, not a JS Error -- decode it
    // with the exported getExceptionMessage helper (see ../README.md's
    // build notes for why this build exports it).
    const msg = typeof e === "number" ? Module.getExceptionMessage(e) : String(e?.message ?? e);
    throw new Error(`ncc compile threw: ${msg}${stderr ? `\n${stderr}` : ""}`);
  }
  if (exitCode !== 0) {
    throw new Error(`ncc compile exited with status ${exitCode}${stderr ? `\n${stderr}` : ""}`);
  }

  return Module.FS.readFile("/output.kmodel");
}
