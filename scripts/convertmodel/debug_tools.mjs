// Browser glue for two debugging panels on the converter page:
//
//   1. "Parse a text graph" — parse the ONNX textual representation (the form
//      onnx.parser.parse_graph / parse_model accept) into a model with the
//      module's `onnxsim_parse_graph`, then drive it through the same
//      Simplify/visualize path as an uploaded file. Also makes the parsed model
//      the source for the single-feature passes below.
//
//   2. "Run a single feature" — run exactly one of the transforms Simplify
//      otherwise drives to a fixed point (shape inference, ONNX data
//      propagation, or constant folding) once, so its isolated effect can be
//      inspected. The pass runs in the conversion worker (constant folding needs
//      the same model executor Simplify uses), and the result is shown in the
//      "After" Netron pane and offered as a download.
//
// Parsing needs no model executor, so it runs on this page's runtime directly;
// the single-feature passes go through the worker.

import { downloadBytes } from "./download.mjs";

// Resolve this page's WASM runtime (published by index.html's inline loader).
function getRuntime() {
  return window.__onnxsimRuntimePromise;
}

// Resolve the conversion worker, waiting for index.html to create it if needed.
function getWorker() {
  return new Promise((resolve) => {
    if (window.__onnxsimWorker) {
      resolve(window.__onnxsimWorker);
      return;
    }
    window.addEventListener(
      "onnxsim:worker-ready",
      () => resolve(window.__onnxsimWorker),
      { once: true },
    );
  });
}

// Resolve once BOTH WASM runtimes are up (index.html dispatches "onnxsim:ready"
// from maybeEnableInput). A pass posted before the worker's runtime has
// registered its message listener would be silently dropped, so the pass
// buttons wait for this first.
function whenReady() {
  return new Promise((resolve) => {
    if (window.__onnxsimReady) {
      resolve();
      return;
    }
    window.addEventListener("onnxsim:ready", () => resolve(), { once: true });
  });
}

// The model the single-feature passes operate on: an uploaded/parsed/loaded
// model, or a previous pass's output (so passes can be chained). Kept as a copy
// so the worker transfer of a per-run duplicate never detaches it.
let debugSource = null; // { bytes: Uint8Array, name: string }

function setDebugSource(bytes, name) {
  debugSource = { bytes, name };
  const label = document.getElementById("debug-source-name");
  if (label) label.textContent = name ? `current model: ${name}` : "no model loaded yet";
}

// ---- Parse a text graph ----------------------------------------------------

async function initParsePanel() {
  const textEl = document.getElementById("parse-graph-text");
  const btn = document.getElementById("parse-graph-button");
  const convertChk = document.getElementById("parse-graph-convert");
  const statusEl = document.getElementById("parse-graph-status");
  const dlBtn = document.getElementById("parse-graph-download");
  if (!btn) return;

  let lastBytes = null;
  const setStatus = (msg) => {
    if (statusEl) statusEl.textContent = msg;
  };

  btn.addEventListener("click", async () => {
    const text = (textEl && textEl.value) || "";
    if (!text.trim()) {
      setStatus("Enter an ONNX text graph first.");
      return;
    }
    btn.disabled = true;
    setStatus("parsing…");
    try {
      const runtime = await getRuntime();
      const res = runtime.onnxsim_parse_graph(text);
      if (!res || res.error) {
        setStatus("parse failed: " + (res ? res.error : "unknown error"));
        return;
      }
      // res.model is a view into the wasm heap — copy it out immediately.
      const bytes = new Uint8Array(res.model);
      lastBytes = bytes;
      const name = "parsed_graph.onnx";
      setStatus(`parsed OK (${bytes.length} bytes).`);
      if (dlBtn) {
        dlBtn.style.display = "";
        dlBtn.onclick = () => downloadBytes(lastBytes, name);
      }
      // Show the parsed model in the "Before" Netron pane and make it the
      // single-feature source.
      if (window.netronShowBefore) window.netronShowBefore(bytes, name);
      setDebugSource(bytes, name);
      // Optionally run it straight through the converter (a fresh copy, since
      // startConversion transfers and detaches the buffer it is given).
      if ((!convertChk || convertChk.checked) && window.__onnxsimStartConversion) {
        window.__onnxsimOriginal = { bytes, name };
        window.__onnxsimStartConversion(name, bytes.slice().buffer);
      }
    } catch (err) {
      setStatus("parse error: " + (err && err.message ? err.message : err));
    } finally {
      btn.disabled = false;
    }
  });
}

// ---- Run a single feature (debug passes) -----------------------------------

async function initSinglePanel() {
  const fileInput = document.getElementById("debug-file-input");
  const useConvertedBtn = document.getElementById("debug-use-converted");
  const thresholdInput = document.getElementById("debug-tensor-size-threshold");
  const out = document.getElementById("debug-output");
  const dlBtn = document.getElementById("debug-download");
  const buttons = {
    infer_shapes: document.getElementById("debug-infer-shapes"),
    data_propagation: document.getElementById("debug-data-propagation"),
    fold_constant: document.getElementById("debug-fold-constant"),
  };
  if (!buttons.infer_shapes) return;

  const log = (msg) => {
    if (!out) return;
    out.value += msg + "\n";
    out.scrollTop = out.scrollHeight;
  };

  // Its own file input keeps the debug source independent of the converter's
  // transfer-and-detach upload flow.
  if (fileInput) {
    fileInput.addEventListener("change", async (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const bytes = new Uint8Array(await file.arrayBuffer());
      setDebugSource(bytes, file.name);
      log(`loaded ${file.name} (${bytes.length} bytes) as the debug source`);
    });
  }

  if (useConvertedBtn) {
    useConvertedBtn.addEventListener("click", () => {
      const c = window.__onnxsimConverted;
      if (!c || !c.bytes) {
        log("no converted model yet — run a conversion above first.");
        return;
      }
      setDebugSource(new Uint8Array(c.bytes), c.name || "converted.onnx");
      log(`using the converted result (${c.name || "converted.onnx"}) as the debug source`);
    });
  }

  const setBusy = (busy) => {
    for (const b of Object.values(buttons)) if (b) b.disabled = busy;
  };

  let lastResult = null; // { bytes, name }
  if (dlBtn) {
    dlBtn.onclick = () => {
      if (lastResult) downloadBytes(lastResult.bytes, lastResult.name);
    };
  }

  const PASS_LABELS = {
    infer_shapes: "shape inference",
    data_propagation: "data propagation",
    fold_constant: "constant folding",
  };

  const runPass = async (type) => {
    if (!debugSource) {
      log("no model loaded — pick a file (or use the converted result / a parsed graph) first.");
      return;
    }
    setBusy(true);
    log(`running ${PASS_LABELS[type]} on ${debugSource.name}…`);
    try {
      // Wait until the worker's runtime is initialized so the pass message is
      // not dropped, then grab the worker handle.
      await whenReady();
      const worker = await getWorker();
      // Send a fresh copy of the source bytes; the worker transfers the buffer.
      const buf = debugSource.bytes.slice().buffer;
      const threshold = parseInt(thresholdInput ? thresholdInput.value : "", 10);
      const done = new Promise((resolve) => {
        const onMessage = (e) => {
          if (!e.data || e.data[0] !== "debug-done" || e.data[1] !== type) return;
          worker.removeEventListener("message", onMessage);
          resolve({ dataUrl: e.data[2], error: e.data[3] });
        };
        worker.addEventListener("message", onMessage);
      });
      const args =
        type === "fold_constant"
          ? [type, buf, Number.isFinite(threshold) ? threshold : 1000000000]
          : [type, buf];
      worker.postMessage(args, [buf]);
      const { dataUrl, error } = await done;
      if (error || !dataUrl) {
        log(`${PASS_LABELS[type]} failed: ${error || "no model produced"}`);
        return;
      }
      const resultBytes = dataUrlToBytes(dataUrl);
      const name = `${debugSource.name.replace(/\.onnx$/i, "")}.${type}.onnx`;
      lastResult = { bytes: resultBytes, name };
      if (dlBtn) dlBtn.style.display = "";
      // Show the result in the shared "After" Netron pane and make it the next
      // debug source so passes can be chained (e.g. shape-infer then fold).
      if (window.netronShowAfter) window.netronShowAfter(dataUrl, name);
      setDebugSource(resultBytes, name);
      log(`${PASS_LABELS[type]} done → ${name} (${resultBytes.length} bytes). Shown in the "After" Netron pane; download available.`);
    } catch (err) {
      log(`${PASS_LABELS[type]} error: ${err && err.message ? err.message : err}`);
    } finally {
      setBusy(false);
    }
  };

  for (const [type, btn] of Object.entries(buttons)) {
    if (btn) btn.addEventListener("click", () => runPass(type));
  }
}

// Decode a "data:...;base64,<b64>" URL into a Uint8Array.
function dataUrlToBytes(dataUrl) {
  const b64 = dataUrl.slice(dataUrl.indexOf(",") + 1);
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

initParsePanel();
initSinglePanel();
