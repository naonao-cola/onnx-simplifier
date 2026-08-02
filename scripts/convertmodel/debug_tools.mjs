// Browser glue for the "Parse a text graph" panel on the converter page.
//
// It parses the ONNX textual representation (the form onnx.parser.parse_graph /
// parse_model accept) into a model with the module's `onnxsim_parse_graph`, then
// drives it through the same Simplify / visualize path as an uploaded file. The
// single-pass debugging transforms (shape inference, data propagation, constant
// folding) are exposed as options of the converter's mode radios, not here.
//
// Parsing needs no model executor, so it runs on this page's runtime directly.

import { downloadBytes } from "./download.mjs";
import { syncInputUrl } from "./query_params.mjs";

// Resolve this page's WASM runtime (published by index.html's inline loader).
function getRuntime() {
  return window.__onnxsimRuntimePromise;
}

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
      // Show the parsed model in the "Before" Netron pane and publish it as the
      // inference panel's "original" source.
      if (window.netronShowBefore) window.netronShowBefore(bytes, name);
      window.__onnxsimOriginal = { bytes, name };
      // Reflect the graph text in the address bar so the link is shareable.
      syncInputUrl("graph", text);
      // Optionally run it straight through the converter using the currently
      // selected mode (a fresh copy, since startConversion transfers and
      // detaches the buffer it is given).
      if ((!convertChk || convertChk.checked) && window.__onnxsimStartConversion) {
        window.__onnxsimStartConversion(name, bytes.slice().buffer);
      }
    } catch (err) {
      setStatus("parse error: " + (err && err.message ? err.message : err));
    } finally {
      btn.disabled = false;
    }
  });
}

initParsePanel();
