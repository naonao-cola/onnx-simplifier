// Wires the "Load from Hugging Face" panel on the converter page.
//
// It populates the dropdown from the curated regression set, resolves whatever
// the user picked or typed to model bytes (hf_models.mjs), and then hands those
// bytes to the same conversion path an uploaded file uses. index.html publishes
// that entry point as `window.__onnxsimStartConversion` and enables the controls
// below (via `maybeEnableInput`) only once both WASM runtimes are ready.

import { loadModelList, fetchModelBytes } from "./hf_models.mjs";

const select = document.getElementById("hf-model-select");
const refInput = document.getElementById("hf-model-input");
const loadBtn = document.getElementById("hf-load-button");
const statusEl = document.getElementById("hf-status");
const fileInput = document.getElementById("file-input");
const logOutput = document.getElementById("log-output");

const log = (msg) => {
  if (!logOutput) return;
  logOutput.value += msg + "\n";
  logOutput.scrollTop = logOutput.scrollHeight;
};

// Fill the dropdown with the curated onnxmodelzoo set. The list is advisory —
// the free-text box accepts any repo id or .onnx URL regardless.
loadModelList().then((ids) => {
  if (!select) return;
  if (!ids.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(model list unavailable — type a repo id or URL)";
    select.appendChild(opt);
    return;
  }
  for (const id of ids) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = id.replace(/^onnxmodelzoo\//, "");
    select.appendChild(opt);
  }
});

// Picking from the dropdown mirrors into the text box so the two never disagree
// about what "Load" will fetch, and the user can tweak the id before loading.
if (select && refInput) {
  select.addEventListener("change", () => {
    if (select.value) refInput.value = select.value;
  });
}

async function doLoad() {
  const ref = ((refInput && refInput.value) || (select && select.value) || "").trim();
  if (!ref) {
    log("enter a Hugging Face repo id or .onnx URL, or pick one from the list");
    return;
  }
  if (typeof window.__onnxsimStartConversion !== "function") {
    log("WebAssembly runtime not ready yet — try again in a moment");
    return;
  }

  loadBtn.disabled = true;
  if (fileInput) fileInput.disabled = true;
  if (statusEl) statusEl.textContent = "loading…";
  try {
    log(`loading model from Hugging Face: ${ref}`);
    const { bytes, name } = await fetchModelBytes(ref, log);
    // Publish as the "original" model so the Run inference panel can run it —
    // there is no file-input entry for a Hugging Face download.
    window.__onnxsimOriginal = { bytes, name };
    // Hand a *copy* to the worker: the buffer is transferred (detached) on
    // postMessage, so copying keeps `bytes` intact for the inference panel.
    const copy = bytes.slice();
    window.__onnxsimStartConversion(name, copy.buffer);
    if (statusEl) statusEl.textContent = "";
  } catch (e) {
    log("Hugging Face load failed: " + (e && e.message ? e.message : String(e)));
    if (statusEl) statusEl.textContent = "failed — see console output";
    // The conversion never started, so re-enable the file picker here (normally
    // the worker's convert-done message does that).
    if (fileInput) fileInput.disabled = false;
  } finally {
    loadBtn.disabled = false;
  }
}

if (loadBtn) loadBtn.addEventListener("click", doLoad);
if (refInput) {
  refInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      doLoad();
    }
  });
}
