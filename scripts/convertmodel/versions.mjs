// Fills the "Versions" panel on the converter page and publishes the versions
// for the "Report an issue" body, so bug reports say exactly which onnxsim /
// onnx-optimizer / onnx / protobuf the WebAssembly module was built against.
//
// The numbers come from the module itself: `onnxsim_versions()` (interface.cpp)
// reports the onnxsim and onnx-optimizer versions baked in by CMake, plus the
// onnx IR/opset and protobuf versions read from the linked libraries. We call it
// on this page's runtime (the same one that populates the pass list), which
// needs no onnxruntime-web, so the panel fills as soon as the runtime is ready.

// Normalize the embind-returned object into a plain JS object of strings.
export function normalizeVersions(v) {
  if (!v) return null;
  return {
    onnxsim: v.onnxsim ?? "unknown",
    onnx_optimizer: v.onnx_optimizer ?? "unknown",
    onnx: v.onnx ?? "unknown",
    protobuf: v.protobuf ?? "unknown",
  };
}

// The human labels shown in the panel and the issue report, in display order.
const LABELS = [
  ["onnxsim", "onnxsim"],
  ["onnx_optimizer", "onnx-optimizer"],
  ["onnx", "onnx"],
  ["protobuf", "protobuf"],
];

// Render the versions object into `panel` as a row of labelled badges.
export function renderVersions(panel, versions) {
  if (!panel) return;
  panel.textContent = "";
  for (const [key, label] of LABELS) {
    const badge = document.createElement("span");
    badge.className = "version-badge";
    const k = document.createElement("span");
    k.className = "version-k";
    k.textContent = label;
    const val = document.createElement("span");
    val.className = "version-v";
    val.textContent = versions[key];
    badge.appendChild(k);
    badge.appendChild(val);
    panel.appendChild(badge);
  }
}

async function initVersions() {
  const panel = document.getElementById("versions-panel");
  try {
    // Set by index.html's inline loader as soon as create_onnxsim() is called.
    const runtime = await window.__onnxsimRuntimePromise;
    const versions = normalizeVersions(runtime.onnxsim_versions());
    // Publish for the issue-report builder (read in index.html).
    window.__onnxsimVersions = versions;
    renderVersions(panel, versions);
  } catch (err) {
    if (panel) {
      panel.textContent = "could not read versions: " + (err && err.message ? err.message : err);
    }
  }
}

// Only run in the browser; the exports above stay importable by the Node test.
if (typeof window !== "undefined" && typeof document !== "undefined") {
  initVersions();
}
