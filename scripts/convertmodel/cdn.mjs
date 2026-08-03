// Single source of truth for every third-party CDN the converter page loads
// *executable code* (or an embeddable app) from at runtime.
//
// The page's external runtime dependencies used to be declared inline in each
// consumer — the onnxruntime-web URL alone was copied into three files
// (inference_browser.mjs, backend_test.mjs, worker.js) with "keep in sync"
// comments, so a version bump in one and not the others would silently mismatch
// the JS glue against the .wasm binaries. Centralizing them here means the UI's
// external dependencies live in one auditable place: one file to see what the
// page trusts, one constant to bump a version, one file to self-host from.
//
// Only executable / embedded dependencies belong here. Everything else the page
// fetches over the network is *data* (models, backend test cases, model lists)
// pulled from Hugging Face / GitHub per feature; that stays with its feature.
//
//   * onnxruntime-web  (cdn.jsdelivr.net)  — inference + the ORT-web build's
//                                            constant folding
//   * @huggingface/hub (esm.sh)            — experimental Xet download (lazy)
//   * Perfetto UI      (ui.perfetto.dev)   — optional embedded trace viewer
//
// Pure constants only (no window/document access) so the Node tests can import
// and assert on them.

// --- onnxruntime-web -------------------------------------------------------
// The JS bundles and the matching wasm binaries are pulled from the same
// versioned dist/ directory (see ORT_BASE), so the glue and the .wasm can never
// disagree. Bump this one constant to move the whole page to a new
// onnxruntime-web.
export const ORT_VERSION = "1.27.0";

// Base URL of that version's dist/ directory. Callers append the bundle name
// (e.g. `${ORT_BASE}ort.min.mjs`) and point `ort.env.wasm.wasmPaths` at it so
// the wasm artifacts come from the same directory as the glue.
export const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

// --- @huggingface/hub ------------------------------------------------------
// Pinned build for the experimental Xet download path, loaded lazily (only when
// the Xet toggle is used) via dynamic import.
export const HF_HUB_VERSION = "2.14.4";
export const HF_HUB_ESM = `https://esm.sh/@huggingface/hub@${HF_HUB_VERSION}`;

// --- Perfetto UI -----------------------------------------------------------
// The trace viewer can embed the full Perfetto UI (or open it in a new tab) for
// deep analysis, driven over Perfetto's postMessage embedding protocol.
export const PERFETTO_ORIGIN = "https://ui.perfetto.dev";
// `mode=embedded` hides Perfetto's sidebar for a clean in-page view.
export const PERFETTO_EMBED_SRC = `${PERFETTO_ORIGIN}/#!/?mode=embedded`;
