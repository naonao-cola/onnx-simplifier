// Load ONNX models straight from the Hugging Face Hub into the converter page.
//
// The page can already convert a locally-picked file; this module lets it pull
// a model over HTTPS instead, either from the curated regression set (the
// `onnxmodelzoo` org listed in scripts/regression/models.json) or from any repo
// id / .onnx URL the user pastes in. The Hub serves both its JSON API and the
// file `resolve` endpoint with permissive CORS, so a browser `fetch` is all
// that's needed — nothing is proxied through a server.
//
// This module is pure data/networking: it returns model bytes, and index.html
// feeds them into the same conversion path an uploaded file takes.

const HF = "https://huggingface.co";
// Default org for bare names, matching scripts/regression/model_zoo.py.
const ORG = "onnxmodelzoo";

// Where to find the curated model list. The static deploy ships models.json
// next to this page (see .github/workflows/static.yml); the raw GitHub copy is
// a fallback for local dev, where the sibling regression dir isn't served.
const MODEL_LIST_SOURCES = [
  "./models.json",
  "https://raw.githubusercontent.com/onnxsim/onnxsim/master/scripts/regression/models.json",
];

// Fetch the curated regression model ids for the dropdown. Returns [] (rather
// than throwing) when no source is reachable, so the free-text box still works.
export async function loadModelList() {
  for (const src of MODEL_LIST_SOURCES) {
    try {
      const r = await fetch(src);
      if (!r.ok) continue;
      const data = await r.json();
      const ids = (data.models || []).map((m) => m.id).filter(Boolean);
      if (ids.length) return ids;
    } catch {
      // try the next source
    }
  }
  return [];
}

// Turn a user reference into { repo, file?, revision?, url?, name? }.
//
//   "resnet18d_Opset18"                         -> onnxmodelzoo/resnet18d_Opset18
//   "onnxmodelzoo/resnet18d_Opset18"            -> that repo (file discovered)
//   ".../<repo>/resolve/<rev>/<path>.onnx"      -> that exact file
//   ".../<repo>/blob/<rev>/<path>.onnx"         -> that exact file
//   "https://host/…/model.onnx"                 -> that URL verbatim
export function parseRef(ref) {
  ref = (ref || "").trim();
  if (!ref) throw new Error("empty model reference");

  if (/^https?:\/\//i.test(ref)) {
    const u = new URL(ref);
    const isHF = u.hostname === "huggingface.co" || u.hostname.endsWith(".huggingface.co");
    if (isHF) {
      const parts = u.pathname.replace(/^\/+/, "").split("/");
      const [owner, repo, kind, rev, ...rest] = parts;
      if (!owner || !repo) throw new Error(`cannot parse Hugging Face URL: ${ref}`);
      const repoId = `${owner}/${repo}`;
      if ((kind === "resolve" || kind === "blob") && rest.length) {
        return { repo: repoId, file: rest.join("/"), revision: rev || "main" };
      }
      return { repo: repoId };
    }
    // A non-Hub URL: use it as-is and hope it points at an .onnx.
    return { url: ref, name: u.pathname.split("/").pop() || "model.onnx" };
  }

  // Not a URL: a repo id, defaulting a bare name to the onnxmodelzoo org.
  const repo = ref.includes("/") ? ref : `${ORG}/${ref}`;
  return { repo };
}

// Build the Hub `resolve` URL for a file in a repo, percent-encoding each path
// segment (so files in subdirs and with spaces still resolve).
export function fileUrl(repo, file, revision = "main") {
  const encoded = file.split("/").map(encodeURIComponent).join("/");
  return `${HF}/${repo}/resolve/${revision}/${encoded}`;
}

// List a repo's files (with sizes) via the Hub API.
async function repoSiblings(repo) {
  const r = await fetch(`${HF}/api/models/${repo}?blobs=true`);
  if (!r.ok) {
    throw new Error(`Hugging Face API returned HTTP ${r.status} for ${repo}`);
  }
  const info = await r.json();
  return info.siblings || [];
}

// Pick the main .onnx file in a repo (largest, matching the regression workers)
// and warn when the graph relies on external-data blobs the in-browser,
// single-file converter can't load.
async function findOnnxFile(repo, onLog) {
  const siblings = await repoSiblings(repo);
  const onnx = siblings.filter((s) => s.rfilename && s.rfilename.endsWith(".onnx"));
  if (!onnx.length) throw new Error(`no .onnx file found in ${repo}`);
  onnx.sort((a, b) => (b.size || 0) - (a.size || 0));
  const chosen = onnx[0].rfilename;
  if (onnx.length > 1) {
    onLog(`repo has ${onnx.length} .onnx files; using the largest: ${chosen}`);
  }
  const external = siblings.some(
    (s) => s.rfilename && /\.(onnx_data|onnx\.data|data|weight|weights)$/.test(s.rfilename),
  );
  if (external) {
    onLog(
      "note: this repo ships external-data/weight blobs; the in-browser " +
        "converter loads a single .onnx file only, so conversion may fail if " +
        "the graph stores its weights externally.",
    );
  }
  return chosen;
}

// Human-readable byte size, e.g. 4.7 MB. Used in progress messages.
export function humanBytes(n) {
  if (!Number.isFinite(n) || n < 0) return "?";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${i === 0 ? v : v.toFixed(1)} ${units[i]}`;
}

// Read a fetch Response body to a Uint8Array, reporting progress as it streams.
// `onProgress({ loaded, total })` is called per chunk (`total` is 0 when the
// server sends no Content-Length). Falls back to a single arrayBuffer() read
// when the environment has no streaming body.
async function readWithProgress(resp, onProgress) {
  const total = Number(resp.headers && resp.headers.get("content-length")) || 0;
  if (!resp.body || typeof resp.body.getReader !== "function") {
    const bytes = new Uint8Array(await resp.arrayBuffer());
    onProgress({ loaded: bytes.length, total: total || bytes.length });
    return bytes;
  }
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress({ loaded, total });
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

// Resolve a reference and download the model. Returns { bytes, name, repo }.
// `onLog` receives progress lines for the page's console output; `onProgress`
// receives { loaded, total } byte counts as the download streams, for a live
// progress indicator.
export async function fetchModelBytes(ref, onLog = () => {}, onProgress = () => {}) {
  const parsed = parseRef(ref);

  let url;
  let name;
  if (parsed.url) {
    url = parsed.url;
    name = parsed.name;
  } else {
    const revision = parsed.revision || "main";
    let file = parsed.file;
    if (!file) {
      onLog(`querying Hugging Face for the .onnx file in ${parsed.repo}…`);
      file = await findOnnxFile(parsed.repo, onLog);
    }
    url = fileUrl(parsed.repo, file, revision);
    name = file.split("/").pop();
  }

  onLog(`downloading ${url}`);
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`download failed: HTTP ${resp.status} ${resp.statusText}`);
  }
  const bytes = await readWithProgress(resp, onProgress);
  onLog(`downloaded ${name} (${bytes.length.toLocaleString()} bytes)`);
  return { bytes, name, repo: parsed.repo };
}
