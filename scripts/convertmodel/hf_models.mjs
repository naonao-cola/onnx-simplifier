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

import { HF_HUB_ESM } from "./cdn.mjs";

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

// Fetch the curated regression model set for the dropdown, as
// [{ id, size }] where `size` is the byte size of the repo's largest .onnx
// (baked into models.json by select_models.py --refresh-sizes) or null when the
// list predates that field. Returns [] (rather than throwing) when no source is
// reachable, so the free-text box still works.
export async function loadModelList() {
  for (const src of MODEL_LIST_SOURCES) {
    try {
      const r = await fetch(src);
      if (!r.ok) continue;
      const data = await r.json();
      const models = (data.models || [])
        .filter((m) => m && m.id)
        .map((m) => ({
          id: m.id,
          size: Number.isFinite(m.size) && m.size > 0 ? m.size : null,
        }));
      if (models.length) return models;
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

// Build a fetch `headers` object carrying a Hub access token, or undefined when
// no token is given, so every call below can just spread `authHeaders(token)`
// without an if/else at each call site. The token is only ever sent to
// huggingface.co (and its subdomains) — see fetchModelBytes / probeModelSize.
function authHeaders(token) {
  return token ? { Authorization: `Bearer ${token}` } : undefined;
}

// List a repo's files (with sizes) via the Hub API. `token` (a Hub access
// token, "hf_…") is sent as a Bearer token so gated / private repos the token
// has access to resolve like any other.
async function repoSiblings(repo, token) {
  const r = await fetch(`${HF}/api/models/${repo}?blobs=true`, { headers: authHeaders(token) });
  if (!r.ok) {
    throw new Error(`Hugging Face API returned HTTP ${r.status} for ${repo}`);
  }
  const info = await r.json();
  return info.siblings || [];
}

// From a repo's sibling listing, pick the main .onnx file (largest, matching the
// regression workers). Returns { sibling, count } where `sibling` carries both
// `rfilename` and `size`; throws when the repo has no .onnx file.
function pickOnnxSibling(siblings, repo) {
  const onnx = siblings.filter((s) => s.rfilename && s.rfilename.endsWith(".onnx"));
  if (!onnx.length) throw new Error(`no .onnx file found in ${repo}`);
  onnx.sort((a, b) => (b.size || 0) - (a.size || 0));
  return { sibling: onnx[0], count: onnx.length };
}

// List a repo's .onnx files as [{ file, size }], largest first — so the UI can
// let the user pick which file to convert instead of always taking the auto-
// detected (largest) one. `size` is a byte count or null when undisclosed.
// Returns [] when the repo has no .onnx file (the caller decides what to do).
export async function listOnnxFiles(repo, token) {
  const siblings = await repoSiblings(repo, token);
  return siblings
    .filter((s) => s.rfilename && s.rfilename.endsWith(".onnx"))
    .map((s) => ({
      file: s.rfilename,
      size: Number.isFinite(s.size) && s.size > 0 ? s.size : null,
    }))
    .sort((a, b) => (b.size || 0) - (a.size || 0));
}

// Pick the main .onnx file in a repo (largest, matching the regression workers)
// and warn when the graph relies on external-data blobs the in-browser,
// single-file converter can't load.
async function findOnnxFile(repo, onLog, token) {
  const siblings = await repoSiblings(repo, token);
  const { sibling, count } = pickOnnxSibling(siblings, repo);
  const chosen = sibling.rfilename;
  if (count > 1) {
    onLog(`repo has ${count} .onnx files; using the largest: ${chosen}`);
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

// Read a URL's byte size from its Content-Length via a HEAD request, without
// downloading the body. Returns a positive integer, or null when the size can't
// be determined (request failed, header absent, or CORS hid it). `token` is
// only meaningful (and should only ever be passed) for a Hugging Face URL —
// callers must not leak it to an arbitrary third-party host.
async function headContentLength(url, token) {
  try {
    const r = await fetch(url, { method: "HEAD", headers: authHeaders(token) });
    if (!r.ok) return null;
    const n = Number(r.headers && r.headers.get("content-length"));
    return Number.isFinite(n) && n > 0 ? n : null;
  } catch {
    return null;
  }
}

// Resolve a reference to the size of the model that would be downloaded, WITHOUT
// fetching the model bytes — so the page can show how big a model is before
// committing to the download. For a Hub repo id this reuses the Hub API listing
// (which carries per-file sizes); for an exact file or a direct URL it issues a
// HEAD request and reads Content-Length. Returns { size, name, repo?, file?,
// url? }, where `size` is a byte count or null when it can't be determined.
// Throws only when the reference itself can't be resolved (empty ref, or a repo
// with no .onnx file). `token` (a Hub access token) is sent only to
// huggingface.co endpoints — never to a direct/non-Hub URL, since `parsed.url`
// is an arbitrary third-party host.
export async function probeModelSize(ref, token) {
  const parsed = parseRef(ref);
  if (parsed.url) {
    return { size: await headContentLength(parsed.url), name: parsed.name, url: parsed.url };
  }
  const revision = parsed.revision || "main";
  if (parsed.file) {
    const url = fileUrl(parsed.repo, parsed.file, revision);
    return {
      size: await headContentLength(url, token),
      name: parsed.file.split("/").pop(),
      repo: parsed.repo,
      file: parsed.file,
    };
  }
  const siblings = await repoSiblings(parsed.repo, token);
  const { sibling } = pickOnnxSibling(siblings, parsed.repo);
  const size = Number.isFinite(sibling.size) && sibling.size > 0 ? sibling.size : null;
  return {
    size,
    name: sibling.rfilename.split("/").pop(),
    repo: parsed.repo,
    file: sibling.rfilename,
  };
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
// progress indicator. `token` (a Hub access token) is sent as a Bearer token
// only when downloading from a Hugging Face repo — never to a direct URL,
// which may point at an arbitrary third-party host.
export async function fetchModelBytes(ref, onLog = () => {}, onProgress = () => {}, token) {
  const parsed = parseRef(ref);

  let url;
  let name;
  let sendToken = false;
  if (parsed.url) {
    url = parsed.url;
    name = parsed.name;
  } else {
    sendToken = true;
    const revision = parsed.revision || "main";
    let file = parsed.file;
    if (!file) {
      onLog(`querying Hugging Face for the .onnx file in ${parsed.repo}…`);
      file = await findOnnxFile(parsed.repo, onLog, token);
    }
    url = fileUrl(parsed.repo, file, revision);
    name = file.split("/").pop();
  }

  onLog(`downloading ${url}`);
  const resp = await fetch(url, { headers: authHeaders(sendToken ? token : undefined) });
  if (!resp.ok) {
    throw new Error(`download failed: HTTP ${resp.status} ${resp.statusText}`);
  }
  const bytes = await readWithProgress(resp, onProgress);
  onLog(`downloaded ${name} (${bytes.length.toLocaleString()} bytes)`);
  return { bytes, name, repo: parsed.repo };
}

// The pinned @huggingface/hub build for the experimental Xet download path
// (HF_HUB_ESM) is declared in cdn.mjs. It is loaded lazily (only when the Xet
// toggle is used) from a CDN, mirroring how the inference panel lazy-loads
// onnxruntime-web.

// Read a Blob (e.g. the XetBlob returned by downloadFile) to a Uint8Array,
// reporting streaming progress against its known size.
async function readBlobWithProgress(blob, onProgress) {
  const total = blob.size || 0;
  if (typeof blob.stream !== "function") {
    const bytes = new Uint8Array(await blob.arrayBuffer());
    onProgress({ loaded: bytes.length, total: total || bytes.length });
    return bytes;
  }
  const reader = blob.stream().getReader();
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

// EXPERIMENTAL: download a model over the Hugging Face Xet protocol via
// @huggingface/hub's downloadFile, which reconstructs the file from
// content-addressed blocks. Pass `fetchImpl` (e.g. the caching fetch from
// xet_cache.mjs) to persist those blocks across loads. Only Hub repos are
// supported (not arbitrary URLs). Returns { bytes, name, repo }; the caller is
// expected to fall back to fetchModelBytes() on any failure (CORS, unsupported).
//
// `concurrency` sets the ceiling on parallel connections XetBlob may open
// while reconstructing the file. >=2.15.0 of @huggingface/hub does this
// adaptively itself (huggingface/huggingface.js#2350) -- it fetches several
// xorb entries at once and tunes the connection count from measured
// throughput -- so this just forwards the ceiling as `parallelDownloads`
// rather than working around a serial-only XetBlob by re-slicing the blob
// into byte-range shards ourselves.
export async function fetchModelBytesXet(
  ref,
  { onLog = () => {}, onProgress = () => {}, fetchImpl, concurrency = 1, token } = {},
) {
  const parsed = parseRef(ref);
  if (!parsed.repo) {
    throw new Error("Xet download needs a Hugging Face repo, not a direct URL");
  }
  const revision = parsed.revision || "main";
  let file = parsed.file;
  if (!file) {
    onLog(`querying Hugging Face for the .onnx file in ${parsed.repo}…`);
    file = await findOnnxFile(parsed.repo, onLog, token);
  }

  const shards = Math.max(1, Math.floor(concurrency) || 1);
  const parallelDownloads = shards > 1 ? { maxConcurrency: shards } : false;
  onLog(
    shards > 1
      ? `downloading ${file} from ${parsed.repo} via Xet (up to ${shards} parallel connections)…`
      : `downloading ${file} from ${parsed.repo} via Xet…`,
  );
  const { downloadFile } = await import(/* @vite-ignore */ HF_HUB_ESM);
  const blob = await downloadFile({
    repo: parsed.repo,
    path: file,
    revision,
    fetch: fetchImpl,
    parallelDownloads,
    // @huggingface/hub's CredentialsParams: pass accessToken only when set, so
    // an anonymous load never sends `accessToken: undefined` through.
    ...(token ? { accessToken: token } : {}),
  });
  if (!blob) throw new Error(`file not found: ${parsed.repo}/${file}`);

  const name = file.split("/").pop();
  const bytes = await readBlobWithProgress(blob, onProgress);
  onLog(`downloaded ${name} (${bytes.length.toLocaleString()} bytes via Xet)`);
  return { bytes, name, repo: parsed.repo };
}
