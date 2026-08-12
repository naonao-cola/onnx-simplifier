// Fetch real sample inputs (an image, a sentence) from Hugging Face datasets
// for the "Run inference" panel's "sample data" fill mode — a real-looking
// counterpart to input_fill.mjs's synthetic random/ones/zeros/arange modes.
//
// Like hf_models.mjs, this module is pure data/networking: it returns bytes
// (or text) and a human-readable label, never touching the DOM. The Hub's
// public Dataset Viewer API (datasets-server.huggingface.co) serves both the
// row JSON and (for images) a short-lived signed asset URL with permissive
// CORS — the same endpoint HF's own embeddable dataset-viewer widget uses — so
// a browser `fetch` is all that's needed, mirroring how hf_models.mjs pulls
// model files straight from the Hub with no server-side proxy.
//
// The image URLs the rows API returns are signed and expire, so callers should
// use the bytes immediately rather than caching the URL itself; fetchSample*()
// always does a fresh row lookup, never memoizes a picked row.

const ROWS_API = "https://datasets-server.huggingface.co/rows";

// frgfm/imagenette (Apache-2.0): 10 easily-recognized ImageNet classes (tench,
// church, chainsaw, ...). The 160px config keeps the fetched JPEG small. Row
// count is the known size of 160px/validation (checked via the Hub dataset
// viewer) — only used to bound the random offset, so it doesn't need to track
// the dataset exactly.
const IMAGE_DATASET = { dataset: "frgfm/imagenette", config: "160px", split: "validation", rows: 3900 };

// stanfordnlp/sst2: short single-sentence movie-review snippets. Row count is
// the known size of default/validation.
const TEXT_DATASET = { dataset: "stanfordnlp/sst2", config: "default", split: "validation", rows: 872 };

function rowsUrl({ dataset, config, split }, offset, length = 1) {
  const p = new URLSearchParams({ dataset, config, split, offset: String(offset), length: String(length) });
  return `${ROWS_API}?${p.toString()}`;
}

// Fetch a single random row from a dataset config/split. Returns the row's
// `row` object (the dataset's own column shape) or throws on a network/HTTP
// failure.
async function fetchRandomRow(source) {
  const offset = Math.floor(Math.random() * Math.max(1, source.rows));
  const url = rowsUrl(source, offset, 1);
  const r = await fetch(url);
  if (!r.ok) {
    throw new Error(`Hugging Face dataset viewer returned HTTP ${r.status} for ${url}`);
  }
  const data = await r.json();
  const row = data.rows && data.rows[0] && data.rows[0].row;
  if (!row) throw new Error(`no rows returned for ${source.dataset} (${source.config}/${source.split})`);
  return row;
}

// Fetch one random sample image from frgfm/imagenette. Returns
// { bytes: Uint8Array, label: string } where `label` is the class name (e.g.
// "church"), for a log line describing what was fed to the model.
export async function fetchSampleImageBytes() {
  const row = await fetchRandomRow(IMAGE_DATASET);
  const src = row.image && row.image.src;
  const labelNames = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
  ];
  const label = labelNames[row.label] || `class ${row.label}`;
  if (!src) throw new Error("sample row had no image.src");
  const imgResp = await fetch(src);
  if (!imgResp.ok) {
    throw new Error(`failed to download sample image: HTTP ${imgResp.status}`);
  }
  const bytes = new Uint8Array(await imgResp.arrayBuffer());
  return { bytes, label };
}

// Fetch one random sample sentence from stanfordnlp/sst2. Returns
// { text: string, label: string } where `label` is "positive"/"negative".
export async function fetchSampleSentence() {
  const row = await fetchRandomRow(TEXT_DATASET);
  const text = row.sentence;
  if (!text) throw new Error("sample row had no sentence field");
  const label = row.label === 1 ? "positive" : row.label === 0 ? "negative" : `label ${row.label}`;
  return { text, label };
}
