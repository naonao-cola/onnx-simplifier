// Unit tests for hf_datasets.mjs — the pure fetch/networking behind the "Run
// inference" panel's real sample-data fill mode. `fetch` and `Math.random`
// are stubbed, so no network or browser is needed.
//
// Usage:
//   node test/hf_datasets.test.mjs

import assert from "node:assert/strict";
import { fetchSampleImageBytes, fetchSampleSentence } from "../hf_datasets.mjs";

let passed = 0;
async function acheck(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

// Installs a fetch stub driven by a list of `(url) => response|null` matchers,
// tried in order, and a fixed Math.random() so the row offset is deterministic.
// Restores both in `finally`.
async function withStubs(matchers, fn) {
  const savedFetch = globalThis.fetch;
  const savedRandom = Math.random;
  Math.random = () => 0; // -> offset 0 in fetchRandomRow's Math.floor(random * rows)
  globalThis.fetch = async (url) => {
    for (const m of matchers) {
      const r = m(url);
      if (r) return r;
    }
    throw new Error(`unexpected fetch: ${url}`);
  };
  try {
    await fn();
  } finally {
    globalThis.fetch = savedFetch;
    Math.random = savedRandom;
  }
}

function jsonResponse(body) {
  return { ok: true, status: 200, json: async () => body };
}

function bytesResponse(bytes) {
  return { ok: true, status: 200, arrayBuffer: async () => bytes.buffer };
}

await acheck("fetchSampleImageBytes fetches a row then its image URL, and maps the label", async () => {
  const imgBytes = new Uint8Array([1, 2, 3, 4]);
  await withStubs(
    [
      (url) =>
        url.startsWith("https://datasets-server.huggingface.co/rows") &&
        url.includes("dataset=frgfm%2Fimagenette")
          ? jsonResponse({ rows: [{ row: { image: { src: "https://example.com/img.jpg" }, label: 4 } }] })
          : null,
      (url) => (url === "https://example.com/img.jpg" ? bytesResponse(imgBytes) : null),
    ],
    async () => {
      const { bytes, label } = await fetchSampleImageBytes();
      assert.deepEqual([...bytes], [1, 2, 3, 4]);
      assert.equal(label, "church"); // class index 4 in the imagenette label list
    },
  );
});

await acheck("fetchSampleImageBytes throws when the row has no image.src", async () => {
  await withStubs(
    [
      (url) =>
        url.startsWith("https://datasets-server.huggingface.co/rows")
          ? jsonResponse({ rows: [{ row: { image: {}, label: 0 } }] })
          : null,
    ],
    async () => {
      await assert.rejects(() => fetchSampleImageBytes(), /no image\.src/);
    },
  );
});

await acheck("fetchSampleImageBytes throws on a non-OK rows response", async () => {
  await withStubs(
    [
      (url) =>
        url.startsWith("https://datasets-server.huggingface.co/rows")
          ? { ok: false, status: 500 }
          : null,
    ],
    async () => {
      await assert.rejects(() => fetchSampleImageBytes(), /HTTP 500/);
    },
  );
});

await acheck("fetchSampleSentence fetches a row and maps the sentiment label", async () => {
  await withStubs(
    [
      (url) =>
        url.startsWith("https://datasets-server.huggingface.co/rows") &&
        url.includes("dataset=stanfordnlp%2Fsst2")
          ? jsonResponse({ rows: [{ row: { sentence: "a solid, well-made film", label: 1 } }] })
          : null,
    ],
    async () => {
      const { text, label } = await fetchSampleSentence();
      assert.equal(text, "a solid, well-made film");
      assert.equal(label, "positive");
    },
  );
});

await acheck("fetchSampleSentence throws when the row has no sentence field", async () => {
  await withStubs(
    [
      (url) =>
        url.startsWith("https://datasets-server.huggingface.co/rows")
          ? jsonResponse({ rows: [{ row: { label: 0 } }] })
          : null,
    ],
    async () => {
      await assert.rejects(() => fetchSampleSentence(), /no sentence field/);
    },
  );
});

console.log(`PASS: ${passed} checks`);
