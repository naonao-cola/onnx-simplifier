// Unit test for the pure helpers behind the converter page's "Load from
// Hugging Face" panel. `fetch` is stubbed, so no network or browser is needed.
//
// Usage:
//   node test/hf_models.test.mjs

import assert from "node:assert/strict";
import {
  parseRef,
  fileUrl,
  loadModelList,
  fetchModelBytes,
  probeModelSize,
  readBlobSharded,
  humanBytes,
} from "../hf_models.mjs";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

async function acheck(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

// Install a fetch stub that maps URL -> a route descriptor:
//   { ok?, status?, statusText?, json?, arrayBuffer?, chunks?, contentLength? }
// A `chunks` array (of Uint8Array) yields a streaming ReadableStream-style body
// so the progress path can be exercised; otherwise only arrayBuffer() is served.
function withFetch(routes, fn) {
  const saved = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const r = routes[url];
    if (!r) {
      return { ok: false, status: 404, statusText: "Not Found", headers: headersOf() };
    }
    const resp = {
      ok: r.ok !== false,
      status: r.status ?? 200,
      statusText: r.statusText ?? "OK",
      headers: headersOf(r.contentLength),
      json: async () => r.json,
      arrayBuffer: async () => r.arrayBuffer ?? new ArrayBuffer(0),
    };
    if (r.chunks) resp.body = streamOf(r.chunks);
    return resp;
  };
  return Promise.resolve(fn()).finally(() => {
    globalThis.fetch = saved;
  });
}

function headersOf(contentLength) {
  const map = {};
  if (contentLength != null) map["content-length"] = String(contentLength);
  return { get: (k) => map[k.toLowerCase()] ?? null };
}

// Minimal getReader()-style body over a fixed list of chunks.
function streamOf(chunks) {
  let i = 0;
  return {
    getReader: () => ({
      read: async () =>
        i < chunks.length
          ? { done: false, value: chunks[i++] }
          : { done: true, value: undefined },
    }),
  };
}

check("bare name defaults to the onnxmodelzoo org", () => {
  assert.deepEqual(parseRef("resnet18d_Opset18"), {
    repo: "onnxmodelzoo/resnet18d_Opset18",
  });
});

check("an explicit owner/repo is kept as-is", () => {
  assert.deepEqual(parseRef("microsoft/resnet-50"), {
    repo: "microsoft/resnet-50",
  });
});

check("a Hub repo URL parses to just the repo", () => {
  assert.deepEqual(
    parseRef("https://huggingface.co/onnxmodelzoo/resnet18d_Opset18"),
    { repo: "onnxmodelzoo/resnet18d_Opset18" },
  );
});

check("a Hub resolve URL parses to repo + file + revision", () => {
  assert.deepEqual(
    parseRef("https://huggingface.co/onnxmodelzoo/foo/resolve/main/sub/model.onnx"),
    { repo: "onnxmodelzoo/foo", file: "sub/model.onnx", revision: "main" },
  );
});

check("a Hub blob URL is treated like resolve", () => {
  assert.deepEqual(
    parseRef("https://huggingface.co/o/r/blob/v2/m.onnx"),
    { repo: "o/r", file: "m.onnx", revision: "v2" },
  );
});

check("a non-Hub URL is used verbatim", () => {
  assert.deepEqual(parseRef("https://example.com/models/net.onnx"), {
    url: "https://example.com/models/net.onnx",
    name: "net.onnx",
  });
});

check("an empty reference is rejected", () => {
  assert.throws(() => parseRef("   "), /empty model reference/);
});

check("fileUrl percent-encodes each path segment", () => {
  assert.equal(
    fileUrl("o/r", "a dir/model v2.onnx"),
    "https://huggingface.co/o/r/resolve/main/a%20dir/model%20v2.onnx",
  );
});

await acheck("loadModelList reads ./models.json first, carrying sizes", async () => {
  await withFetch(
    {
      "./models.json": {
        json: {
          models: [
            { id: "onnxmodelzoo/a", size: 1234 },
            { id: "onnxmodelzoo/b" },
          ],
        },
      },
    },
    async () => {
      const models = await loadModelList();
      assert.deepEqual(models, [
        { id: "onnxmodelzoo/a", size: 1234 },
        { id: "onnxmodelzoo/b", size: null },
      ]);
    },
  );
});

await acheck("loadModelList returns [] when no source is reachable", async () => {
  await withFetch({}, async () => {
    assert.deepEqual(await loadModelList(), []);
  });
});

await acheck("fetchModelBytes discovers the largest .onnx in a repo", async () => {
  const api = "https://huggingface.co/api/models/onnxmodelzoo/foo?blobs=true";
  const big = "https://huggingface.co/onnxmodelzoo/foo/resolve/main/big.onnx";
  await withFetch(
    {
      [api]: {
        json: {
          siblings: [
            { rfilename: "small.onnx", size: 10 },
            { rfilename: "big.onnx", size: 999 },
            { rfilename: "readme.md", size: 1 },
          ],
        },
      },
      [big]: { arrayBuffer: new Uint8Array([1, 2, 3, 4]).buffer },
    },
    async () => {
      const { bytes, name } = await fetchModelBytes("foo");
      assert.equal(name, "big.onnx");
      assert.deepEqual(Array.from(bytes), [1, 2, 3, 4]);
    },
  );
});

await acheck("fetchModelBytes downloads an exact resolve URL directly", async () => {
  const url = "https://huggingface.co/o/r/resolve/main/m.onnx";
  await withFetch(
    { [url]: { arrayBuffer: new Uint8Array([9, 9]).buffer } },
    async () => {
      const { bytes, name } = await fetchModelBytes(url);
      assert.equal(name, "m.onnx");
      assert.deepEqual(Array.from(bytes), [9, 9]);
    },
  );
});

await acheck("fetchModelBytes surfaces a repo with no .onnx", async () => {
  const api = "https://huggingface.co/api/models/onnxmodelzoo/empty?blobs=true";
  await withFetch(
    { [api]: { json: { siblings: [{ rfilename: "readme.md", size: 1 }] } } },
    async () => {
      await assert.rejects(fetchModelBytes("empty"), /no \.onnx file found/);
    },
  );
});

await acheck("fetchModelBytes surfaces a failed download", async () => {
  const url = "https://huggingface.co/o/r/resolve/main/m.onnx";
  await withFetch(
    { [url]: { ok: false, status: 403, statusText: "Forbidden" } },
    async () => {
      await assert.rejects(fetchModelBytes(url), /HTTP 403/);
    },
  );
});

check("humanBytes formats sizes", () => {
  assert.equal(humanBytes(0), "0 B");
  assert.equal(humanBytes(512), "512 B");
  assert.equal(humanBytes(1536), "1.5 KB");
  assert.equal(humanBytes(5 * 1024 * 1024), "5.0 MB");
  assert.equal(humanBytes(-1), "?");
});

await acheck("fetchModelBytes streams chunks and reports progress", async () => {
  const url = "https://huggingface.co/o/r/resolve/main/m.onnx";
  await withFetch(
    {
      [url]: {
        contentLength: 6,
        chunks: [new Uint8Array([1, 2, 3]), new Uint8Array([4, 5, 6])],
      },
    },
    async () => {
      const progress = [];
      const { bytes } = await fetchModelBytes(url, () => {}, (p) => progress.push(p));
      // Bytes are reassembled in order across chunks.
      assert.deepEqual(Array.from(bytes), [1, 2, 3, 4, 5, 6]);
      // Progress is reported per chunk, with the total from Content-Length.
      assert.deepEqual(progress, [
        { loaded: 3, total: 6 },
        { loaded: 6, total: 6 },
      ]);
    },
  );
});

await acheck("fetchModelBytes falls back to arrayBuffer without a body", async () => {
  const url = "https://huggingface.co/o/r/resolve/main/m.onnx";
  await withFetch(
    { [url]: { arrayBuffer: new Uint8Array([7, 8]).buffer } },
    async () => {
      const progress = [];
      const { bytes } = await fetchModelBytes(url, () => {}, (p) => progress.push(p));
      assert.deepEqual(Array.from(bytes), [7, 8]);
      // One final progress tick; total defaults to the byte count.
      assert.deepEqual(progress, [{ loaded: 2, total: 2 }]);
    },
  );
});

await acheck("probeModelSize reports the largest .onnx size from the Hub API", async () => {
  const api = "https://huggingface.co/api/models/onnxmodelzoo/foo?blobs=true";
  await withFetch(
    {
      [api]: {
        json: {
          siblings: [
            { rfilename: "small.onnx", size: 10 },
            { rfilename: "big.onnx", size: 999 },
            { rfilename: "readme.md", size: 1 },
          ],
        },
      },
    },
    async () => {
      const info = await probeModelSize("foo");
      assert.equal(info.size, 999);
      assert.equal(info.name, "big.onnx");
      assert.equal(info.repo, "onnxmodelzoo/foo");
    },
  );
});

await acheck("probeModelSize HEADs an exact resolve URL for its size", async () => {
  const url = "https://huggingface.co/o/r/resolve/main/m.onnx";
  await withFetch({ [url]: { contentLength: 4096 } }, async () => {
    const info = await probeModelSize(url);
    assert.equal(info.size, 4096);
    assert.equal(info.name, "m.onnx");
  });
});

await acheck("probeModelSize returns a null size when Content-Length is absent", async () => {
  const url = "https://example.com/models/net.onnx";
  await withFetch({ [url]: {} }, async () => {
    const info = await probeModelSize(url);
    assert.equal(info.size, null);
    assert.equal(info.name, "net.onnx");
  });
});

await acheck("probeModelSize surfaces a repo with no .onnx", async () => {
  const api = "https://huggingface.co/api/models/onnxmodelzoo/empty?blobs=true";
  await withFetch(
    { [api]: { json: { siblings: [{ rfilename: "readme.md", size: 1 }] } } },
    async () => {
      await assert.rejects(probeModelSize("empty"), /no \.onnx file found/);
    },
  );
});

// A minimal Blob-like whose slice(a,b) streams that sub-range in small chunks,
// mirroring how XetBlob.slice() yields only its byte range.
function fakeBlob(bytes, chunk = 3) {
  const make = (buf) => ({
    size: buf.length,
    stream() {
      let i = 0;
      return {
        getReader: () => ({
          read: async () => {
            if (i >= buf.length) return { done: true, value: undefined };
            const value = buf.slice(i, i + chunk);
            i += value.length;
            return { done: false, value };
          },
        }),
      };
    },
  });
  return {
    size: bytes.length,
    slice: (start, end) => make(bytes.slice(start, end)),
  };
}

await acheck("readBlobSharded reassembles shards in order", async () => {
  const data = new Uint8Array(20);
  for (let i = 0; i < data.length; i++) data[i] = i;
  for (const k of [1, 3, 4, 7, 100]) {
    const progress = [];
    const out = await readBlobSharded(fakeBlob(data), k, (p) => progress.push(p));
    assert.equal(out.length, data.length);
    assert.deepEqual(Array.from(out), Array.from(data), `k=${k}`);
    // Progress ends at the full size, never exceeding it.
    const last = progress[progress.length - 1];
    assert.deepEqual(last, { loaded: data.length, total: data.length }, `k=${k} progress`);
    assert.ok(progress.every((p) => p.loaded <= p.total));
  }
});

await acheck("readBlobSharded falls back to arrayBuffer without a stream", async () => {
  const data = new Uint8Array([5, 6, 7, 8, 9]);
  const blob = {
    size: data.length,
    slice: (start, end) => ({
      size: end - start,
      arrayBuffer: async () => data.slice(start, end).buffer,
    }),
  };
  const out = await readBlobSharded(blob, 2);
  assert.deepEqual(Array.from(out), Array.from(data));
});

console.log(`\n${passed} checks passed`);
