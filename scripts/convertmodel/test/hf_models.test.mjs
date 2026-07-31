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

// Install a fetch stub that maps URL -> { ok, status, json?, arrayBuffer? }.
function withFetch(routes, fn) {
  const saved = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const r = routes[url];
    if (!r) return { ok: false, status: 404, statusText: "Not Found" };
    return {
      ok: r.ok !== false,
      status: r.status ?? 200,
      statusText: r.statusText ?? "OK",
      json: async () => r.json,
      arrayBuffer: async () => r.arrayBuffer ?? new ArrayBuffer(0),
    };
  };
  return Promise.resolve(fn()).finally(() => {
    globalThis.fetch = saved;
  });
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

await acheck("loadModelList reads ./models.json first", async () => {
  await withFetch(
    {
      "./models.json": {
        json: { models: [{ id: "onnxmodelzoo/a" }, { id: "onnxmodelzoo/b" }] },
      },
    },
    async () => {
      const ids = await loadModelList();
      assert.deepEqual(ids, ["onnxmodelzoo/a", "onnxmodelzoo/b"]);
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

console.log(`\n${passed} checks passed`);
