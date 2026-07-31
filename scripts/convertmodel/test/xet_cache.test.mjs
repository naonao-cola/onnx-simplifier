// Unit test for the pure cache-key + caching-fetch logic behind the
// experimental Xet download path. The IndexedDB wrapper (openXetCache) needs a
// real browser and is not covered here; makeCachingFetch is tested against a
// plain Map-backed cache and a stubbed fetch.
//
// Usage:
//   node test/xet_cache.test.mjs

import assert from "node:assert/strict";
import { casCacheKey, isCacheable, makeCachingFetch } from "../xet_cache.mjs";

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

check("casCacheKey drops the (volatile presign) query string", () => {
  const a = casCacheKey("https://cas.example/blk/abc123?sig=AAA&exp=1", "bytes=0-99");
  const b = casCacheKey("https://cas.example/blk/abc123?sig=BBB&exp=2", "bytes=0-99");
  assert.equal(a, b); // same block + range -> same key despite rotating signature
  assert.equal(a, "https://cas.example/blk/abc123#bytes=0-99");
});

check("casCacheKey separates different ranges", () => {
  const a = casCacheKey("https://cas.example/blk/x", "bytes=0-99");
  const b = casCacheKey("https://cas.example/blk/x", "bytes=100-199");
  assert.notEqual(a, b);
});

check("isCacheable accepts a ranged GET to a non-Hub host", () => {
  assert.equal(isCacheable("https://cas.example/blk/x", "GET", "bytes=0-9"), true);
});

check("isCacheable rejects Hub hosts, non-GET, and range-less requests", () => {
  assert.equal(isCacheable("https://huggingface.co/api/x", "GET", "bytes=0-9"), false);
  assert.equal(isCacheable("https://x.huggingface.co/y", "GET", "bytes=0-9"), false);
  assert.equal(isCacheable("https://cas.example/blk/x", "POST", "bytes=0-9"), false);
  assert.equal(isCacheable("https://cas.example/blk/x", "GET", null), false);
});

// A Map-backed cache with the { get, put } shape makeCachingFetch expects.
function mapCache() {
  const map = new Map();
  return {
    get: async (k) => map.get(k) || null,
    put: async (k, v) => void map.set(k, v),
    _map: map,
  };
}

await acheck("caches a CAS range GET on miss, replays it on hit", async () => {
  const cache = mapCache();
  let networkCalls = 0;
  const realFetch = async () => {
    networkCalls += 1;
    return new Response(new Uint8Array([1, 2, 3, 4]).buffer, {
      status: 206,
      headers: { "content-range": "bytes 0-3/4", "content-type": "application/octet-stream" },
    });
  };
  const f = makeCachingFetch(realFetch, cache);

  const url = "https://cas.example/blk/abc?sig=1";
  const r1 = await f(url, { headers: { Range: "bytes=0-3" } });
  assert.equal(r1.status, 206);
  assert.deepEqual(Array.from(new Uint8Array(await r1.arrayBuffer())), [1, 2, 3, 4]);
  assert.equal(networkCalls, 1);

  // Second call with a *different* presigned query hits the cache (no network).
  const r2 = await f("https://cas.example/blk/abc?sig=2", { headers: { Range: "bytes=0-3" } });
  assert.equal(r2.status, 206);
  assert.equal(r2.headers.get("content-range"), "bytes 0-3/4");
  assert.deepEqual(Array.from(new Uint8Array(await r2.arrayBuffer())), [1, 2, 3, 4]);
  assert.equal(networkCalls, 1); // still 1 -> served from cache
});

await acheck("passes through non-cacheable requests untouched", async () => {
  const cache = mapCache();
  let networkCalls = 0;
  const realFetch = async () => {
    networkCalls += 1;
    return new Response("ok", { status: 200 });
  };
  const f = makeCachingFetch(realFetch, cache);

  // Hub host (token refresh / file info): not cached.
  await f("https://huggingface.co/api/models/x", { headers: { Range: "bytes=0-9" } });
  // No Range header: not cached.
  await f("https://cas.example/blk/x");
  assert.equal(networkCalls, 2);
  assert.equal(cache._map.size, 0);
});

await acheck("fires onHit / onMiss hooks", async () => {
  const cache = mapCache();
  const realFetch = async () =>
    new Response(new Uint8Array([9]).buffer, { status: 206 });
  const events = [];
  const f = makeCachingFetch(realFetch, cache, {
    onHit: (_k, n) => events.push(["hit", n]),
    onMiss: () => events.push(["miss"]),
  });
  const url = "https://cas.example/blk/z";
  await f(url, { headers: { Range: "bytes=0-0" } });
  await f(url, { headers: { Range: "bytes=0-0" } });
  assert.deepEqual(events, [["miss"], ["hit", 1]]);
});

await acheck("a cache read error falls back to the network", async () => {
  const cache = {
    get: async () => {
      throw new Error("idb blew up");
    },
    put: async () => {},
  };
  let networkCalls = 0;
  const realFetch = async () => {
    networkCalls += 1;
    return new Response(new Uint8Array([1]).buffer, { status: 206 });
  };
  const f = makeCachingFetch(realFetch, cache);
  const r = await f("https://cas.example/blk/x", { headers: { Range: "bytes=0-0" } });
  assert.equal(r.status, 206);
  assert.equal(networkCalls, 1);
});

console.log(`\n${passed} checks passed`);
