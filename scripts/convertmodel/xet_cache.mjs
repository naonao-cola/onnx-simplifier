// Persistent, content-addressed cache for Hugging Face Xet downloads.
//
// EXPERIMENTAL / prototype. The Xet protocol reconstructs a file from
// deduplicated content-addressed blocks (CAS). Its payoff over a plain
// `resolve` download only shows up when those blocks are *reused* — otherwise a
// browser with no persistent store just re-fetches everything. This module adds
// that store: an IndexedDB-backed cache plus a `fetch` wrapper that transparently
// serves CAS block / reconstruction range requests from it.
//
// The cache is keyed by the request's origin + path + Range, deliberately
// dropping the query string: CAS block URLs are presigned (a rotating signature
// in the query), but the path already encodes the block hash, so this stays a
// stable content address across sessions.
//
// Only the pure cache-key and caching-fetch logic is unit-tested (test/
// xet_cache.test.mjs); the IndexedDB wrapper needs a real browser.

// A GET with a Range header to a host other than the Hub is a CAS block or
// reconstruction fetch — content-addressed and safe to cache. Token refresh and
// file-info calls go to huggingface.co and are passed through untouched.
function isHubHost(hostname) {
  return hostname === "huggingface.co" || hostname.endsWith(".huggingface.co");
}

// Normalize a request into { url, method, headers } whether fetch was called
// with a string/URL + init or a Request object.
function reqInfo(input, init) {
  if (typeof input === "string" || input instanceof URL) {
    return {
      url: String(input),
      method: (init && init.method) || "GET",
      headers: init && init.headers,
    };
  }
  // A Request-like object.
  return { url: input.url, method: input.method || "GET", headers: input.headers };
}

// Read a header by name from a Headers instance, a plain object, or an array of
// [name, value] pairs — the three shapes fetch accepts.
function headerGet(headers, name) {
  if (!headers) return null;
  const lower = name.toLowerCase();
  if (typeof headers.get === "function") return headers.get(name);
  if (Array.isArray(headers)) {
    const found = headers.find(([k]) => String(k).toLowerCase() === lower);
    return found ? found[1] : null;
  }
  for (const k of Object.keys(headers)) {
    if (k.toLowerCase() === lower) return headers[k];
  }
  return null;
}

// The content-addressed cache key: origin + path + Range, without the query.
export function casCacheKey(url, rangeHeader) {
  const u = new URL(url);
  return `${u.origin}${u.pathname}#${rangeHeader || ""}`;
}

// Which request headers to preserve so a cached response replays faithfully
// (Xet issues Range requests and may read Content-Range).
const REPLAY_HEADERS = ["content-type", "content-range", "content-length"];

function pickHeaders(headers) {
  const out = {};
  for (const name of REPLAY_HEADERS) {
    const v = headers.get(name);
    if (v != null) out[name] = v;
  }
  return out;
}

// Decide whether a request is a cacheable CAS/reconstruction range fetch.
export function isCacheable(url, method, rangeHeader) {
  if ((method || "GET").toUpperCase() !== "GET") return false;
  if (!rangeHeader) return false;
  try {
    return !isHubHost(new URL(url).hostname);
  } catch {
    return false;
  }
}

// Wrap a fetch so cacheable requests are served from / stored into `cache`
// (any { get(key), put(key, entry) }). Optional progress callbacks:
// `hooks.onHit(key, bytes)` (served from cache), `hooks.onMiss(key)` (about to
// fetch), and `hooks.onNetwork(bytes)` (bytes actually pulled over the network
// on a miss — the honest denominator for a download-speed estimate, since cached
// chunks never cross the network). A cache failure never breaks the request — it
// just falls back to the network.
export function makeCachingFetch(realFetch, cache, hooks = {}) {
  // Background cache writes in flight. `fetchImpl.settled()` resolves once they
  // all finish, so a caller can read an accurate cache size right after a
  // download (the writes are deliberately not awaited on the download path).
  const pendingWrites = new Set();

  const fetchImpl = async (input, init) => {
    const { url, method, headers } = reqInfo(input, init);
    const range = headerGet(headers, "range");
    if (!isCacheable(url, method, range)) {
      return realFetch(input, init);
    }
    const key = casCacheKey(url, range);

    let hit = null;
    try {
      hit = await cache.get(key);
    } catch {
      hit = null;
    }
    if (hit && hit.body) {
      if (hooks.onHit) hooks.onHit(key, hit.body.byteLength);
      return new Response(hit.body, { status: hit.status || 200, headers: hit.headers });
    }

    if (hooks.onMiss) hooks.onMiss(key);
    const resp = await realFetch(input, init);
    if (resp.ok || resp.status === 206) {
      try {
        const body = await resp.clone().arrayBuffer();
        if (hooks.onNetwork) hooks.onNetwork(body.byteLength);
        // Persist in the background — do NOT await it. IndexedDB serializes
        // writes per object store and structured-cloning multi-MB blocks is
        // slow, so awaiting the write here would funnel even parallel range
        // fetches through IDB's write throughput and throttle the download.
        // Serving `resp` doesn't depend on the block being stored. Tracked in
        // `pendingWrites` so settled() can report when the cache is consistent.
        const entry = { status: resp.status, headers: pickHeaders(resp.headers), body };
        const write = Promise.resolve(cache.put(key, entry)).catch(() => {
          // Non-fatal: the block just won't be cached for next time.
        });
        pendingWrites.add(write);
        write.then(() => pendingWrites.delete(write));
      } catch {
        // Non-fatal: serve the live response even if reading the body failed.
      }
    }
    return resp;
  };

  // Resolve once every background cache write started so far has completed.
  fetchImpl.settled = () => Promise.all([...pendingWrites]);
  return fetchImpl;
}

// An IndexedDB-backed cache. Falls back to an in-memory Map when IndexedDB is
// unavailable (e.g. private-mode restrictions), so the caller never has to
// branch. Browser-only; not exercised by the node unit tests.
export function openXetCache(dbName = "onnxsim-xet-cache", storeName = "blocks") {
  if (typeof indexedDB === "undefined") return memoryCache();

  let dbPromise;
  const db = () => {
    if (!dbPromise) {
      dbPromise = new Promise((resolve, reject) => {
        const req = indexedDB.open(dbName, 1);
        req.onupgradeneeded = () => req.result.createObjectStore(storeName);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
    }
    return dbPromise;
  };
  const run = (mode, fn) =>
    db().then(
      (d) =>
        new Promise((resolve, reject) => {
          const tx = d.transaction(storeName, mode);
          const req = fn(tx.objectStore(storeName));
          req.onsuccess = () => resolve(req.result);
          req.onerror = () => reject(req.error);
        }),
    );

  // Total stored bytes + entry count, by scanning every cached block with a
  // cursor (there's no running aggregate). Used only for the UI's cache-size
  // readout, so a full scan is acceptable for this experimental feature.
  const size = () =>
    db().then(
      (d) =>
        new Promise((resolve, reject) => {
          const req = d.transaction(storeName, "readonly").objectStore(storeName).openCursor();
          let bytes = 0;
          let count = 0;
          req.onsuccess = () => {
            const cursor = req.result;
            if (!cursor) return resolve({ bytes, count });
            const v = cursor.value;
            if (v && v.body) bytes += v.body.byteLength;
            count += 1;
            cursor.continue();
          };
          req.onerror = () => reject(req.error);
        }),
    );

  return {
    get: (key) => run("readonly", (s) => s.get(key)),
    put: (key, value) => run("readwrite", (s) => s.put(value, key)),
    clear: () => run("readwrite", (s) => s.clear()),
    size,
  };
}

function memoryCache() {
  const map = new Map();
  return {
    get: async (key) => map.get(key) || null,
    put: async (key, value) => void map.set(key, value),
    clear: async () => void map.clear(),
    size: async () => {
      let bytes = 0;
      for (const v of map.values()) if (v && v.body) bytes += v.body.byteLength;
      return { bytes, count: map.size };
    },
  };
}
