// Unit test for the Hugging Face token storage helpers behind the converter
// page's token field. No DOM/browser needed -- storage is a plain fake.
//
// Usage:
//   node test/hf_token.test.mjs

import assert from "node:assert/strict";
import { loadStoredToken, saveToken, clearToken } from "../hf_token.mjs";

let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log("  ok -", name);
}

// Minimal in-memory Storage stand-in.
function fakeStorage() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k),
    _map: map,
  };
}

check("loadStoredToken returns '' when nothing is stored", () => {
  assert.equal(loadStoredToken(fakeStorage()), "");
});

check("loadStoredToken returns '' for a null/undefined storage", () => {
  assert.equal(loadStoredToken(null), "");
  assert.equal(loadStoredToken(undefined), "");
});

check("saveToken then loadStoredToken round-trips", () => {
  const s = fakeStorage();
  saveToken(s, "hf_abc123");
  assert.equal(loadStoredToken(s), "hf_abc123");
});

check("saveToken with an empty token clears any stored value", () => {
  const s = fakeStorage();
  saveToken(s, "hf_abc123");
  saveToken(s, "");
  assert.equal(loadStoredToken(s), "");
});

check("saveToken on a null storage does not throw", () => {
  assert.doesNotThrow(() => saveToken(null, "hf_abc123"));
});

check("clearToken removes a stored token", () => {
  const s = fakeStorage();
  saveToken(s, "hf_abc123");
  clearToken(s);
  assert.equal(loadStoredToken(s), "");
});

check("clearToken on a null storage does not throw", () => {
  assert.doesNotThrow(() => clearToken(null));
});

check("a storage that throws on getItem is swallowed", () => {
  const s = {
    getItem: () => {
      throw new Error("storage disabled");
    },
  };
  assert.equal(loadStoredToken(s), "");
});

check("a storage that throws on setItem is swallowed", () => {
  const s = {
    setItem: () => {
      throw new Error("quota exceeded");
    },
  };
  assert.doesNotThrow(() => saveToken(s, "hf_abc123"));
});

console.log(`\n${passed} checks passed`);
