// Persistence for the optional Hugging Face access token the "Load from
// Hugging Face" panel uses to reach gated / private repos.
//
// Kept pure (no DOM access, `storage` is injected) so it's unit-testable with a
// plain in-memory fake; hf_load.mjs does the actual browser wiring against
// window.localStorage. The token itself never touches the URL / shareable
// link (see share_url.mjs / query_params.mjs, which only reflect the model
// reference and conversion options) and is only ever attached, by hf_models.mjs,
// to requests aimed at huggingface.co.

const STORAGE_KEY = "onnxsim:hfToken";

// Read the remembered token, or "" if none is stored / storage is unavailable
// (private browsing, disabled storage, quota errors, …).
export function loadStoredToken(storage) {
  try {
    return (storage && storage.getItem(STORAGE_KEY)) || "";
  } catch {
    return "";
  }
}

// Persist (or, for an empty token, remove) the remembered token. Best-effort:
// a storage failure must not break the page, so it's swallowed.
export function saveToken(storage, token) {
  try {
    if (!storage) return;
    if (token) storage.setItem(STORAGE_KEY, token);
    else storage.removeItem(STORAGE_KEY);
  } catch {
    // storage unavailable — the token just won't persist across reloads
  }
}

// Forget any remembered token (used when the user unchecks "remember").
export function clearToken(storage) {
  try {
    if (storage) storage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
