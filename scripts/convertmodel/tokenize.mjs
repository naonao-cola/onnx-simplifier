// Tokenize real sample text for the "Run inference" panel's sample-data fill
// mode, for the NLP models in the curated Hugging Face model set
// (scripts/regression/models.json) — bert, gpt2, t5, albert, ...
//
// Those onnxmodelzoo `*_Opset*.onnx` files carry no tokenizer.json of their
// own (they're bare ONNX graphs), but their filenames name the model family
// they were exported from (e.g. "bert_Opset18" -> BertModel), and each
// family's exporter script (github.com/onnx/turnkeyml/tree/main/models/
// transformers) pins a specific, identifiable checkpoint — e.g.
// `BertModel.from_pretrained("bert-base-uncased")`. FAMILY_TOKENIZER records
// those checkpoints, verified against the exporter scripts, so a family's
// vocab/token scheme is fetched from the checkpoint that actually produced
// the exported graph rather than guessed.
//
// Deliberately no generic fallback tokenizer for an unrecognized family:
// reusing e.g. BERT's vocab on an unrelated model risks emitting a token id
// past that model's embedding table (an out-of-bounds Gather) — the same
// hazard input_fill.mjs's header comment already flags for integer inputs,
// which is why it always leaves them at zero rather than guessing a value.
// tokenizeForModel() throws for an unmapped family; callers fall back to
// synthetic fill and log why.

import { TOKENIZERS_ESM } from "./cdn.mjs";

// family (matched against a lowercased, "_Opset<N>"/".onnx"-stripped model
// filename) -> the Hugging Face repo whose tokenizer.json/tokenizer_config.json
// matches the checkpoint that family's turnkeyml export script loads.
export const FAMILY_TOKENIZER = {
  bert: "bert-base-uncased",
  gpt2: "gpt2",
  albert: "albert-base-v2",
  distilbert: "distilbert-base-uncased",
  roberta: "roberta-base",
  xlmroberta: "xlm-roberta-base",
  electra: "google/electra-small-discriminator",
  deberta: "microsoft/deberta-base",
  mobilebert: "google/mobilebert-uncased",
  squeezebert: "squeezebert/squeezebert-uncased",
  bart: "facebook/bart-base",
  mvp: "RUCAIBox/mvp",
  t5: "t5-small",
  longt5: "google/long-t5-local-base",
};

// Resolve a model filename (e.g. "bert_Opset18.onnx", "onnxmodelzoo/gpt2_Opset18")
// to a known family key, or null when nothing matches. Pure — no network/DOM.
// Matches the WHOLE normalized name against FAMILY_TOKENIZER's keys (not a
// substring test), so e.g. "distilbert" resolves to itself rather than
// accidentally also matching a shorter unrelated family name.
export function familyFromFilename(name) {
  if (!name) return null;
  const base = String(name)
    .toLowerCase()
    .replace(/^.*\//, "") // drop a leading "org/" if a repo id slipped in
    .replace(/\.onnx$/, "")
    .replace(/[-_]+/g, "_") // normalize separators before stripping "_opsetN"
    .replace(/_opset\d+$/, "")
    .replace(/_+/g, ""); // drop remaining separators for an exact-key lookup
  return Object.prototype.hasOwnProperty.call(FAMILY_TOKENIZER, base) ? base : null;
}

const tokenizerModulePromise = (() => {
  let p = null;
  return () => {
    if (!p) p = import(/* @vite-ignore */ TOKENIZERS_ESM);
    return p;
  };
})();

// tokenizer.json/tokenizer_config.json fetches, memoized per checkpoint repo
// so switching back and forth between demo models doesn't re-download them.
const tokenizerPromises = {};
async function loadTokenizer(repo) {
  if (!tokenizerPromises[repo]) {
    tokenizerPromises[repo] = (async () => {
      const { Tokenizer } = await tokenizerModulePromise();
      const base = `https://huggingface.co/${repo}/resolve/main`;
      const [tokenizerJson, tokenizerConfig] = await Promise.all([
        fetch(`${base}/tokenizer.json`).then((r) => {
          if (!r.ok) throw new Error(`no tokenizer.json in ${repo} (HTTP ${r.status})`);
          return r.json();
        }),
        fetch(`${base}/tokenizer_config.json`)
          .then((r) => (r.ok ? r.json() : {}))
          .catch(() => ({})),
      ]);
      return new Tokenizer(tokenizerJson, tokenizerConfig);
    })();
  }
  return tokenizerPromises[repo];
}

// Tokenize `text` for the model named `modelName`, producing fixed-length
// arrays sized to `seqLen` (truncated, or right-padded with 0 — the pad id for
// every family in FAMILY_TOKENIZER's WordPiece/BPE vocabs). Returns
// { input_ids, attention_mask, token_type_ids } as plain Number arrays (the
// caller converts to the model's actual tensor dtype); token_type_ids is a
// single-segment default (all zero), matching how these encoder models are
// normally fed one sentence.
//
// Throws when `modelName`'s family isn't in FAMILY_TOKENIZER, or when the
// checkpoint's tokenizer files can't be loaded — callers should catch this and
// fall back to synthetic fill.
export async function tokenizeForModel(modelName, text, seqLen) {
  const family = familyFromFilename(modelName);
  if (!family) {
    throw new Error(`no known tokenizer family for model '${modelName}'`);
  }
  const repo = FAMILY_TOKENIZER[family];
  const tokenizer = await loadTokenizer(repo);
  const encoded = tokenizer.encode(text);
  const ids = (encoded.ids || []).slice(0, seqLen);
  const mask = (encoded.attention_mask || ids.map(() => 1)).slice(0, seqLen);
  while (ids.length < seqLen) ids.push(0);
  while (mask.length < seqLen) mask.push(0);
  return {
    input_ids: ids,
    attention_mask: mask,
    token_type_ids: new Array(seqLen).fill(0),
  };
}
