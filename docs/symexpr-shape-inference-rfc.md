# RFC: symbolic shape inference via `SymExpr`

Status: **Draft**. Closes #597. Builds on the merged #527 (`SymExpr` core) and
the in-repo M0-M3 milestones of issue #532 (`sym_expr`, `sym_value_eval`,
`sym_shape_infer`, wired into `_EvalPartialShape` in `onnxsim.cpp`). This
document is the design record #597 asked for: why the feature exists, how it
compares to the rest of the ONNX ecosystem, a real model that needs it, and
what is still open.

## 1. Problem

onnxsim already folds shape-computing subgraphs (`Shape -> Gather -> ... ->
Reshape`) into constants when the graph is fully static, and — since issue
\#139 — also when ONNX's own `enable_data_propagation` shape inference can
resolve a value that mixes concrete dims with an *opaque* `dim_param`. That
covers a `Reshape` target like `[batch, -1]`. It does **not** cover a target
like `[batch, num_heads * head_dim]` or `[past_seq_len + seq_len, ...]`,
because `onnx::TensorShapeProto`'s data-propagation representation has no
notion of arithmetic on a `dim_param`: each dimension is either a concrete
`dim_value` or an uninterpreted string. The moment two dynamic dims (or a
dynamic dim and a constant) need to be added, multiplied, or divided, data
propagation gives up and the entire producing subgraph — `Shape`, `Gather`,
`Add`, `Concat`, sometimes a `Cast`/`Unsqueeze` chain around each — survives
simplification as dead weight the runtime still has to execute on every
inference call.

`model_info.py` hit the same wall for a different purpose (reporting MAC and
byte-count *formulas* rather than rewriting the graph) and solved it with
`sympy.Symbol` for each `dim_param`. That's descriptive-only and sympy is an
optional Python dependency; it cannot be the answer for the C++ core, which
also has to build under Emscripten/WASM (`build_wasm.sh`, `npm/`) where sympy
isn't available and SymEngine's GMP dependency isn't practical (see the
comment at the top of `sym_expr.h`). Issue #597 — filed against #527 — asks
for the RFC that gives the actual graph-rewriting use of `SymExpr` (not just
`model_info`'s reporting use) a proper design writeup.

## 2. What already exists

Three dependency-free C++ layers, each independently unit-tested and each
touching no `onnx::` type until the adapter at the top:

| Layer | File | Role |
|---|---|---|
| M0 | `onnxsim/sym_expr.{h,cpp}` | `SymExpr`: an integer-coefficient polynomial in dim-symbol names (`std::map<Monomial, int64_t>`). `+ - *`, `TryExactDivide`, `TryEqual`, `str()`/`str_factored()`. |
| M1 | `onnxsim/sym_value_eval.{h,cpp}` | `EvaluateSymbolicValues`: a symbolic *value* evaluator for the shape-scaffolding op family (`Shape, Gather, Slice, Squeeze/Unsqueeze, Concat, Reshape, Expand, Where, Equal, Div, Cast, Range, ReduceProd, Tile, Constant, ConstantOfShape, Floor, Max, Min, Neg, Size, Sub, Transpose`), operating over `SymTensor` (a rank-0/1 tensor of `SymExpr`). |
| M2 | `onnxsim/sym_shape_infer.{h,cpp}` | `InferSymbolicShapes`: symbolic *activation-shape* inference — re-derives each tensor's shape as a `SymShape` (`vector<SymExpr>`) across compute ops (`Conv, MatMul, Gemm, Add/Sub/Mul/Div` broadcasting, `AveragePool/MaxPool/LpPool, Softmax, LayerNormalization, Transpose, Concat, Squeeze/Unsqueeze, Slice, Gather, Reshape, Cast, Identity, Where, Equal/Greater/Less, And/Or, Pow`), so a symbol survives ops that ONNX's own shape inference would otherwise collapse to an anonymous unknown dim. |
| M3 | `onnxsim/onnxsim.cpp` (`EvaluateModelSymbolicValues`, `_EvalPartialShape`) | The `onnx::ModelProto` adapter: builds a `ShapeGraph`/`SymGraph` from the model, runs M2 then M1, and merges whatever it resolves into the same `folded_values` / `reshape_fixes` rewrite maps the ONNX-data-propagation path already populates. Fully concrete results become `Constant` nodes; a `Reshape` target left with exactly one symbolic entry becomes `[-1, ...]`. |

Two properties are deliberate and load-bearing, not oversights:

- **Fails closed.** A dim a rule can't compute exactly (a strided `Conv` over
  a symbolic spatial length, which isn't a polynomial) gets a *fresh, never
  re-merged* symbol rather than being dropped or guessed — the rest of the
  shape stays usable and nothing is ever equated that only happens to agree
  at one representative value. `TryEqual` returns `std::nullopt` rather than
  `false` when a symbolic difference can't be decided, so a `Where` it can't
  resolve is left alone rather than picking a branch that might be wrong.
- **Gated by output**, not by algebra alone. Whatever this pass folds still
  has to survive onnxsim's own input/output equivalence check (`check_n`).
  The algebra only has to be conservative, not proven correct in isolation.

`SymExpr` is intentionally *not* a general CAS: no `floor`/`ceil`/`min`/`max`
symbolic terms, no inequality solving, and `TryExactDivide` only handles a
single-monomial divisor. That matches what ONNX graphs actually generate —
reshape targets and cache-length arithmetic are sums and products of
`dim_param`s and constants, not general rational functions — and keeps the
representation exact and cheap (`std::map` insert/merge, no simplification
search).

## 3. Landscape: how others solve this

| Project | Symbolic engine | Where it runs | Scope | WASM-safe |
|---|---|---|---|---|
| ONNX core (`onnx.shape_inference`, `enable_data_propagation`) | none (opaque `dim_param` string) | C++/Python, in-tree | Baseline: no arithmetic across a symbolic dim | yes |
| [`onnxruntime/python/tools/symbolic_shape_infer.py`](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/symbolic_shape_infer.py) | `sympy.Symbol` (`integer=True, nonnegative=True`) | Python, standalone script | ~100+ per-op `_infer_*` handlers, `sympy_data_` value propagation through `Shape->Slice->Concat->Reshape`, heuristic dimension merging (`_add_suggested_merge`, `auto_merge_`), `int_max_` clamp heuristics for unbounded literals | no (sympy/GMP) |
| [`justinchuby/onnx-shape-inference`](https://github.com/justinchuby/onnx-shape-inference) | `sympy` again, but built directly on `onnx_ir` (no protobuf round-trip) | Python, standalone library | Anonymous engine symbols (`_d0`, `_d1`) reconciled against author-declared `dim_param`s via constraint resolution; renames compound expressions like `2*_d0 -> 2*batch`; extensible registry incl. `com.microsoft` contrib ops | no (sympy) |
| [`onnxslim`](https://github.com/inisis/onnxslim) | none of its own | Python, graph-cleanup tool | Prefers ORT's shape inference (falls back to `onnx.shape_inference`) purely to get *shapes*, then edits the graph with `onnx_graphsurgeon`; does not itself do symbolic-dim algebra, so it inherits whatever ORT's tool resolved (or didn't) upstream | no (depends on ORT's tool for the symbolic case) |
| onnxsim (`SymExpr`, this RFC) | hand-rolled integer-coefficient polynomial, no external CAS | C++, in-tree, part of the simplifier pipeline itself | Narrower op coverage than ORT's tool by design (§2), but the result directly *rewrites the graph* (folds to `Constant`/`[-1]`) rather than only annotating it, and every fold is gated by `check_n` | **yes** — the whole point |

The three external symbolic tools all converge on `sympy` for the algebra,
which is the right call for a standalone Python analysis tool but is not an
option for onnxsim's C++ core: it has to build under Emscripten
(`ONNXSIM_BUILTIN_ORT` and the wheel build are separate from the WASM/npm
build, but both share this same core — see `CLAUDE.md`), and it has to
*rewrite* the graph inline as one step of a larger fixed-point simplification
loop rather than run as a separate preprocessing pass a user invokes by hand.
`onnxslim` shows what happens without an in-tree symbolic engine at all: it's
only as good at dynamic shapes as whatever ORT's tool (if installed) already
resolved.

`justinchuby/onnx-shape-inference`'s anonymous-symbol reconciliation is worth
tracking for M4 (§5) — onnxsim's M2 already mints "fresh" symbols
(`seedunk_N`/`unk_N`) for dims a rule can't compute, and never merges them
back; a constraint-based reconciliation pass could recover some of those
without violating the fail-closed rule, as long as it stays a *hint* gated by
`check_n` rather than a source of truth.

## 4. A model that actually needs this: decoder-only transformer with KV cache

The shape-scaffolding pattern this RFC is about is not a corner case — it's
the standard shape of every autoregressive LLM export with a key/value cache,
which is how essentially every transformer decoder (Llama, Mistral, Qwen,
GPT-2/Neo, Phi, ...) is shipped to ONNX Runtime today (this is exactly what
`optimum.exporters.onnx` and HF's own ONNX configs produce, and what
`torch.onnx.export(..., dynamo=True)` is now the supported path for).

### 4.1 Export

```python
import torch
from torch.export import Dim

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, hidden=1024, n_head=16):
        super().__init__()
        self.n_head, self.head_dim = n_head, hidden // n_head
        self.qkv = torch.nn.Linear(hidden, 3 * hidden)
        self.proj = torch.nn.Linear(hidden, hidden)

    def forward(self, x, past_key, past_value):
        b, s, h = x.shape
        q, k, v = self.qkv(x).split(h, dim=-1)
        def split_heads(t):
            return t.view(b, s, self.n_head, self.head_dim).transpose(1, 2)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # KV cache: total_seq is a *sum* of two dynamic dims.
        k = torch.cat([past_key, k], dim=2)
        v = torch.cat([past_value, v], dim=2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        # num_heads * head_dim merge back into hidden -- a *product* of a
        # dynamic-looking dim (b, s stay dynamic) and two static ones.
        out = out.transpose(1, 2).reshape(b, s, h)
        return self.proj(out), k, v

model = CausalSelfAttention()
x = torch.randn(2, 8, 1024)
past_k = torch.randn(2, 16, 5, 64)
past_v = torch.randn(2, 16, 5, 64)

batch, seq_len, past_len = Dim("batch"), Dim("seq_len"), Dim("past_len")
onnx_program = torch.onnx.export(
    model,
    (x, past_k, past_v),
    dynamic_shapes={
        "x": {0: batch, 1: seq_len},
        "past_key": {0: batch, 2: past_len},
        "past_value": {0: batch, 2: past_len},
    },
    dynamo=True,
    input_names=["x", "past_key", "past_value"],
    output_names=["out", "present_key", "present_value"],
)
onnx_program.save("attn_kv_cache.onnx")
```

(HF's own `optimum` ONNX configs export the same shape with the dim names
`past_sequence_length` and `sequence_length`, and a `present_key` output
whose sequence axis is documented as `past_sequence_length +
sequence_length` — same pattern, same symbol names, in production today.)

### 4.2 The subgraph this produces

The exported graph contains, among others:

```
Shape(past_key)              -> [batch, num_heads, past_len, head_dim]
Gather(..., axis=0, index=2) -> past_len                     (dim_param)
Shape(k_new)                 -> [batch, num_heads, seq_len, head_dim]
Gather(..., axis=0, index=2) -> seq_len                      (dim_param)
Add(past_len, seq_len)       -> total_len                    <-- symbol + symbol
Concat(past_key, k_new, axis=2)               -> present_key  # shape [b, nh, total_len, hd]
...
Reshape(attn_out, [batch, seq_len, num_heads * head_dim])     <-- symbol * const, const
```

`onnx.shape_inference` with data propagation resolves `Shape(past_key)` and
the `Gather` down to the opaque symbol `past_len`, but `Add(past_len,
seq_len)` is arithmetic between two `dim_param`s — `TensorShapeProto` has no
representation for "the sum of two symbols", so `data_map` has no entry for
`total_len`, and the `Reshape`/`Concat`/mask-building subgraph downstream of
it survives simplification untouched, along with the `num_heads * head_dim`
multiplication feeding the final `Reshape`.

`SymExpr` resolves both: `Symbol("past_len") + Symbol("seq_len")` is a
two-term polynomial (`str()` prints `past_len + seq_len`), so M1/M2 track
`total_len` as a real symbolic value through the rest of the KV-cache
plumbing; and `SymExpr(16) * SymExpr(64)` collapses to the constant `1024`
via `operator*`, so the final `Reshape([batch, seq_len, 1024])` has exactly
one symbolic entry and M3's rewrite emits `[-1, seq_len, 1024]`... i.e. the
`Shape -> Gather -> Mul -> Concat` chain computing `num_heads*head_dim` at
runtime becomes dead and is removed by the optimizer, on every model built
with this near-universal export shape.

This pattern is also why PyTorch's dynamo ONNX exporter tracker has repeated,
independent reports of exactly this symbol-arithmetic shape surfacing on real
LLM exports — e.g. pytorch/pytorch#172903 ("Dynamo ONNX export of LLM with
dynamic shapes fails: Name L is not defined", a KV-cache model), and the
`huggingface_llm_models_with_kv_cache` test added in pytorch/pytorch#143158 —
confirming this isn't a synthetic example but the shape every dynamo-exported
autoregressive LLM produces.

## 5. Remaining work (M4+)

- **Op coverage.** M2 doesn't yet carry a symbolic shape through attention's
  full pattern (`Expand`+`Reshape` for GQA head repetition, `RoPE`'s
  `cos`/`sin` gather-by-position, `If`/`Loop`/`Scan` subgraphs the way ORT's
  tool does for encoder-decoder / beam-search graphs). Grow `sym_shape_infer`
  op-by-op against onnxsim's existing model test corpus, rather than chasing
  ORT's ~100-handler surface wholesale — the class of pattern in §4 (linear
  KV-cache + reshape-merge) covers most of the transformer-decoder graphs
  onnxsim actually sees.
- **`model_info.py` on `SymExpr` instead of `sympy`.** `model_info.py`
  solved the *reporting* half of this problem with `sympy` before `SymExpr`
  existed (see `CLAUDE.md`'s note that the wheel build has no dependency on
  building ONNX Runtime — sympy is a separate, similarly optional,
  dependency). `sym_expr.h`'s own header comment already documents the
  mapping (`sympy.Symbol -> SymExpr::Symbol`, `sympy.factor -> str_factored`,
  ...). Routing `model_info.py` through the same `SymExpr` core (already
  exposed to Python via `cpp2py_export.cc`) instead of `sympy` would drop the
  optional `sympy` dependency entirely and mean the CLI's reported formulas
  and the C++ core's graph rewrites are provably using the same algebra —
  currently they're two independent implementations that happen to agree.
  This is a compatibility-affecting change (`str_factored`'s output format
  may not byte-for-byte match `sympy.factor`) and needs its own follow-up,
  not silently folded into this one.
- **Bounded reconciliation of fresh symbols**, inspired by
  `justinchuby/onnx-shape-inference`'s anonymous-symbol/constraint
  resolution: today a `seedunk_N`/`unk_N` symbol from an undecidable rule is
  never merged back even when two independent fresh symbols are provably the
  same dim. Any such pass must stay a hint gated by `check_n`, never a
  second source of truth for equality (§2's fail-closed rule stays absolute).
- **Range/positivity reasoning.** ORT's tool leans on `int_max_` heuristics
  and sympy's `positive=True` assumption for `Slice`/`Pad` clamping decisions
  `SymExpr` currently can't make (no inequalities). Worth scoping separately
  since it's a real soundness question (an assumption like "this dim is
  always positive" is a policy choice, not a derivable fact) rather than a
  pure engineering extension.
- **Upstream conversation.** Given `onnx/ir-py`'s own open issue "Integrate
  symbolic shape inference" (onnx/ir-py#57) and `justinchuby/onnx-shape-inference`
  explicitly targeting `onnx_ir`, it may be worth raising a WASM-safe,
  dependency-free polynomial engine as a point of interest there — but that
  is a separate, later conversation from landing M4 in onnxsim itself.

## 6. Alternatives considered

- **SymEngine** (C++ CAS): rejected per `sym_expr.h`'s existing comment —
  its practical build needs GMP, or an experimental boost-multiprecision
  fallback, neither viable under Emscripten.
- **Embed sympy via CPython in the C++ core**: rejected — the wheel and WASM
  builds cannot embed a Python interpreter, and this repo's core is meant to
  be usable from a plain C++/WASM host with no Python present at all.
  (`sympy` remains fine as a *Python-only, optional* dependency for
  `model_info.py`'s CLI-only reporting, independent of §5's proposal to
  eventually retire it there too.)
- **Do nothing beyond ONNX data propagation**: the status quo ante of #597 —
  rejected because it leaves the pattern in §4 permanently unsimplified,
  which is precisely what motivated #532/#527 in the first place.

## 7. Non-goals

- A general computer-algebra system. No trig, no rationals, no symbolic
  `floor`/`ceil`/`min`/`max` terms, no equation solving.
- Matching every op `symbolic_shape_infer.py` handles. onnxsim optimizes
  for the shapes its own model corpus and users' graphs actually contain,
  gated by `check_n`, not for standalone shape-inference completeness.
- Changing what onnxsim folds when data propagation *already* succeeds — the
  ONNX-data-propagation path in `_EvalPartialShape` is untouched; `SymExpr`
  only picks up what it can't reach.

## 8. Open questions

1. Should `model_info.py` actually migrate off `sympy` onto `SymExpr` (§5),
   and if so, is a `str_factored()` output difference from `sympy.factor()`
   acceptable as a breaking CLI-output change, or does it need a
   compatibility mode?
2. How wide should M4's op coverage push — is there an existing internal
   or user-reported model corpus (beyond the KV-cache pattern in §4) that
   should drive which ops get symbolic shape rules next?
3. Is any bounded reconciliation of fresh (`unk_N`) symbols worth the risk
   surface, or should undecidable dims stay permanently distinct, accepting
   the folding loss?
