# Activation memory planning (`plan_activation_memory`)

## What this is

`onnxsim.plan_activation_memory` is a read-only analysis that computes a
static byte-offset allocation plan for a model's activation tensors: instead
of one permanent buffer per activation, it packs every tensor into a single
shared arena, reusing space from a tensor once nothing downstream still needs
it. It answers "how big a buffer would a deployment target need, and where
would each tensor live in it" without executing the model.

It never modifies the model — this is a report, like `onnxsim.model_info`'s
MACs/FLOPs metrics, not a graph rewrite. The allocation itself runs in C++
(`onnxsim/memory_planning.h`/`.cpp`), built on the same liveness pass behind
`ModelInfo.memory_footprint` (`onnxsim/model_metrics.h`/`.cpp`); this needs
onnxsim's C++ extension built (the normal case for any `pip install onnxsim`
/ built-from-source install).

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
plan = onnxsim.plan_activation_memory(model)
onnxsim.print_memory_plan(plan)
print(plan.arena_bytes, plan.compression_ratio)
```

## How the plan is built

Two passes, run once per call:

1. **Liveness.** Every non-weight tensor (a graph input, a node output, or a
   graph output) gets an interval `[produced_at, last_used_at]` in terms of
   node index — the exact same convention `PeakMemoryFootprint` uses to
   compute the *ideal* peak byte count (weights stay resident throughout and
   are excluded; a graph output is "used" through the end of the graph so it
   never gets freed early). `PeakMemoryFootprint` already reports what a
   perfectly packed allocator would need; this module is what actually
   produces that allocator's offsets.

2. **Greedy best-fit placement.** Tensors are placed largest-first (ties
   broken by name for determinism). For each tensor, every already-placed
   tensor whose liveness interval overlaps it marks out an offset range this
   tensor must avoid; the tensor goes into the smallest gap that still fits
   among those ranges, or gets appended after the last one if nothing fits.
   This is the standard linear-scan/greedy-by-size register-allocation
   heuristic — it's what makes the arena size close to (not exactly) the
   liveness-only lower bound `memory_footprint` reports, rather than a naive
   bump allocator that never reuses space at all.

Two intervals are only ever placed at overlapping offsets when they are
**not** simultaneously live — and the overlap check is conservative at
interval boundaries: a tensor whose last use is node `i` and one produced by
node `i` itself still count as overlapping, since a node's own output isn't
generally safe to alias with its inputs' storage in general. That's exactly
the exception the next pass carves out, explicitly, for the ops it's actually
safe for.

## In-place aliasing

Before the two passes above run, a separate pass unions a safe elementwise
op's input with its output whenever overwriting the input in place is
provably correct:

- the op is on a curated allowlist of shape-/dtype-preserving elementwise ops
  (`Relu`, `Sigmoid`, `Tanh`, `Neg`, `Identity`, `Clip`, ... — see
  `IsInPlaceSafeOp` in `onnxsim/memory_planning.cpp` for the exact list);
- the input isn't a weight, a graph input, or a declared graph output (an
  externally-owned buffer, or one the caller only reads after the whole
  graph finishes — overwriting either mid-run would be observed as
  corruption, not reused space);
- the input is consumed exactly once, by this node — the only read of the
  original value, so overwriting it can't corrupt some other node's read.

A chain of such ops (`Relu -> Sigmoid -> Tanh -> ...`) unions transitively
into one group that needs a single slot for its *entire* span, rather than
one slot per node — so a long elementwise chain's arena stays roughly
constant instead of growing with its length, even though `naive_bytes` still
grows with every tensor in it. Aliased tensors report the **identical**
`(offset, size)` in `tensor_offsets` on purpose: unlike two disjoint-interval
tensors (which merely may reuse each other's freed space), they're the same
logical storage. Honoring that part of the plan means actually running that
node's kernel in place — writing its output over the input's own buffer —
not just treating a repeated offset as "safe to reuse afterward."

A second, separate allowlist (`IsViewOp`) covers pure view ops — `Reshape`,
`Flatten`, `Squeeze`, `Unsqueeze` — that reinterpret the same bytes under a
different shape with no computation at all: ONNX requires row-major/
contiguous tensor layout, and none of these ops permute element order, so
their output is byte-for-byte identical to their input whenever the sizes
actually agree (checked the same way as every other candidate). A chain may
freely mix both allowlists — e.g. `Reshape -> Relu -> Flatten` all collapse
into one group — since the eligibility conditions and the resulting
placement are identical either way.

A third, independent category extends this to **binary operand donation**:
an elementwise binary op (`Add`, `Sub`, `Mul`, `Div`, `Max`, `Min`, `And`,
`Or`, `Xor`, `Mod` — see `IsInPlaceSafeBinaryOp`) can compute its output by
overwriting *one* of its two operands, the same way real NN memory planners
(MXNet, XLA) donate an operand's buffer to a binary kernel's output. The
candidate operand must clear the exact same bar as the unary pass's input —
not a weight/graph input/graph output, consumed exactly once, and matching
the output's byte size — checked against **each** operand in turn (`input[0]`
then `input[1]`), aliasing **at most one** of them. Matching the output's byte
size is what rules out a *broadcast* operand: ONNX broadcasts a smaller
operand up to the output's shape, so a genuinely broadcast operand's byte
size never equals the output's, and it is correctly left un-aliased no
matter how eligible it otherwise looks. The operand that isn't donated (if
any) is simply left for the ordinary liveness pass to place, same as any
other tensor. Because this unions into the very same `UnionFind` as the
other two categories, a chain mixing any of them (e.g.
`Reshape -> Relu -> Add -> Sigmoid`) still collapses into one group end to end.

## Control-flow subgraphs (If/Loop/Scan)

A node's control-flow subgraph bodies are each planned independently, in
their own arena starting at offset 0 -- entirely separate address space from
the owning graph's, and from a sibling subgraph's. The owning graph reserves
room for a subgraph's peak requirement inside *its own* arena for the span
the owning node executes, the same "subgraph peak added on top of the live
set at that node" convention `PeakMemoryFootprint` already uses for its own
number. When a node owns more than one subgraph (an `If`'s `then_branch` and
`else_branch`), every one of them counts towards the reservation, even
though only one branch actually runs -- the same conservative
"sum every subgraph" rule `PeakMemoryFootprint` applies.

```python
plan = onnxsim.plan_activation_memory(model)
plan.subgraph_reserved_bytes  # extra bytes reserved in this arena for subgraphs
plan.subgraph_plans           # {"<key>": MemoryPlan, ...} -- one per subgraph
```

Each nested plan is keyed by `"<node's first non-empty output name>#<subgraph
index>"` -- an `If` node producing output `y` keys its `then_branch` as
`"y#0"` and `else_branch` as `"y#1"`; a `Loop`/`Scan` node's single body keys
as `"y#0"`. `print_memory_plan` prints each nested plan under its own
"Subgraph '\<key\>'" heading after the top-level table.

This is a genuinely *joint* plan in the sense that nothing is left
unaccounted for -- every tensor anywhere in the model, subgraph bodies
included, gets a real offset in some arena, and the owning arena is sized to
actually hold every subgraph concurrently live with it -- but it is not a
*shared* one: a subgraph's tensors can never alias something live in the
outer scope (or in a sibling subgraph), even when their liveness would
otherwise allow it. That is a materially bigger allocator problem this
doesn't take on.

One consequence worth calling out: `naive_bytes` only ever counts a graph's
*own* tensors (the "one permanent slot per tensor, no reuse" baseline it has
always meant), never subgraph reservations -- so a model whose memory is
dominated by a large control-flow subgraph can end up with `arena_bytes`
*larger* than `naive_bytes`, and therefore a negative `compression_ratio`.
That is not a bug: it means the subgraph reservation is overhead a
naive top-level-only allocation never had to pay, not that this plan is
worse than the naive baseline at placing the tensors it actually reasons
about.

## What's in `MemoryPlan`

- **`tensor_offsets`** — `{name: (offset, size)}` for every planned tensor.
- **`arena_bytes`** — the arena size the plan requires (the high-water mark
  of every `offset + size`).
- **`naive_bytes`** — the sum of every planned tensor's size as if each got
  its own permanent slot (no reuse at all). This is the baseline
  `compression_ratio` (`1 - arena_bytes / naive_bytes`) is measured against.
- **`unplanned`** — tensors that could not be given a concrete size (unknown
  shape/dtype, or a dynamic dimension) and so are excluded from both
  `tensor_offsets` and `naive_bytes`. A plan with a non-empty `unplanned` is
  a partial lower bound, not a complete one — check it before trusting
  `arena_bytes` as the true requirement.
- **`subgraph_reserved_bytes`** / **`subgraph_plans`** — see
  [Control-flow subgraphs](#control-flow-subgraphs-ifloopscan) above; 0 and
  `{}` for a model with no `If`/`Loop`/`Scan` nodes.

## Annotating a model with the plan (`annotate_memory_plan`)

`onnxsim.annotate_memory_plan` writes a `plan_activation_memory` result
straight onto a shape-inferred copy of the model's `metadata_props`, the same
way `onnxsim.annotate_metadata` persists the MACs/FLOPs/memory-footprint
report. This is how the plan actually reaches a consumer that can't (or
shouldn't have to) run onnxsim itself — an embedded runtime, a code
generator, or any other tool that just reads the `.onnx` file:

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
annotated = onnxsim.annotate_memory_plan(model)
onnx.save(annotated, "model.annotated.onnx")
```

- **Model**: `onnxsim.memory_plan_arena_bytes`, `naive_bytes` and
  `compression_ratio` (the `MemoryPlan` totals); `unplanned_count` always,
  plus `unplanned` (a capped, comma-joined list of names) when non-empty.
- **Value** (every planned graph input, node output, or graph output):
  `onnxsim.mem_offset` and `onnxsim.mem_size`. A tensor that
  `plan_activation_memory` couldn't plan is simply left unannotated, the same
  way `annotate_metadata` leaves an unknown-shape tensor's `bytes` unset.
- **Control-flow subgraph body** (an `If`/`Loop`/`Scan` node's
  `then_branch`/`else_branch`/`body` `GraphProto`, recursively): the same
  value-level `mem_offset`/`mem_size` pairs from that subgraph's own nested
  `MemoryPlan`, plus its own `memory_plan_arena_bytes`/`naive_bytes`/
  `compression_ratio`/`unplanned_count`/`unplanned` written directly onto the
  subgraph `GraphProto` itself — there's no model-level object a subgraph
  body could otherwise attach totals to.

The input model is never mutated; `annotate_memory_plan` returns a shape-
inferred copy, since the per-value metadata needs a matching `value_info`
entry to attach to (a `NodeProto` doesn't get its own offset — a node's
*output value* does, keyed by tensor name, same as the metric annotations).

## Scope (v1)

- **Concrete shapes only.** A tensor with a dynamic `dim_param` dimension has
  no fixed byte size to place, so it's reported in `unplanned` rather than
  guessed. Run `--overwrite-input-shape` / `overwrite_input_shape=` (see the
  main README) to pin a dynamic model's shapes before planning it.
- **Subgraphs are planned, but not jointly.** Every `If`/`Loop`/`Scan`
  subgraph body is visited and gets its own complete, independent plan (see
  [Control-flow subgraphs](#control-flow-subgraphs-ifloopscan)) — but always
  in its own separate arena, never one *shared* address space spanning graph
  scopes. A subgraph's tensors can never alias something live in the outer
  scope even when their liveness would otherwise allow it; a genuinely joint
  plan is a materially bigger allocator problem this still doesn't take on.
- **A plan, not a runtime.** Nothing in onnxsim executes against this arena;
  it's up to whatever deployment target consumes the plan (or a future
  onnxsim pass) to actually allocate `arena_bytes` and place each tensor at
  its offset.
