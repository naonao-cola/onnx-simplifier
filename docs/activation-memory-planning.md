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
generally safe to alias with its inputs' storage (only true in-place ops
allow that, and this allocator has no notion of which ops are in-place-safe).

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

The input model is never mutated; `annotate_memory_plan` returns a shape-
inferred copy, since the per-value metadata needs a matching `value_info`
entry to attach to (a `NodeProto` doesn't get its own offset — a node's
*output value* does, keyed by tensor name, same as the metric annotations).

## Scope (v1)

- **Concrete shapes only.** A tensor with a dynamic `dim_param` dimension has
  no fixed byte size to place, so it's reported in `unplanned` rather than
  guessed. Run `--overwrite-input-shape` / `overwrite_input_shape=` (see the
  main README) to pin a dynamic model's shapes before planning it.
- **Top-level graph only.** Tensors inside `If`/`Loop`/`Scan` subgraph bodies
  are not visited or planned at all — a possible follow-up is a joint
  cross-subgraph plan the way `PeakMemoryFootprint` adds a subgraph's own
  peak on top of its owning node's live set, but sharing offset space
  *across* graph scopes is a materially bigger allocator problem than this
  first version takes on.
- **A plan, not a runtime.** Nothing in onnxsim executes against this arena;
  it's up to whatever deployment target consumes the plan (or a future
  onnxsim pass) to actually allocate `arena_bytes` and place each tensor at
  its offset.
