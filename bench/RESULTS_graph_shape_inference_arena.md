# Arena allocation in graph-native shape inference (ProcessNode)

**Change:** `onnxsim/onnx#claude/graph-shape-inference-arena-alloc` — arena-allocates
`ProcessNode`'s per-node-visit scratch messages (`NodeProto` plus its attribute
tree, and the per-input `TypeProto`/`TensorProto` adapters) in
`onnx/common/graph_shape_inference.cc`, the same treatment `constant_folding.cpp`'s
`RunOps` already got for its throwaway sub-model.
**Reproduce:** `bench/graph_shape_inference_teardown_bench.sh` (isolated
microbenchmark) and `ONNXSIM_PROFILE=trace.json onnxsim <model> <out>
--skip-optimization` (end-to-end, via onnxsim's built-in profiler).

## TL;DR

| Measurement | Before | After | Speedup |
|---|---:|---:|---:|
| Isolated message build+destroy, typical node (3 inputs, 2 attrs, rank 4, 1 constant input) | 1.11 us/op | 0.71 us/op | **1.56x** |
| Isolated message build+destroy, larger node (8 inputs, 4 attrs, rank 6, 2 constant inputs) | 5.35 us/op | 1.69 us/op | **3.17x** |
| Isolated **teardown only**, typical node | 1.22 us/op | 0.14 us/op | **8.7x** |
| End-to-end `InferShapes` phase, 10,000-node synthetic model (mostly trivial `Relu`), avg of 3 runs | 100.8 ms | 89.8 ms | **1.12x** |

The isolated numbers show what the arena change actually removes: per-sub-message
heap frees on every node visit. The end-to-end number is smaller because it's
diluted by everything else `ProcessNode` also does (schema lookup, symbol-table
bookkeeping, the schema's own inference function) and because the synthetic
model here is dominated by `Relu` -- the cheapest, smallest node shape the
microbenchmark tests. A model with more attribute-heavy ops (`Conv`, attention
patterns) or `If`/`Loop`/`Scan` subgraphs (a full body export per visit) would
see a bigger share of the isolated win reflected end-to-end.

## Isolated microbenchmark

`bench/graph_shape_inference_teardown_bench.cpp` builds the same
`NodeProto`/`TypeProto`/`TensorProto` tree `ProcessNode` builds for one node
visit -- without going through the real schema registry / inference functions,
for the same reason `fold_teardown_bench.cpp` isolates `RunOp`'s arena change
from ONNX Runtime session-creation noise: an end-to-end measurement mixes in
costs the change doesn't touch, which would drown the message-construction
delta. `iters=200000` for all three shapes below.

```
--- tiny node (1 input, 0 attrs, rank 2, no const) ---
[full cycle]  heap=0.283 us/op   arena=0.217 us/op   speedup=1.31x
[teardown]    heap=0.202 us/op   arena=0.145 us/op   speedup=1.39x

--- typical node (3 inputs, 1 attr-pair, rank 4, 1 const input) ---
[full cycle]  heap=1.111 us/op   arena=0.712 us/op   speedup=1.56x
[teardown]    heap=1.221 us/op   arena=0.140 us/op   speedup=8.69x

--- larger node (8 inputs, 4 attrs, rank 6, 2 const inputs) ---
[full cycle]  heap=5.347 us/op   arena=1.685 us/op   speedup=3.17x
[teardown]    heap=2.481 us/op   arena=0.286 us/op   speedup=8.69x
```

Two things stand out:

- **Teardown speedup is roughly constant (~8.7x) once a node has any real
  structure**, and grows with sub-message count -- exactly the "one bulk free
  vs. walk-and-free-each-submessage" effect the change targets.
- **Full-cycle speedup grows with node complexity** (1.3x -> 1.56x -> 3.17x
  tiny -> typical -> larger), because construction cost is common-mode
  (arena vs. heap allocation of the same number of objects costs about the
  same to *build*) while teardown cost is where the two diverge, so bigger
  nodes spend a larger fraction of their per-visit time in the now-cheap part.

## End-to-end: onnxsim's own `InferShapes` phase

A synthetic 10,000-node model (`Relu` chain with an interspersed `Constant` +
`Reshape` every 4th node, to also exercise the constant-input/`TensorProto`
path) run through `onnxsim`'s CLI with `--skip-optimization` and
`ONNXSIM_PROFILE` set, so the built-in profiler reports the cumulative
`InferShapes` span across the whole run (3 fixed-point rounds x 10,000 nodes =
30,000 `ProcessNode` calls per run). Three trials each, before/after:

```
before (heap):   100.41 ms, 105.71 ms, 96.24 ms   -> avg 100.79 ms
after  (arena):   87.16 ms,  88.87 ms, 93.31 ms   -> avg  89.78 ms
```

**~11% faster** on this node mix. `FoldConstant` (which spins up an ONNX
Runtime/reference-evaluator session per foldable node) dominates total wall
time in this synthetic model by two orders of magnitude and is unaffected by
this change -- it's excluded from the comparison above by using
`--skip-optimization` and reading only the `InferShapes` line.

## Correctness

Both variants were also checked for correctness, not just speed: a plain
Reshape-shape-inference case and an `If` node with divergent then/else branch
shapes (exercising the subgraph-attribute-export path `AddAttributeForInference`
covers) produce identical, `onnx.checker`-valid output before and after: same
inferred shapes, same graph structure. See the PR description for the full
verification notes (wheel build, targeted pytest subset, smoke tests).
