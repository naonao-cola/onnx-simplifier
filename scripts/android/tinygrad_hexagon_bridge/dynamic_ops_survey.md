# Survey: could the dynamic-shape ops be ported to hand-written HVX kernels?

Planning document only -- no kernel code here. Covers the four op types in Mask R-CNN's `rest.onnx`
that have genuinely data-dependent behavior: **proposal decode** (RPN box-delta application),
**TopK**, **NonMaxSuppression**, **RoiAlign**. A separate, concurrent effort is building an actual
kernel for the box-head fc6 MatMul (the one *static-shape* GEMM-like op in this same graph) --
not covered here.

All findings below come from running `scripts/android/maskrcnn_e2e/prepare.py` fresh (the
Hugging Face model was already cached locally) and inspecting the real `rest.onnx` graph directly
via `onnx.shape_inference`/constant-tracing -- not from assumptions about generic Mask R-CNN
architecture. Real values are called out explicitly; anything not directly verified from the graph
is marked as such.

## What the real graph actually contains

`rest.onnx`: 2405 nodes, dominated by data-movement ops (755 `Gather`, 246 `Unsqueeze`, 202
`Squeeze`, 123 `Slice`, 96 `Shape`) around a compute core of: **85 `NonMaxSuppression`**, **94
`NonZero`**, **85 `ConstantOfShape`**, **8 `RoiAlign`**, **7 `TopK`**, **8 `ScatterElements`**, plus
5 `Conv` + 1 `ConvTranspose` (mask head) and 4 `MatMul` (box head).

One finding worth flagging even though it's outside this survey's four ops: tracing every mask-head
`Conv`/`ConvTranspose` and box-head `MatMul` weight input back through the graph shows **all of
them are genuinely QDQ-quantized** (`int8` weight initializers behind `DequantizeLinear`, activations
routed through matching `QuantizeLinear`/`DequantizeLinear` pairs) -- the same convention the
backbone uses, not a plain fp32 subgraph as the "first-ever fp32 op" framing around `sigmoid`
might suggest by analogy. The one exception: the mask head's single `ConvTranspose` (256->256,
2x2 kernel) has a **plain fp32 weight initializer**, no Q/DQ wrapping at all -- genuinely
unquantized, unlike its neighboring `Conv`s. Relevant context for whoever picks up mask-head
coverage later; not something this survey resolves further.

## 1. Proposal decode (RPN box-delta application) -- LOW difficulty, do this first

**Real shapes, verified via shape inference**: this model's RPN operates on 5 FPN levels at fixed
per-level anchor counts (800x1088 input, 3 anchors/location, standard scheme) -- confirmed by
walking the graph back from each level's `TopK` input to its unconsumed source tensor:

| FPN level | Spatial size | Anchors (3/location) |
|---|---|---:|
| P2 | 200x272 | **163,200** |
| P3 | 100x136 | **40,800** |
| P4 | 50x68 | **10,200** |
| P5 | 25x34 | **2,550** |
| P6 | 13x17 | **663** |

All five are fully static -- fixed the moment the input size (800x1088, already fixed throughout
this whole project) is fixed. The delta-application arithmetic itself (`Exp`/`Sub`/`Mul`/`Div`/
`Concat`/`Clip`, the standard anchor-center + `exp(dw,dh)*anchor_wh` -> corner-coordinates decode)
runs on **every anchor at every level**, before any `TopK`/NMS narrows the set -- i.e. this op's
shape is determined entirely by the fixed input size, with zero dependence on runtime data.

**Dtype, verified by tracing an `Exp` node's input back through the graph and checking its
`value_info` element type**: **float32** (`elem_type: 1`), dequantized from `int8` immediately
upstream. Not quantized arithmetic -- the same complexity class `sigmoid` and `hex_add_kernel.py`
already established for this project's fp32-on-Hexagon path (tinygrad's normal codegen does not
reach for HVX vector width on this backend by default for fp32 elementwise ops; a hand-written
`custom_kernel`, not the default codegen path, was what actually won for both of those).

**Why this is the easiest of the four**: no reduction, no accumulator, no data-dependent
addressing anywhere -- every output element's source address is statically known at compile time,
exactly like `hex_add_kernel.py`/`hex_requantize_kernel.py`. It's a longer elementwise chain than
either of those (five/six ops instead of one), but the same `Ops.CUSTOM`/`dtypes.void` idiom this
project has used since `hex_gemm_kernel.py` applies directly -- no new kernel-construction
capability needed, just more elementwise steps than any single existing kernel chains together.

**Recommendation**: tractable, and the natural first target if this thread continues. Effort
estimate: modestly bigger than `hex_add_kernel.py` (more ops chained, and needs the same
`libgcc.a` soft-float link `sigmoid`'s fix already landed for `Exp`/`Div` if they emit libcalls --
check first, don't assume), well short of anything requiring new kernel-construction machinery.

## 2. RoiAlign -- MODERATE difficulty, second priority

**Real params, verified from the `RoiAlign` node attributes directly**: two groups, both fully
fixed-shape per invocation --

- Box head: **7x7** output crop, `sampling_ratio=2`, `avg` pooling, one node per FPN level
  (`spatial_scale` = 0.25, 0.125, 0.0625, 0.03125 for P2-P5 -- 4 of the 5 levels feed RoiAlign,
  matching the standard "P6 used for RPN anchors only, not RoI pooling" convention).
- Mask head: **14x14** output crop, same `sampling_ratio=2`/`avg`/4-level structure.

**The real difficulty, precisely**: two genuinely new things, not one --
1. **Data-dependent source addressing.** Every sample point's feature-map read location depends on
   that RoI's own (runtime) box coordinates. Every kernel in this project so far -- including
   `hex_stem7x7_kernel.py`'s 49-position gather -- reads from a compile-time-fixed offset pattern;
   RoiAlign's offsets are runtime values computed from data. This is a real, unprecedented
   requirement: indirect/gather addressing from a computed index, not a static one.
2. **Bilinear interpolation.** A 4-tap weighted sum with data-dependent (non-integer) weights,
   structurally simple once the 4 source addresses + weights are known, but new arithmetic --
   nothing in this project has done sub-pixel interpolation before.

**Why it's more tractable than TopK/NMS despite being genuinely new work**: the *shape* of the
problem is still fully static -- 7x7 or 14x14 output positions, always a 2x2=4-sample grid per
position (`sampling_ratio=2`, fixed). Only the *address* each sample reads from is dynamic, not
the amount of work or the control flow. That isolates the hard part cleanly: a **hybrid split** --
compute each RoI's sample addresses + bilinear weights on the host/CPU (cheap scalar arithmetic,
already tolerant of a dynamic RoI count since it's not bulk data movement), then run a fixed-shape
HVX kernel that does the actual gather + weighted-sum + pool for up to a compile-time-max RoI
count (the same padding-to-a-fixed-bound trick `hex_gemm_kernel.py`'s `cout=12`/`cout=3` coverage
and `hex_stem7x7_kernel.py`'s `cin=3` padding already established, masking invalid RoIs past the
real count) -- is a concrete, buildable design with this project's existing toolkit.

**Recommendation**: the most tractable of the three remaining "hard" ops, but not a small task.
Effort estimate: bigger than `hex_maxpool_kernel.py` (needs real indirect addressing, which is
new), comparable in scope to the backbone-subgraph chaining work's Stage 1-2 (`hex_requantize_kernel.py`
+ the fused stem/maxpool driver) -- both had to solve one genuinely new problem class, not just
apply an existing pattern to a new shape.

## 3. TopK -- HIGH difficulty

**Real k values, verified by resolving each `TopK` node's `k` input to a constant (or confirming
it's not resolvable, i.e. genuinely dynamic)**:

| Node | Input count | k | Fixed? |
|---|---:|---:|---|
| P2 pre-NMS select | 163,200 | 1000 | yes |
| P3 pre-NMS select | 40,800 | 1000 | yes |
| P4 pre-NMS select | 10,200 | 1000 | yes |
| P5 pre-NMS select | 2,550 | 1000 | yes |
| P6 pre-NMS select | 663 | 663 (pass-through) | yes |
| Final detection cap (x2) | dynamic | **not a resolvable constant** | **no** |

Four of five per-level selections use a **fixed k=1000** (verified, not assumed) -- so "how many
outputs" is a compile-time constant for the bulk of TopK's real use in this graph. The two
downstream nodes (the final per-image detection cap, matching torchvision's default
`box_detections_per_img`) have a genuinely data-dependent `k` -- confirmed by failing to resolve
either to a constant via the same initializer/`Constant`-node tracing used everywhere else in this
survey.

**Why the fixed-k cases are still hard, precisely**: the padding-to-a-fixed-bound trick that
resolves RoiAlign's/proposal-decode's shape concerns does *nothing* here, because TopK's core
difficulty was never "how many outputs" -- it's **selecting which k of n inputs are the largest**,
which needs a real sort or partial-selection algorithm. HVX has no direct top-k/sort instruction.
The realistic options: (a) a full parallel sort (bitonic sort is the standard SIMD-friendly choice,
O(n log^2 n) compare-exchange steps -- real, substantial, and nothing like it exists anywhere in
this project's kernel vocabulary, which has never needed inter-element comparison/reordering
before -- every existing kernel's data movement is a fixed, independent-per-output-element
pattern), or (b) iterative max-and-mask selection (k passes of "find the max, remove it" -- much
simpler code, but k=1000 passes over up to 163,200 elements is a lot of serial work, likely too
slow to be worth building). The two dynamic-k final-cap nodes add the padding-trick complexity
*on top of* this core difficulty, not instead of it.

**Recommendation**: not a good near-term target. This needs a genuinely new kernel-construction
primitive this project has no precedent for -- closer in kind to the `Ops.WMMA`/`TensorCore` dead
end this project already hit and correctly abandoned in favor of a from-scratch approach (see
`README.md`'s "Three more attempts" section) than to any of the shape-variant kernels built since.
Effort estimate: bigger than anything built in this project so far, not a shape variant of an
existing kernel but a new capability.

## 4. NonMaxSuppression -- HIGHEST difficulty, not recommended as a near-term target

**Real structure, verified by resolving every NMS node's constant inputs and tracing its score
tensor's shape/lineage**: **85 nodes = 80 per-class detection-stage NMS (`iou_threshold=0.5`) +
5 per-FPN-level RPN-stage NMS (`iou_threshold=0.7`)** -- confirmed exactly (80+5=85, matching the
op count precisely, not approximately) via a histogram of each node's real `iou_threshold` constant.
`max_output_boxes_per_class=2000` uniformly (a generous cap, not the real effective limit -- the
downstream dynamic-k TopK nodes above are what actually bound final detection count to
`box_detections_per_img`). Box/score counts feeding every NMS node are genuinely dynamic
(`unk__NNN` symbolic dims in shape inference), since they follow the dynamic-count TopK/earlier-NMS
stages.

**Why this is the hardest of the four, precisely**: NMS's greedy suppression is **inherently
sequential** -- whether candidate box *i* survives depends on which higher-scoring boxes were
*already kept*, decided in strict score-sorted order. This is a genuine data-dependent recurrence
(a `while` loop over a shrinking candidate set, exactly matching the reference NumPy implementation
in `scripts/android/maskrcnn_e2e/tinygrad_ops.py`'s `NonMaxSuppression()`: `while len(order) and
len(kept) < max_out`), not a fixed-iteration parallel reduction. The fixed-upper-bound padding
trick bounds the loop's *iteration count* (cap at `min(max_output_boxes_per_class, input_count)`,
mask past the real count) but does **not** remove the sequential dependency itself -- iteration
i's suppression decision genuinely depends on iteration i-1's outcome. This is a fundamentally
different problem shape than every kernel this project has built: every existing reduction loop
accumulates, but accumulation *order* never changes *which* future iterations touch *which* data;
NMS's does.

**One real, graph-structure-specific lever, not a generic mitigation**: the 80 per-class NMS
instances are mutually independent (each class's box list is suppressed against itself only) --
so while any *single* class's suppression loop stays sequential, the 80 classes' loops could run
**in parallel with each other**, one suppression-loop iteration processed across all 80 class-lists
per step instead of 80 fully serial passes. This is a real opportunity specific to this model's
actual structure (found from the graph, not assumed), but it does not remove the core difficulty --
it just amortizes it 80-wide.

**Recommendation**: not a good target without first establishing whether tinygrad's
`custom_kernel`/`UOp.range` machinery can even express a data-dependent loop bound / early-exit at
all -- that's a prerequisite research question, not a kernel-design question, and this project has
no existing example of a data-dependent (as opposed to compile-time-fixed) loop anywhere in its
kernel vocabulary. Effort estimate: the largest of the four by a wide margin -- plausibly needs new
tinygrad-side capability before any kernel code is possible, the same category of prerequisite work
(not just a bigger kernel) that made `Ops.WMMA` a dead end for the vrmpy accumulator problem.

## Ranked recommendation

1. **Proposal decode** -- low difficulty, fully static shape, same idiom as existing elementwise
   kernels, just fp32 (confirmed, not assumed) so needs the same vectorized-`custom_kernel`
   treatment `sigmoid`/`hex_add_kernel.py` already established. Start here.
2. **RoiAlign** -- moderate difficulty, genuinely new (indirect addressing) but well-isolated to
   address computation; a host-computes-addresses + HVX-gathers-and-averages hybrid is a concrete,
   buildable design with tools this project already has.
3. **TopK** -- high difficulty, needs a real parallel sort/selection primitive with no precedent in
   this project's kernel vocabulary. Four of five real uses have a fixed k=1000, which helps with
   *shape* but not with the core *selection-algorithm* difficulty.
4. **NonMaxSuppression** -- highest difficulty, inherently sequential/recurrent control flow, a
   different problem class from everything built so far. Likely needs new tinygrad-side capability
   (research question) before it's even expressible, not just more kernel-construction effort.
