# Three small loose ends, closed: a real Pulsar2 scheduling limit, a confirmed calibration recipe, and a real leading-Reshape check

Three precisely-scoped open items from this session's Conv and Gather work,
done in priority order.

## Item 1: `Conv(512,256,1,1)`'s device fault is not a `patch_scales.py` bug -- it is real, pre-existing Pulsar2 scheduling divergence

`docs/axera-conv-256to512-tiled-fix.md` (PR #1790) isolated the fault to
`patch_scales.py`'s mcode-literal patch, finding 1,548 differing bytes
between the patched model and a real native rebuild, and speculated Pulsar2's
"real scheduling for this specific shape apparently diverges between
same-shape builds by more than scale literals alone." This confirms that
precisely, with a control that settles it:

Using the same held-out weights and the committed reference/native builds
under `scripts/axera/fixtures/conv_256to512_tiled_fix/` and
`/home/takecheeze/npu-scratch/t_conv_learn_256to512/c1x1_holdout` (a genuine
native Pulsar2 compile of the holdout weights, not an emitted artifact):

| comparison | differing bytes (of 10,360) |
| --- | --- |
| unpatched reference mcode vs. native holdout rebuild | **1,548** |
| `patch_scales.py`-patched mcode vs. native holdout rebuild | **1,548** (unchanged) |
| unpatched reference vs. patched (i.e. what the patch itself wrote) | **16** |

`patch_scales.py`'s own slot search reports 12 slots at this shape (32 bytes
of intended writes; only 16 bytes actually change value -- some slots'
old/new bf16 truncations share a byte). Patching moves **zero** bytes closer
to the native rebuild: the count is identical before and after. The entire
1,548-byte divergence already exists between the *unpatched* reference and a
plain native compile of different weights/scales -- it is intrinsic to how
Pulsar2 schedules this exact shape (`Conv(x[16,256,14,14],
w[512,256,1,1], stride 2)`) differently for different scale values, not
something introduced by, or fixable within, `patch_scales.py`'s
substring-search-and-replace approach. Its own docstring already names this
class of thing as out of scope ("the scheduling-divergent bytes, which no
value edit can reach") -- this is the first shape in the project where that
caveat is empirically confirmed to bind, with exact numbers.

**No device run was needed for this diagnosis.** The static byte accounting
above is deterministic and conclusive; PR #1790 already confirmed the fault
itself on-device, and re-triggering it would add no new information.

**Practical note, offered at low confidence:** this shape's native mcode is
unusually small (10,360 bytes total, vs. 1.17 MB for the `3,3` sibling in the
same pair) -- a cheap, checkable flag for "verify a patched artifact on
device before trusting it" on any future shape that compiles unusually small
relative to its weight count, though this is one data point, not a rule.

**Verdict: still open, and not fixable by correcting `patch_scales.py`'s
patching logic.** The only correct fix for this shape is a full native
rebuild whenever its scale changes -- which is exactly the cost
`patch_scales.py` exists to avoid, so this shape is a real counterexample to
the tool's general applicability, not a bug in it.

## Item 2: does Gather's "calibrate on the full input range" safety recipe hold for a real `Add`-sum aggregator, not just `MatMul`?

`docs/axera-gather-aggregate-real.md` (PR #1761) validated the recipe
(narrow-calibration retargeting is unsafe; calibrating the Gather's *input*
across its full valid range fixes it) against a real `Gather -> Mul(mask) ->
Reshape -> MatMul` chain. No Gather in the real training step
(`/home/takecheeze/npu-scratch/t6-r18fold/step.onnx`) feeds an `Add` or
`ReduceSum` within three hops -- checked directly by walking the graph's
producer/consumer edges, not assumed; `legalize.py`'s tap-rewrite functions
also build taps from `Slice`, not `Gather`, confirming PR #1761's own note.
So this item is **synthetic**, constructed to mirror PR #1761's real shape
and methodology as closely as possible: `Gather(x[16,1,512,49], idx[448],
axis=3) -> Mul(mask) -> Reshape([16,1,512,4,112]) -> Slice*4 -> Squeeze*4 ->
Add -> Add -> Add` (a genuine 4-way tree-sum of gathered/masked groups, the
same shape family and the same structured two-band calibration range PR
#1761 used: input positions 0-38 drawn from `[-0.3,0.3]`, 39-48 from
`[-0.9,0.9]`).

Three Pulsar2 7.0-lite (AX650, MinMax) builds, scratch under
`/home/takecheeze/npu-scratch/t_gather_aggregate_addsum`:

- **`a_reference_narrow`**: index set drawn entirely from the "small" band
  (`idx = arange(448) % 39`), narrow/structured calibration.
- **`b_native_adversarial`**: a fresh native build with an index set drawn
  entirely from the "large" band (`idx = 39 + arange(448) % 10`), same
  structured calibration -- the ground truth for "what the retargeted output
  should be."
- **`c_wide_reference`**: `a`'s index set, but `x`'s calibration is uniform
  `[-0.9,0.9]` at every position (the "wide" recipe).

`npu_params`' leading 448 little-endian uint32 words are exactly that
build's own index array in every build (confirmed directly) -- the
"leading words are indices" layout holds again, for a fourth real op
combination now (2-op mask, 4-op `MatMul`, and this 9-op `Add`-tree chain).
`gather_compose_real_check.patch_indices` was reused unmodified: this
table is 3,636 bytes, a whole number of uint32 words, so PR #1759's
word-truncation bug (found for a 10,861-byte, non-whole-word table) does not
apply here and no workaround was needed.

**Device result (AX8850, `axcl-vm`, serialized under the shared lock,
control run before, health run after -- both clean, `0.0217` max err, ruling
out drift):**

| variant | max err vs. the correct (`idx_adv`) output |
| --- | --- |
| `b_native_adversarial` (ground truth) | 0.0372 |
| `a` retargeted to `idx_adv`, narrow calibration | **2.1891** |
| `c` retargeted to `idx_adv`, wide calibration | **0.0387** |

The failure mode reproduces exactly for a genuine `Add`-sum aggregator, at
essentially the same severity PR #1761 found for `MatMul` (roughly 60x the
correct error there; roughly 59x here). Wide-range calibration on the
Gather's input fixes it just as completely: `0.0387` is inside normal
device-to-device noise of the `0.0372` ground truth, not a residual error.

**Verdict: confirmed.** PR #1761's recipe -- calibrate the Gather's input
across its full valid range at every position the indices could select, not
the aggregate's range and not only the currently-used indices' positions --
is not `MatMul`-specific. It is a property of *aggregation* generally: any
op that can combine multiple gathered elements into one output value
(summing, whether via `Add` or a matrix contraction) can produce an output
magnitude the reference's own narrow calibration never saw, regardless of
which specific op does the combining.

## Item 3: the stem Gather's real leading `Reshape`

`docs/axera-gather-compose-real-scale.md` and `docs/axera-stem-gather-rechunk.md`
both built the stem's `[16,1,3,50176]` shape directly as a graph input,
skipping the real `Reshape_466` (`[16,3,224,224] -> [16,1,3,50176]`, a pure
flatten of the raw image) that precedes it in the actual training step.

Built the real full chain -- `Reshape_466 -> (Gather -> Mul)*14 -> Concat`,
matching PR #1767's already-working 14-chunk template exactly, with the one
real `Reshape` prepended -- from
`/home/takecheeze/npu-scratch/t_stem_gather_rechunk/rechunk14/t.onnx` (the
exact source graph that template used), same uniform `[-2.5,2.5]` assumed
calibration range PR #1767 used (no real calibration images were available
in this environment either, matching that doc's own caveat), scratch under
`/home/takecheeze/npu-scratch/t_stem_gather_reshape/rechunk14_reshape`.

**Status: inconclusive within this session's time budget -- still building,
not a stall.** The build ran a full 3+ hours (started, real forward progress
confirmed throughout, never left running under the sandbox): `tiling op`,
`build op serially` (11m31s), `build op`, `add ddr swap`, `calc input
dependencies` (2m09s), then `calc output dependencies` -- the same stage PR
#1767 found dominant (49m51s there for the Reshape-free graph) -- completed
here in **1h20m59s**, about 60% slower with the one extra node. After that,
memory grew sharply to a stable 50.5 GiB plateau (from ~11 GiB) and CPU
stayed pinned at 100% for the next 20+ minutes with no further progress line
and no crash, consistent with a job-count-dependent stage (`assign eu
heuristic/onepass/greedy`, silent until it finishes, exactly like `calc
output dependencies` was) working through this graph's 618,737 tracked
tensors -- the same count PR #1767's Reshape-free build also reports, so
this is not new from the added node, just a graph this scale was already
known to produce.

No evidence of a hang was found (CPU never dropped to idle, memory never
oscillated, the container never exited), so this is reported as **real,
unfinished work**, not a negative result: the leading `Reshape`'s effect on
retargeting safety is not yet checked, one way or the other. The build was
left running rather than killed; whether it eventually completes was not
observed in this session. Continuing this item means either resuming to
check on that same build, or budgeting several hours uninterrupted for a
fresh one.

## Reproduction

Item 1's byte accounting: `scripts/axera/conv_scale_divergence_check.py`
(uses the already-committed fixtures from PR #1790 plus a new one,
`c1x1_holdout_native.axmodel.gz`, no new Pulsar2 build needed). Item 2:
`scripts/axera/gather_aggregate_addsum_check.py build [NAME ...]` writes the
three ONNX graphs and calibration data; `check` confirms the index-layout
invariant from already-built outputs. `tests/test_axera_conv_gather_loose_ends.py`
covers both without Docker or a device, against committed fixtures under
`scripts/axera/fixtures/conv_gather_loose_ends/`. Item 3's source graph is
`/home/takecheeze/npu-scratch/t_stem_gather_reshape/rechunk14_reshape/t.onnx`
(built by prepending a `Reshape` node to
`/home/takecheeze/npu-scratch/t_stem_gather_rechunk/rechunk14/t.onnx`); no
fixture or module was committed for it since the build never reached a
verifiable artifact in this session.
