# The stem Gather's real leading `Reshape`: it compiles, it runs correctly, but its index table is not a simple shift of the Reshape-free layout

`docs/axera-conv-gather-loose-ends.md`'s item 3 built the real
`Reshape_466 -> (Gather -> Mul)*14 -> Concat` chain -- the ResNet18 stem
Gather's actual leading `Reshape` (`[16,3,224,224] -> [16,1,3,50176]`, a pure
flatten of the raw image), never included in any composition test before
(`docs/axera-gather-compose-real-scale.md` and
`docs/axera-stem-gather-rechunk.md` both built the Reshape-free
`[16,1,3,50176]` shape directly as a graph input) -- and left the result
"inconclusive within this session's time budget: still building." The build
completed after that report was filed. This records what it found.

## The build

Same source graph `docs/axera-stem-gather-rechunk.md`'s working 14-chunk
template compiled (`/home/takecheeze/npu-scratch/t_stem_gather_rechunk/
rechunk14/t.onnx`), with one `Reshape` node prepended and the graph input
changed from `x[16,1,3,50176]` to the real `image[16,3,224,224]`. Same
uniform `[-2.5,2.5]` assumed calibration range (no real calibration images
were available in this environment either, matching every prior doc's same
caveat), scratch under `/home/takecheeze/npu-scratch/t_stem_gather_reshape/
rechunk14_reshape`.

**Total build time: about 3h13m**, roughly 2.4x PR #1767's 79 minutes for the
Reshape-free graph. Every stage's own duration, from the Pulsar2 log:

| stage | duration |
| --- | --- |
| `build op serially` | 11m31s |
| `add ddr swap` | 29s |
| `calc input dependencies` | 2m09s |
| `calc output dependencies` | 1h20m59s |
| `assign eu heuristic` | 57m28s |
| `assign eu onepass` | 5m30s |
| `assign eu greedy` | 5m32s |
| `build jobs` | 6m58s |
| `assemble model` | ~12m24s |

`calc output dependencies` was the dominant stage in PR #1767 (49m51s there);
here `assign eu heuristic` is a close second contributor that PR's
much-smaller Reshape-free graph did not spend meaningful time in at all. Both
scale with the graph's 618,737 tracked tensors (identical count to the
Reshape-free build -- the one extra node did not change this number, so the
slowdown is from wall-clock/scheduling variance and the added `Reshape`
node's effect on the dependency graph shape, not from tracking more tensors).

## It compiles to one fused `neu mode` node, same as every other composition test in this project

Nodes: `['neu mode']`. Input `image[16,3,224,224]`, output `y[16,1,3,614656]`
-- the real interface, with the `Reshape` fully absorbed. `npu_params` is
2,820,608 bytes, 80 bytes (20 words) longer than the Reshape-free reference's
2,820,528.

## It is numerically correct on device

Ran on the AX8850 (`axcl-vm`, serialized under the shared lock) with a
uniform `[-2.5,2.5]` random image, the real index vector and mask read
directly from the source ONNX graph's own `Gather`/`Mul` initializers (in
`Concat`-input order, not assumed to be simple node-declaration order):

```
control: n=29,503,488  max_err=0.0098  mean_err=0.0048
```

This matches the noise floor every other in-range Gather check in this
project reports (0.003-0.01), including the Reshape-free composition itself.
**The real, complete image -> stem-Gather pipeline is correct.**

## But its `npu_params` index encoding is not a simple shift of the Reshape-free layout

The 80-byte length difference is small enough to suggest a plausible
hypothesis: a small metadata block (20 words) for the `Reshape` got inserted
before the same index table the Reshape-free build already uses, and
retargeting would still be "rewrite the leading (shifted) N words." Tested
directly (`scripts/axera/stem_gather_reshape_check.py`, checking every shift
from 0 to 39 words): **the best match, at any shift, is 1.53% of index
words** -- not the near-100% a pure-shift layout would give. A direct byte
diff of the two tables (Reshape-graph shifted by the 20-word length
difference, vs. the Reshape-free reference) shows 43.2% of bytes differing.

So the index encoding was genuinely **rewritten**, not shifted, once the real
`Reshape` became part of the compiled unit -- consistent with every other
composition finding in this project (`docs/axera-transpose-compose-real.md`,
`docs/axera-conv-compose-real.md`): Pulsar2 does not leave op-shaped or
even shape-preserving seams once ops are fused into one `neu mode` node.

## What this means

- **The complete real pipeline (image in, stem Gather out) works,
  unmodified.** That was never confirmed before this doc; every prior stem
  Gather check either used the pre-flattened `[16,1,3,50176]` shape directly,
  or a smaller sub-scale proxy.
- **The existing retargeting recipe does not reach this composition.**
  `memory_emit.py`'s and `stem_gather_compose_emit.py`'s "rewrite the leading
  N index words, keep everything else" trick, validated for the Reshape-free
  build, cannot be applied here without first decoding where (or whether) the
  indices are recoverably encoded in this table -- not attempted in this
  session. Index retargeting for the *real*, fully-composed stem is an open
  question again, even though it is closed for the artificially-scoped
  Reshape-free version.
- This closes `docs/axera-conv-gather-loose-ends.md`'s item 3 with a real
  answer (compiles, runs correctly, retargeting layout changed) rather than
  the "still building" status recorded there.

## Reproduction

`scripts/axera/stem_gather_reshape_check.py` reproduces the index-layout
finding from the two committed fixtures
(`scripts/axera/fixtures/stem_gather_reshape/reshape14_reference.axmodel.gz`
and the existing `scripts/axera/fixtures/gather_compose_real/
stem_rechunk14_reference.axmodel.gz`), no Docker or device needed.
`tests/test_axera_stem_gather_reshape_check.py` covers both numbers as
regression tests. The device run and the source `t.onnx`/index-extraction
script used above are scratch, not committed
(`/home/takecheeze/npu-scratch/t_stem_gather_reshape/rechunk14_reshape`).
