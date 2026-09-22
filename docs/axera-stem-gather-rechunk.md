# The real-scale stem Gather+Mul composition: fixed, built, and verified

`docs/axera-gather-compose-real-scale.md` left one thing open: the ResNet18
stem's real `Reshape -> Gather(614,656 indices) -> Mul(mask)` chain, at full
scale, with `docs/axera-stem-gather.md`'s own 7-chunk `Gather`+`Concat`
template. It did not finish compiling in over an hour and was abandoned, and
a smaller experiment found a real, separate blocker along the way: a
standalone 87,808-element `Mul` fails outright with `integer 87807 does not
fit 'uint16_t'`. This resolves both.

## The precise limit

A quick standalone sweep around the boundary pins it down exactly:

| last-axis size | standalone `Mul` |
| --- | --- |
| 43,904 | builds |
| 65,535 | builds |
| **65,536** | **builds** |
| 65,537 | fails: `integer 65567 does not fit 'uint16_t'` |
| 87,808 | fails: `integer 87807 does not fit 'uint16_t'` |

So the elementwise engine's per-axis addressing limit is **65,536 elements,
inclusive** -- not 65,535 as the ambiguous phrasing in the earlier doc left
it. This is a hard code-generation limit distinct from the OCM byte-budget
tiling `dma_tile_predict.py` and `docs/axera-stem-gather.md`'s own Gather
job-size accounting both already found; a tensor can be well under either
budget in bytes and still fail here if one axis has too many elements.

## The fix: chunk the `Mul`, not just the `Gather`

The original template does `Gather * 7 -> Concat -> Mul`, so the `Mul` still
sees the full, concatenated 614,656-element axis regardless of how finely the
`Gather` itself is chunked. The fix moves `Concat` to the end:

```
(Gather(x, idx_i) -> Mul(g_i, mask_i))  for i in 0..13  ->  Concat -> y
```

`614656 = 2^8 * 7^4`. Its largest divisor at or below 65,536 is **43,904**,
which divides evenly into **14** chunks -- every `Gather` and every `Mul` job
stays comfortably under both the addressing limit and the OCM byte budget (a
43,904-element `Gather` job is about 1.68 MB by `docs/axera-stem-gather.md`'s
own accounting, versus the 3.14 MB budget; the existing 87,808-element
`Gather` chunks already proved `Gather` itself does not hit the 16-bit
addressing limit the way `Mul` does).

## It compiles, but the same compiler stage is slow regardless of chunking

Built with the real index vector and mask sliced from
`/home/takecheeze/npu-scratch/t6-r18fold/step.onnx` (same data
`docs/axera-gather-compose-real-scale.md` extracted), Pulsar2 7.0-lite,
AX650, MinMax calibration on 4 samples uniform in `[-2.5, 2.5]` (the same
assumed activation range as the earlier doc -- no real calibration images
were available in this environment either):

**It compiled in about 79 minutes.** Almost all of that is one stage:

| compiler stage | wall time |
| --- | --- |
| everything up to and including `calc input dependencies` | ~9 min |
| `calc output dependencies` | **49m51s** |
| `assign eu heuristic` / `onepass` / `greedy` | ~8 min |
| `results2model_gen` through `assemble model` | ~5 min |
| `fuse 1 subgraph(s)` (final write) | ~7 min |

That `calc output dependencies` stall is the *same* symptom the abandoned
unchunked attempt showed (silent, no file writes, high CPU, for 40+ minutes
before that session gave up) -- confirmed here by watching for file-write
activity every few minutes and seeing none for over 45 minutes straight,
followed by a burst of progress once that stage's own internal work finished.
So: **the `uint16_t` fix was necessary to get past the immediate crash, but
the long compile time is a separate, apparently graph-size-driven cost in a
later scheduling stage** (309,613 items tracked through several of these
passes -- this graph, at 29 ONNX nodes over 14 chunks, tracks noticeably more
than the original 9-node/7-chunk graph would have, which may be why this
took even longer than the abandoned attempt's first 22 minutes to reach the
same point, though the abandoned attempt itself never got past it). Building
this is a one-time cost paid once per template, not per retargeting -- the
emitter itself is instant, the same as every other Gather template in this
project.

## Verification

AX8850 in `axcl-vm`, one lock-serialized session, ground truth
`np.take(x, indices, axis=3) * mask`, three checks plus a health run after
each:

| test | model | input range | max err | mean err | elements &gt;0.1 err |
| --- | --- | --- | --- | --- | --- |
| control | reference (own indices) | in calib range `[-2.5,2.5]` | 0.0098 | 0.0048 | 0 / 29,503,488 |
| retargeted | reference patched to a shuffled index vector | in calib range | 0.0098 | 0.0048 | 0 / 29,503,488 |
| out-of-range control | reference (own indices) | `[-5,5]` (exceeds calib) | 2.51 | 0.62 | 13,948,859 / 29,503,488 (47.3%) |

The first two rows are **numerically identical**, confirming at true full
scale (614,656 elements, not the 32,768-element slice the earlier doc used)
the same finding: retargeting this Gather's indices under a `{0,1}`-mask
`Mul` consumer is safe without recalibration, because the mask cannot expand
the value range past the Gather's own output. The third row reproduces the
generic "input exceeds calibration range" failure at the same magnitude the
earlier mid-scale test found (max err ~2.5, ~47% of elements over 0.1 error)
-- confirming this is an ordinary calibration-range issue, not something the
rechunking changed or added. Every device call, including four separate
health runs across this session, was clean -- no `0x8030070C` fault.

The `npu_params` layout survives composition, chunking, and full scale
unchanged: the first 614,656 little-endian uint32 words are the index
vector, in order, the same layout every other emitter in this project
assumes.

## What's shipped

`scripts/axera/stem_gather_compose_emit.py`'s `emit(output_path,
indices=...)` retargets this exact 14-chunk composed template (structural
validation, then the same "rewrite the leading N words" operation every
other Gather emitter performs). Its own output was device-verified directly
(not just an equivalent hand-rolled patch), matching the control run's error
profile exactly. `tests/test_axera_stem_gather_compose_emit.py` covers the
round-trip, template preservation, and rejection paths without Docker or a
device.

The fixture, `scripts/axera/fixtures/gather_compose_real/
stem_rechunk14_reference.axmodel.gz` (182 KB, index words zeroed by
`memory_emit.py`'s `make_gather_fixture`, reused unmodified since its logic
is structure-agnostic), is bigger than the smaller single-Gather templates
elsewhere in this project -- the 14 real mask constants (614,656 floats
total) and the larger MCode (29 fused ONNX nodes instead of 1) don't
compress as well as a mostly-zero index table.

## Not covered

- Only the `{0,1}`-mask `Mul` consumer was tested, same as the earlier doc.
  An aggregating consumer (`MatMul`, `Conv`, a reduction) still needs
  `docs/axera-compose.md`'s calibration-range warning, untested at this
  scale.
- The leading `Reshape` is still not included (same scope cut as both
  earlier docs).
- Whether a different chunk count (e.g. closer to the 65,536 ceiling, if one
  existed as a divisor -- it does not for 614,656) would meaningfully change
  the `calc output dependencies` cost was not tested; 43,904/14 was the only
  configuration built.
- The real calibration range is still an assumption (`[-2.5, 2.5]`), not
  measured from real images.
