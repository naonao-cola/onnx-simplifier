"""Post-training weight pruning for MatMul/vanilla-Gemm and Conv layers.

Surveying the pruning literature against what onnxsim can actually act on
(an exported ONNX graph, no training loop, no gradients, usually no labels)
narrows the field a lot. Most well-known pruning *tools* --
``torch.nn.utils.prune``, NNI's pruning API, Neural Magic's SparseML, Intel
Neural Compressor's pruning API -- assume a live framework model mid-training
or at least a fine-tuning loop to recover accuracy after each pruning step
(iterative magnitude pruning / the Lottery Ticket Hypothesis, movement
pruning, "pattern lock" pruning, ...). That is the same reason onnxsim's
existing weight-only quantization stack (:mod:`onnxsim.gptq`,
:mod:`onnxsim.awq`, ...) reimplements each technique's *algorithm* against
raw ONNX MatMul/Gemm weights rather than depending on those libraries
directly: they operate one level up, on a model object onnxsim never has.

*Structured* pruning (removing whole channels/filters, e.g. Torch-Pruning,
NNI's L1/L2 filter pruning, network slimming, or the expert-intermediate-
channel/Mamba-state pruning inside NVIDIA's "Iterative Puzzle" compression
pipeline for hybrid MoE LLMs, https://arxiv.org/abs/2607.04371) is a
fundamentally bigger project than the rest of this module for two separate
reasons, and this module only takes on one of them. It *does* change tensor
shapes, which ripples through every downstream consumer of the pruned
dimension -- real graph surgery, not the self-contained per-layer weight
rewrite every other ``apply_*``/``quantize_*`` pass in onnxsim is. That part
:func:`apply_structured_pruning` takes on, but deliberately only for the
narrowest topology where the surgery is unambiguous: a single MatMul/Gemm
or ordinary (``group=1``) Conv whose output feeds, through a chain of
shape-preserving elementwise ops (activations, and for MatMul/Gemm also a
bias/scale add/mul) with no other consumer anywhere along that chain, into
exactly one downstream layer of the same family whose reduction/input-
channel dimension matches.
Any multi-consumer fan-out or branch (which needs real dependency-graph
analysis -- what Torch-Pruning's DepGraph does in general) is left
untouched rather than guessed at. One narrow, bounded slice of the
residual/skip-connection case *is* handled -- see
:func:`_find_conv_residual_chains`'s own section comment for the full
reasoning -- because it turns out to have a provably-safe special case: a
Conv chain whose forward walk hits a channel-preserving ``Add(a, b)`` with
two non-constant operands (every residual connection's shape) is, rather
than declined outright, treated as a merge point requiring whichever real
Conv producer(s) feed `a` and `b` to be pruned to one shared channel-index
set, found by walking backward from each operand (through the same
unary-activation/depthwise-pass-through hops the forward walk already
allows) to a real ``group=1`` Conv producer -- or, transitively, to
*another* such `Add` merge point, unioning that one's own group in too (the
"many residual blocks share one spine" case). This is still bounded, not
general DepGraph: it holds every hop, on every branch, to the same
single-consumer safety bar as the rest of this pass, so a real multi-block
ResNet's shared "post-block" tensor -- read by both the next block's own
first Conv *and* directly by that block's own `Add` -- is exactly the kind
of fan-out this still declines rather than guesses at, leaving an
*interior* block of a deep residual stage untouched; what it does reach is
a single residual connection whose branches don't fan out elsewhere (e.g. a
projection-shortcut block, or a stage's last block before something else
consumes its output) and a genuinely linear stack of `Add`-only merges.

The exact same construction is repeated for MatMul/Gemm chains -- see
:func:`_find_matmul_residual_chains`'s own section comment for the full
reasoning, which mirrors the Conv case's above closely enough that only the
differences are worth restating here: the backward walk mirrors
:func:`_walk_to_consumer`'s own *wider* MatMul/Gemm hop set (unary
activations plus a per-channel bias/scale ``Add``/``Mul`` against a
constant, not just unary activations) rather than the Conv walk's narrower
one, since MatMul/Gemm has no depthwise-Conv-style transparent pass-through
hop at all. This is exactly the residual-stream shape every current
transformer block takes (``x = x + SelfAttn(LN(x))``, ``x = x + MLP(LN(x))``)
-- previously declined outright, now reached the same way a Conv
projection-shortcut block is. Two compositions were checked and found not
safe to fold in silently, so both are declined the same conservative way as
everything else in this pass: a gated (SwiGLU/GeGLU) combine feeding a
residual branch directly, with no output-projection MatMul in between (the
backward walk has no way to pick "the" one branch through a `Mul` of two
non-constant operands without silently dropping the other's contribution);
and a residual branch that would need to cross a fused self-attention op
boundary (`Attention`/`GroupQueryAttention`, see the "Attention-head
pruning" section far below) to reach a real producer, with no
output-projection MatMul between the attention op and the `Add` either.
Both are narrow, exporter-dependent shapes rather than the common case --
an FFN's own down-projection or an attention block's own output projection
feeding the residual `Add` (overwhelmingly the normal shape) needs no
special handling at all, since the backward walk stops at that projection's
own MatMul/Gemm node without ever looking further upstream at what feeds
it.

A bare `Add` merge point is, however, the exception rather than the rule
once a transformer has actually been run through onnxruntime's own
transformer-optimizer tool: that pass fuses each residual `Add` (plus an
optional per-channel bias `Add`) together with the *following*
`LayerNorm`/RMSNorm into one ``com.microsoft::SkipLayerNormalization``/
``SkipSimplifiedLayerNormalization`` node, so
:func:`_match_matmul_residual_merge` recognizes that fused node as an
eligible merge point too -- its `input`/`skip` inputs playing `Add`'s own
two-operand role -- while also treating it as a per-channel affine hop
whose `gamma` (required) and, if present, `beta` (``SkipLayerNormalization``
only, dropped by the RMSNorm variant) and `bias` get sliced by the group's
own `keep` set alongside everything else. See
:func:`_find_matmul_residual_chains`'s own section comment for the exact
fused arithmetic (confirmed against onnxruntime's own kernel source and by
direct execution) and how a non-constant/tied `gamma`/`beta`/`bias`, or a
consumed optional `mean`/`inv_std_var` output, is declined the same
conservative way as everything above.

The same optimizer tool typically fuses an FFN block's own bias-add and
activation together too, the same way it fuses a residual `Add` into
`SkipLayerNormalization` above: ``com.microsoft::BiasGelu(A, B) = Gelu(A +
B)`` (erf-based, the common case) and ``com.microsoft::FastGelu(X[, bias])``
(the tanh-approximated Gelu, with `bias` optional) both collapse an ordinary
``MatMul -> Add(bias) -> Gelu`` FFN hop into one node -- confirmed against
onnxruntime's own schema (`contrib_defs.cc`) and CPU kernel (`bias_gelu.cc`)
and by direct execution. Without also recognizing these, an FFN chain this
whole feature exists for (``up = MatMul(x, W1); h = BiasGelu(up, Bias1);
down = MatMul(h, W2)``) would fail at that one hop and the whole chain would
go unpruned, the same gap the `SkipLayerNormalization` fix above closed for
residual connections. :func:`_walk_to_consumer`/
:func:`_walk_matmul_producer_backward` (the forward and backward MatMul/Gemm
hop walkers) both recognize a `BiasGelu`/`FastGelu` node the same way they
already recognize a bias/scale `Add`/`Mul` hop -- see
:func:`_match_fused_bias_gelu` and `_FUSED_BIAS_GELU_OPS`'s own comment --
sliced by the same `keep` set alongside everything else; a non-constant bias
(`BiasGelu`'s own schema requires one; `FastGelu`'s is optional) declines
the node outright, never guessed at. This is a MatMul/Gemm-chain-only hop,
deliberately not extended to Conv chains: a real Conv already carries any
bias in its own third input (see the Conv paragraph below), and neither
fusion targets Conv graphs in practice. `com.microsoft::QuickGelu(X) = X *
Sigmoid(alpha * X)` -- the third Gelu-family fusion the same optimizer tool
emits, used by some model families in place of `BiasGelu`/`FastGelu` -- is a
simpler case still: it takes no bias operand at all (`alpha` is a node
*attribute*, not an input), so it is exactly as unary/shape-preserving as
`Gelu`/`Sigmoid` already in `_UNARY_PASS_THROUGH`, and is matched by simply
being added to that set -- extending every walker that already consults it
(both MatMul/Gemm and Conv, forward and backward alike, plus
:func:`_trace_gate_producer_backward`'s own gated-pair gate-activation
matcher below) for free, with no dedicated hop machinery needed. A gated
(SwiGLU/GeGLU) pair's own gate branch fused into `BiasGelu`/`FastGelu`
specifically (as opposed to plain `Gelu`/`Sigmoid`, or the now-unary
`QuickGelu`) is *not* recognized by :func:`_trace_gate_producer_backward`,
and is left out of scope deliberately rather than extended: that tracer only
ever walks back through single-input unary ops, with nowhere on
:class:`_Producer` to carry a gate-branch-local bias constant the way
`_Chain.chain_ops` already does for the shared post-combine chain, so
supporting it would need new machinery (a `_Producer`-local `chain_ops`
counterpart), not a one-line addition like `QuickGelu`'s. It is also the
narrower case in practice: a gated FFN's gate projection commonly carries no
bias at all (e.g. Llama-family linear layers), and when it does, only a
*separate* `Add`+`Gelu` on that branch is even eligible for onnxruntime's
own fusion pass to collapse into `BiasGelu`/`FastGelu` in the first place --
a plain unfused `Gelu`/`Sigmoid` gate (already handled) is the shape that
survives when there's no bias to fuse to begin with.

General multi-branch dependency-graph pruning -- arbitrary fan-out, non-Add
merges (`Concat`, ...) -- remains out of scope. The
other part of the paper's pipeline -- an architecture *search* over what to prune,
alternated with knowledge-distillation/RL recovery afterwards -- needs a
training loop onnxsim does not have and is not in scope here at all; this
is a single, static, no-retraining structural cut, closer in spirit to Li
et al.'s L2-norm filter pruning (below) than to anything iterative.

What *does* fit that mold, and needs no retraining loop: post-training
*unstructured* (or semi-structured N:M) pruning, à la magnitude pruning
(Han et al., 2015, "Learning both Weights and Connections for Efficient
Neural Networks", https://arxiv.org/abs/1506.02626) and, for the
calibrated variant, Wanda (Sun et al., 2023, "A Simple and Effective
Pruning Approach for Large Language Models",
https://arxiv.org/abs/2306.11695 -- the pruning analogue of this module's
neighbors :mod:`onnxsim.awq`/:mod:`onnxsim.smoothquant`: a single forward
pass over calibration data, no weight update, no backward pass at all).
Both zero out individual weight entries and leave every tensor's shape
exactly as it was, so -- like every ``quantize_weight_only_*`` pass here --
the result is a plain ONNX model, correct by construction (a MatMul/Gemm
with some zeroed entries computes the same op, just with less nonzero
data), that a runtime with sparse-kernel support (or a later, separate
dense-to-sparse repacking step) can exploit for speed.

:func:`apply_magnitude_pruning` uses ``|W|`` as the importance metric and
needs no calibration data at all -- the simple, data-free baseline.
:func:`apply_wanda_pruning` weights that by each input feature's activation
norm over calibration data (``|W_ij| * ||X_j||_2``), which -- per the
Wanda paper -- better protects weights that multiply high-magnitude
activations even when the weight itself is individually small, the same
class of outlier-activation effect that motivates :mod:`onnxsim.smoothquant`.

Both also match 2-D ``Conv`` weights, not just MatMul/vanilla-Gemm: a
Conv's ``[out_channels, in_channels/group, kH, kW]`` weight is reshaped to
``[out_channels, (in_channels/group)*kH*kW]`` -- the same convention
:func:`apply_structured_pruning` already uses for Conv filter importance
below -- and each output filter becomes one comparison group, exactly like
a MatMul/Gemm output channel. Unlike :func:`apply_structured_pruning`'s
producer/consumer chain matching below, this reshape-and-rank-per-filter
operation is completely agnostic to ``group``: it never touches another
layer's channel indices, only ranks each output filter's own row against
itself, so ordinary (``group=1``), depthwise (``group == in_channels ==
out_channels``), and general grouped (``group`` neither 1 nor the channel
count) Conv are all matched identically here by
:func:`_match_conv_weight_only` -- a materially different, and much
easier, bar than the shape-changing coupling problem
:func:`_match_conv_producer`/:func:`_match_conv_consumer` decline part of
below. For magnitude pruning that's the entire story: ``|W|`` on the
reshaped weight, mask computed, reshaped back, working unchanged for every
``group``. For Wanda it needs one more step, since ``X_j`` isn't simply
"input feature ``j``" once a sliding kernel is involved: ``j`` indexes one
``(in_channel, kh, kw)`` offset within the receptive field, so its
activation statistic is the norm of the *im2col-unfolded* input patch
value at that specific offset, over every output spatial position and
calibration sample -- computed via ``numpy.lib.stride_tricks.
sliding_window_view`` on the zero-padded input (per the Conv node's own
``pads``/``strides`` attributes) rather than materializing an explicit
im2col matrix. A Conv whose attributes aren't a combination this
confidently handles (non-default ``auto_pad``, since ``SAME_*``/``VALID``
padding depends on input spatial size rather than being fixed per node;
or non-all-ones ``dilations``, since a dilated receptive field's offsets
aren't evenly spaced in the padded input the way ``sliding_window_view``
assumes) falls back to plain magnitude for that layer, the same as any
other layer whose activation norm was never observed.

For a grouped or depthwise Conv, Wanda's per-offset activation norm needs
one more piece of care beyond the reshape above: that norm is always
computed once from the *raw, full-channel* input (:func:`_conv_patch_sq_sum`
never looks at `group` at all, and doesn't need to -- unfolding the whole
input once is cheaper than unfolding it again per group), but a grouped
Conv's output filter ``i`` only ever *reads* its own group's
``in_channels/group``-wide slice of that input (filter ``i`` belongs to
group ``i // (out_channels/group)``, per ONNX's grouped-Conv weight
layout), so "local receptive-field offset ``j``" names a *different*
global input channel depending on which group filter ``i`` falls in.
Sharing one norm row across every filter -- correct, and what this module
did before grouped Conv was matched here at all, when `group` was always 1
-- would silently score every filter outside group 0 against the wrong
channels' statistics for any `group` > 1. :func:`_conv_group_relative_norm`
is the fix: it slices the full-input norm's ``[Cin, kh, kw]`` shape along
its channel axis once per group and repeats each group's own slice across
exactly the filter rows belonging to it, before that expanded,
per-filter-row norm ever reaches the ``|W_ij| * ||X_j||_2`` importance
computation -- collapsing to the previous single-shared-row behavior
exactly when ``group=1``. Verified against a dedicated test engineering one
group's calibration input to have deliberately different activation
statistics from another group's, confirming each filter's resulting mask
reflects its own group's statistics and not another group's or a global
average (``test_wanda_pruning_conv_grouped_uses_own_groups_activation_norm``).

:func:`apply_sparsegpt_pruning` also matches Conv layers, but -- unlike
magnitude and Wanda above -- only ordinary (``group=1``) ones; see its own
docstring below for the full ``[K, K]`` im2col cross-covariance Hessian
this needed (a real step up from Wanda's per-offset norm above, not just a
reuse of it), how it's verified, and why extending it to grouped/depthwise
Conv was deliberately left undone rather than guessed at: doing so
correctly needs a *per-group* Hessian (each group's own filters only ever
see their own group's input-channel patches -- the same channel-slicing
subtlety Wanda's grouped support above handles, but now for the full
cross-covariance rather than a per-offset norm, and needing the sequential
column-processing/error-compensation loop partitioned per group rather
than run once across the whole weight) -- comparable in scope to the
original SparseGPT+Conv work this module already did from first
principles, and not undertaken speculatively here.

All three unstructured/N:M functions also match the two fused self-
attention ops the "Attention-head pruning" section below performs
*structural* (whole-head) pruning on -- ``com.microsoft::Attention``'s
merged QKV weight (``[K, Nq+Nk+Nv]``, matched by
:func:`_match_attention_weight_only`, reusing :func:`_match_attention_producer`'s
own criteria) and ``com.microsoft::GroupQueryAttention``'s separate Q/K/V
projections, which need no special-casing at all: per
:func:`_match_gqa_producer`'s own docstring they are ordinary MatMul/
vanilla-Gemm nodes feeding into that op, not weights the op itself owns, so
:func:`_candidates`' existing MatMul/Gemm matching already reaches them
(ranked no differently from any other MatMul/Gemm layer). This is a
completely different code path from head pruning below: it zeros
individual weight entries within a head's columns rather than removing
whole heads, exactly as useful on its own (e.g. reaching NVIDIA Ampere's
2:4 sparse Tensor Cores, via :func:`onnxsim.convert_matmul_to_gemm`, on an
already-``fuse_attention``'d model's QKV weight) as it is combined with
head pruning first. Since unstructured/N:M pruning only ever zeros values
and never changes shape, an ``Attention`` node's ``num_heads``/
``qkv_hidden_sizes`` attributes -- which describe the merged weight's
column layout, not any zeroed-vs-nonzero distinction within it -- can never
drift out of sync with the (unchanged) weight shape, the same invariant
this holds for every other matched layer type.

Both magnitude and Wanda pruning support two sparsity patterns, chosen per
invocation:

- unstructured: for every output row (comparison group), the lowest-
  importance entries are zeroed until that row reaches the target
  ``sparsity`` fraction.
- semi-structured N:M (e.g. ``n=2, m=4`` -- NVIDIA Ampere's 2:4 structured
  sparsity, the pattern Wanda's own paper evaluates most): within every
  consecutive group of ``m`` input-channel entries in a row, only the
  ``n`` highest-importance survive.

:func:`weight_sparsity` reports the fraction of exact-zero entries across
every matched layer's weight, as a quick way to confirm a pruning call
reached its target (or to measure an already-sparse model).

:func:`apply_structured_pruning` actually removes channels (real shape
reduction, real FLOP/parameter reduction on any runtime, no sparse-kernel
support needed) from every producer -> consumer chain it can prove safe to
cut, per output-channel L2-norm importance (Li et al., 2017, "Pruning
Filters for Efficient ConvNets", https://arxiv.org/abs/1608.08710) -- for a
MatMul/Gemm chain, that criterion is a transplant from Conv filters to
output channels (the same one :func:`apply_magnitude_pruning`/
:func:`apply_wanda_pruning` already made for Han et al./Wanda's element-wise
criteria); for a Conv chain it is the paper's own original setting, applied
directly: each output filter's full ``[in_channels, kH, kW]`` kernel is
flattened and ranked by its own L2 norm. Conv support is deliberately
narrower than the MatMul/Gemm path in one respect that stays true
regardless of grouping: producers/consumers are joined by unary activations
alone -- no per-channel ``Add``/``Mul`` scale-or-bias op, since a real Conv
already carries any bias in its own optional third input, and
``BatchNormalization`` is expected to already be fused into the preceding
Conv's weight by the time this pass runs (onnxsim's own default
optimization does exactly that, see ``fuse_bn_into_conv``), so a raw
per-channel affine between two Convs isn't a shape this pass special-cases.

Within that, three ``group`` shapes are distinguished. Ordinary
(``group=1``) Conv is the base case already described above. The
*depthwise* special case (``group == in_channels == out_channels``, weight
``[C, 1, kH, kW]``) is different: with one filter per channel and no
cross-channel mixing at all, output channel ``i`` depends only on input
channel ``i``, so a depthwise Conv sitting between a chain's real producer
and real consumer needs no independent importance of its own -- the chain
walk (:func:`_walk_to_conv_consumer`) crosses it transparently, like one
more shape-preserving activation hop, carrying whatever channel-index set
survives upstream straight through unchanged, while still slicing that
depthwise layer's own weight/bias by the same indices and shrinking its
``group`` attribute to match. This is exactly the ``Conv(1x1, group=1) ->
DepthwiseConv(3x3, group=C) -> Conv(1x1, group=1)`` "inverted residual"
block MobileNet/EfficientNet-style efficient CNN backbones use throughout,
so it's worth the special case; a depthwise Conv is never itself matched as
a producer or consumer (see
:func:`_match_conv_producer`/:func:`_match_conv_consumer`), only ever a
transparent hop between two real Conv boundaries -- one sitting last before
a graph output or an unhandled branch simply ends the chain unmatched, same
as any other topology this pass declines to guess at.

A *general* grouped Conv (``group`` neither 1 nor equal to its channel
count, weight ``[out_channels, in_channels/group, kH, kW]``) is the
remaining case, and -- unlike the fully-out-of-scope treatment an earlier
version of this pass gave it -- is now matched as a real producer and/or
consumer, because its structure turns out to be tractable in a way general
dependency-graph coupling isn't: since a grouped Conv's ``group`` blocks
never mix (full mixing *within* a block, none across), pruning block ``k``
is completely independent of every other block, as long as the *same
count* is pruned from every block (so ``channels % group == 0`` survives,
exactly as ONNX's Conv schema requires). Concretely:

- **As a producer**, its output-channel axis is flat/global regardless of
  grouping (grouping only ever splits the *input* axis), so each of its
  ``group`` output-filter blocks is ranked and pruned independently by the
  same per-filter L2-norm criterion above, applied within each block's own
  slice rather than across the whole ``out_channels`` axis -- keeping the
  same count from every block.
- **As a consumer**, its input-channel axis *is* per-group-relative (weight
  column ``j`` on a filter belonging to group ``g`` means global input
  channel ``g * (in_channels / group) + j``, not global channel ``j``), so
  slicing it needs dedicated per-group-relative logic
  (:func:`_slice_grouped_consumer_conv_weight`) rather than the flat
  column selection an ordinary consumer's weight uses -- but the *set* of
  surviving channels is still whatever the chain's producer side decided,
  now constrained to keep a uniform count within each of the consumer's own
  ``group`` blocks.
- **Composing the two**: a grouped producer feeding an ordinary
  (``group=1``) consumer is supported -- the consumer imposes no grouping
  constraint of its own, so the producer's own per-block selection is
  already all that's needed. An ordinary producer feeding a grouped
  consumer is likewise supported -- the producer has no grouping constraint
  of its own either, so its selection is simply constrained to the
  consumer's own block boundaries instead of an unconstrained global top-k.
  Both sides grouped is supported *only* when both share the exact same
  ``group`` count (their blocks then partition the shared channel count
  identically, so either side's per-block selection already satisfies the
  other); a mismatched ``group`` count on the two sides is declined
  outright and the whole chain is left untouched, since the two sides'
  block boundaries then wouldn't generally align at all, and reconciling
  that would need real cross-chain bookkeeping this pass does not attempt
  (the same kind of boundary attention-head pruning below draws around
  GroupQueryAttention, and general residual/branch dependency-graph
  coupling is left out of this pass entirely). See
  :func:`_match_conv_producer`/:func:`_match_conv_consumer`/
  :func:`_chain_group` for the exact matching and selection logic.
:func:`apply_structured_wanda_pruning` is the calibrated upgrade of that
same technique -- ``||W_row||_2 * ||X||_2`` per channel instead of weight
magnitude alone -- exactly the same relationship Wanda has to plain
magnitude pruning, transplanted from individual weights (or, for Conv,
whole filters) to whole channels. Because either changes shapes, the
result is unconditionally irreversible and, unlike a retrained pipeline,
has no distillation/RL step to recover whatever accuracy the cut costs --
evaluate the result before shipping it, the same caution any lossy onnxsim
pass deserves.

:func:`apply_sparsegpt_pruning` is a third, more accurate way to reach an
unstructured or N:M pattern (alongside magnitude and Wanda pruning above):
SparseGPT (Frantar & Alistarh, 2023, "SparseGPT: Massive Language Models
Can Be Accurately Pruned in One-Shot", https://arxiv.org/abs/2301.00774) --
the pruning sibling of :mod:`onnxsim.gptq`, from the same authors, reusing
the exact same machinery (:func:`onnxsim.gptq._inverse_hessian_cholesky`'s
Cholesky-factored inverse Hessian, and the same left-to-right,
error-propagating column processing) but pruning each column to a mask
instead of quantizing it to a grid point. Where magnitude/Wanda pick a
mask once from a static (weight- or weight-times-activation-) importance
score and stop, SparseGPT computes each column's OBS-style saliency score
``w_ij^2 / Hinv_jj^2`` from calibration data, then -- after masking a
column -- propagates the resulting reconstruction error into every
not-yet-processed column via the same Hessian-based correction GPTQ uses
for quantization error, so later columns compensate for earlier ones'
removal instead of every column being scored independently against the
original, uncorrected weights. This reliably beats magnitude/Wanda at the
same sparsity, at the cost of needing calibration data (there is no
data-free variant, unlike magnitude vs. Wanda) and being noticeably more
expensive per layer (one Cholesky factorization plus a sequential,
Hessian-propagating pass over every column, rather than one static
element-wise score). Ported directly from the reference implementation's
``fasterprune`` (https://github.com/IST-DASLab/sparsegpt), including one
behavior that's otherwise a departure from every other function in this
module: for *unstructured* sparsity, the reference selects one threshold
per ``proc_block_size``-wide column block, shared across every output row
in that block, rather than :func:`apply_magnitude_pruning`/
:func:`apply_wanda_pruning`'s per-row threshold -- faithfully reproduced
here rather than "corrected" to match, since the point of this function is
to reproduce SparseGPT specifically. N:M pruning is unaffected (it is
already per-row in the reference too, and matches this module's own
``n``/``m`` convention exactly).

Unlike Conv, ``com.microsoft::Attention``'s merged QKV weight is **not**
excluded from :func:`apply_sparsegpt_pruning`'s candidate list (nor, since
they were never excluded to begin with, are ``GroupQueryAttention``'s
separate Q/K/V projections -- see above). SparseGPT's correctness rests on
``H`` accurately capturing which columns of *this weight's own input*
correlate; unlike a Conv's im2col-unfolded receptive field, a merged QKV
weight's input is the same plain ``[*, K]`` activation feeding an ordinary
MatMul, with the same ``H = X^T X`` this function already computes for
every other MatMul/Gemm layer, no new numerical machinery needed. What each
output column of that weight is later used for downstream (split into
Q/K/V, fed into a fused attention kernel) has no bearing on the linear-
algebra correctness of pruning *this* layer's own weight against *this*
layer's own input -- exactly why it would be inconsistent to include GQA's
separate Q/K/V MatMuls (already unconditionally in scope, being ordinary
MatMul/Gemm nodes) while excluding Attention's merged one on some notion of
"this is an attention weight, treat it specially": neither needs it.

Wanda's calibrated metric, by contrast, still falls back to plain magnitude
for ``Attention``'s merged weight, the same as it already does for any
MatMul/Gemm layer whose activation isn't a plain 2-D tensor at the probe
point (see :func:`apply_wanda_pruning`'s own docstring): the op's own `X`
input is rank-3 (``[batch, seq, hidden]``), and generalizing that one check
to reduce over leading axes (mirroring :func:`apply_sparsegpt_pruning`'s
own ``x.reshape(-1, x.shape[-1])``) would also change the documented,
tested fallback behavior of every *existing* MatMul/Gemm layer whose own
activation happens to be rank 3+ (a batched-sequence input is an entirely
ordinary shape for a plain linear layer too, not something unique to
Attention) -- a strictly larger behavior change than this task asked for,
and not a free one: a per-row Wanda mask computed from a 2-D activation
reduces over the batch axis alone, while the same computation over a
reshaped 3-D activation reduces over batch *and* sequence position, a
different (arguably more correct, but different) statistic. Left as a
data-free-baseline fallback rather than changed silently underfoot; a
future pass that wants Wanda calibration for a rank-3 activation is free to
generalize this deliberately, as its own decision.

:func:`apply_sparsegpt_pruning` also matches ordinary (``group=1``) 2-D
``Conv`` layers, using exactly the same producer matching, Cholesky
machinery, and column-processing loop above -- the only genuinely new
piece is how ``H`` itself gets built. For a MatMul/Gemm layer
``H = X.T @ X`` is already a full cross-covariance, since each column
*is* an independent input feature; for Conv, a weight column instead
indexes one ``(in_channel, kh, kw)`` receptive-field offset (the same
reshape :func:`apply_magnitude_pruning`/:func:`apply_wanda_pruning` and
:func:`apply_structured_pruning` already use), so a correct ``H`` needs
the full ``[K, K]`` cross-covariance of every offset against every other
-- not merely each offset's own norm the way Wanda's diagonal-only
``_conv_patch_sq_sum`` needs. :func:`_conv_im2col_patches` builds that:
the same zero-padded ``numpy.lib.stride_tricks.sliding_window_view``
unfolding :func:`_conv_patch_sq_sum` already does (reusing
:func:`_conv_spatial_attrs` for the padding/stride extraction, and
declining the same ``auto_pad``/non-unit-``dilations`` combinations that
function does -- in which case the layer is left completely untouched,
there being no data-free fallback for SparseGPT, Conv included), but
returning the actual ``[n_positions, K]`` patch matrix rather than
reducing straight to a per-offset sum of squares, so ``H = patches.T @
patches`` can be formed from it. Verified two independent ways before
being trusted here: a brute-force nested-loop oracle that builds the same
``[K, K]`` Hessian a completely different way (one Python triple-loop per
output position, accumulating an explicit outer product, rather than any
vectorized unfolding), the same bar
``test_conv_patch_sq_sum_matches_naive_nested_loop_oracle`` already set
for Wanda's per-offset norm; and, end to end, the same reconstruction-
error property the MatMul/Gemm path is already validated against -- a
SparseGPT-pruned Conv layer's output should reconstruct the float layer's
output at least as well as naive same-mask zeroing with no compensation,
on well-conditioned calibration data. Because a full patch matrix for a
realistic layer can be large (``n_positions`` grows with output spatial
size, not just channel count), ``H`` accumulates incrementally, one
calibration batch's own unfolded patches at a time (``H += patches.T @
patches``, each batch's patches discarded once folded in), rather than
ever concatenating every batch's patches into one array first the way the
MatMul/Gemm path above still concatenates its (much smaller, already
per-feature) 2-D activations. Unlike the reference implementation's own
``add_batch`` (https://github.com/IST-DASLab/sparsegpt/blob/master/
sparsegpt.py), which never actually unfolds a Conv2d activation at all --
its Conv branch reshapes only the *weight* (``W.flatten(1)``), and its own
driver scripts (``opt.py``, ``llama.py``, ...) never exercise a Conv layer
in the first place, since OPT/BLOOM/Llama have none -- there was no
correct reference to port here, unlike every other technique this module
ports from an upstream implementation; this is original, from-first-
principles machinery, held to the verification bar above precisely
because of that. Unlike :func:`apply_magnitude_pruning`/
:func:`apply_wanda_pruning`, this stays ``group=1``-only rather than also
matching depthwise/general grouped Conv -- see the earlier paragraph above
(right after Wanda's grouped-Conv support) for why extending this
specific Hessian to grouped Conv is a materially bigger, declined-for-now
undertaking rather than a mechanical extension of the matching alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.gptq import _inverse_hessian_cholesky
from onnxsim.smoothquant import _match_matmul_like


def _validate_pattern(sparsity: float, n: Optional[int], m: Optional[int]) -> None:
    if (n is None) != (m is None):
        raise ValueError("n and m must be given together (N:M pruning) or not at all")
    if n is not None and m is not None:
        if not (0 < n <= m):
            raise ValueError(f"require 0 < n <= m, got n={n}, m={m}")
    elif not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")


def _sparsity_mask(importance: np.ndarray, sparsity: float) -> np.ndarray:
    # Per-row (per-output-channel) threshold, matching Wanda's own
    # per-output comparison group rather than a single global threshold --
    # a layer with output-channel-dependent weight/activation scale would
    # otherwise have some rows pruned to nothing and others left untouched.
    rows, cols = importance.shape
    keep = max(1, round(cols * (1.0 - sparsity)))
    if keep >= cols:
        return np.ones((rows, cols), dtype=bool)
    order = np.argsort(importance, axis=1)
    drop = order[:, : cols - keep]
    mask = np.ones((rows, cols), dtype=bool)
    np.put_along_axis(mask, drop, False, axis=1)
    return mask


def _nm_mask(importance: np.ndarray, n: int, m: int) -> np.ndarray:
    """Row-wise N:M mask: within every consecutive group of ``m`` columns,
    keeps only the ``n`` highest-importance entries. A trailing partial
    group (fewer than ``m`` columns) keeps a proportional share (rounded,
    at least 1) instead of raising on a non-multiple-of-``m`` width.
    """
    rows, cols = importance.shape
    mask = np.ones((rows, cols), dtype=bool)
    full_cols = (cols // m) * m
    if full_cols:
        groups = importance[:, :full_cols].reshape(rows, full_cols // m, m)
        order = np.argsort(groups, axis=2)
        drop = order[:, :, : m - n]
        group_mask = np.ones_like(groups, dtype=bool)
        np.put_along_axis(group_mask, drop, False, axis=2)
        mask[:, :full_cols] = group_mask.reshape(rows, full_cols)
    tail = cols - full_cols
    if tail:
        keep = min(tail, max(1, round(n * tail / m)))
        tail_importance = importance[:, full_cols:]
        order = np.argsort(tail_importance, axis=1)
        drop = order[:, : tail - keep]
        tail_mask = np.ones((rows, tail), dtype=bool)
        np.put_along_axis(tail_mask, drop, False, axis=1)
        mask[:, full_cols:] = tail_mask
    return mask


def _match_conv_weight_only(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    allow_grouped: bool = True,
) -> Optional[Tuple[str, str]]:
    """If `node` is a 2-D ``Conv`` with a constant 4-D float32
    ``[out_channels, in_channels/group, kH, kW]`` weight, returns
    ``(x_name, weight_name)``. Mirrors the "Conv2D structured pruning"
    section's own :func:`_match_conv_producer` matching criteria, minus
    that function's bias handling -- magnitude/Wanda/SparseGPT never touch
    bias, only the weight, so there's nothing to validate there.

    Unlike :func:`_match_conv_producer`/:func:`_match_conv_consumer`, a
    grouped or depthwise Conv (``group != 1``) is matched here too when
    `allow_grouped` (the default, used by :func:`apply_magnitude_pruning`/
    :func:`apply_wanda_pruning`): those two functions' own ``group=1``
    restriction exists because *structured* pruning's producer/consumer
    channel-index coupling genuinely doesn't survive grouping (an output
    or input channel's index meaning depends on which of `group`'s blocks
    it falls into on each side of a chain -- see this module's own
    docstring). Nothing here inherits that problem: unstructured/N:M
    pruning never changes any shape or needs any cross-layer index
    agreement -- it only zeros individual weight entries within one output
    filter's own kernel, independently of every other filter -- and
    :func:`_prune_weight`'s ``w.reshape(n, cin*kh*kw)`` (`cin` here already
    being ``in_channels/group`` by ONNX's own grouped-Conv weight layout)
    ranks and masks each of the `n` output filters' rows exactly the same
    way regardless of `group`. `allow_grouped=False` (only
    :func:`apply_sparsegpt_pruning` passes this, see its own docstring for
    why) restores the ``group=1``-only match. When `allow_grouped` and
    ``group > 1``, ``out_channels % group`` must still be zero -- the
    standard grouped-Conv well-formedness requirement (`group` equal-sized
    output blocks) that :func:`_conv_group_relative_norm` also relies on to
    line up each output filter with its own group's input-channel slice.
    """
    if node.op_type != "Conv" or len(node.input) < 2:
        return None
    w_name = node.input[1]
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or w_init.data_type != onnx.TensorProto.FLOAT
        or len(w_init.dims) != 4
    ):
        return None
    group = _conv_group(node)
    if group < 1:
        return None
    if not allow_grouped and group != 1:
        return None
    if group > 1 and w_init.dims[0] % group != 0:
        return None
    return node.input[0], w_name


def _match_attention_weight_only(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, str]]:
    """If `node` is a ``com.microsoft::Attention`` node with a constant 2-D
    float32 merged QKV weight (``[K, Nq+Nk+Nv]``), returns
    ``(x_name, weight_name)``. Mirrors the "Attention-head pruning"
    section's own :func:`_match_attention_producer` matching criteria
    (including its ``num_heads``/``qkv_hidden_sizes`` consistency checks --
    reused verbatim rather than re-implemented, even though nothing here
    reads `num_heads` itself, so that a node this module's *structural*
    head-pruning functions would decline as malformed is declined the same
    way here), minus that function's bias handling -- magnitude/Wanda/
    SparseGPT never touch bias, only the weight, so there's nothing to
    validate there. Per :func:`_match_attention_producer`'s own docstring,
    the merged weight has no transpose attribute of its own -- it is
    already ``[K, N]``-shaped by construction -- so it is matched here
    exactly like a non-transposed MatMul weight (``weight_transposed =
    False`` in :func:`_candidates`' returned tuple); :func:`_prune_weight`'s
    existing non-Conv path handles that shape with no Attention-specific
    code at all. Unstructured/N:M pruning only ever zeros entries -- it
    never changes `w_name`'s shape -- so the un-pruned merged weight's
    ``num_heads``/``qkv_hidden_sizes`` attributes (read by every other
    consumer of this weight, e.g. onnx.checker and any runtime) never drift
    out of sync with its actual shape, the same invariant every other
    matched layer type already gets from this module's value-only rewrite.
    """
    info = _match_attention_producer(node, initializer_map)
    if info is None:
        return None
    return node.input[0], node.input[1]


def _candidates(
    graph: onnx.GraphProto, include_conv: bool = True, allow_grouped_conv: bool = True
):
    """Every MatMul/vanilla-Gemm node with a constant 2-D float32 weight
    (this already includes a ``com.microsoft::GroupQueryAttention`` node's
    separate Q/K/V projections: per :func:`_match_gqa_producer`'s own
    docstring they are ordinary MatMul/vanilla-Gemm nodes feeding into that
    op, not weights the op itself owns, so they need no special-casing here
    at all -- they are ranked and pruned no differently from any other
    MatMul/Gemm layer), plus:

    - every ``com.microsoft::Attention`` node's constant 2-D float32 merged
      QKV weight, matched by :func:`_match_attention_weight_only` -- unlike
      GQA's separate projections, this *is* a weight the op itself owns
      (``node.input[1]``), so it needs its own matcher; and
    - when `include_conv` (the default; :func:`apply_sparsegpt_pruning`
      passes ``False``, see its own docstring for why) -- every 2-D
      ``Conv`` node matched by :func:`_match_conv_weight_only`, which by
      default (`allow_grouped_conv`, also default) includes depthwise and
      general grouped Conv, not just ordinary (``group=1``) Conv --
      :func:`apply_sparsegpt_pruning` still matches Conv at all
      (`include_conv` stays ``True`` there) but passes
      ``allow_grouped_conv=False``, since its own Conv im2col-Hessian
      machinery is ``group=1``-only (see its own docstring).

    Returns ``(node, x_name, w_name, weight_transposed, is_conv)`` tuples;
    `weight_transposed` is always ``False`` (meaningless) for a Conv entry,
    whose output channel always lives on axis 0 of its fixed
    ``[out_channels, in_channels/group, kH, kW]`` layout, and likewise
    always ``False`` (this time literally correct, not just meaningless --
    see :func:`_match_attention_weight_only`) for an Attention entry.
    Attention matching is unconditional (not gated by `include_conv`, which
    -- per its own name and :func:`apply_sparsegpt_pruning`'s own docstring
    -- concerns only Conv's im2col-Hessian gap): see
    :func:`apply_sparsegpt_pruning`'s own docstring for why a merged QKV
    weight has no analogous problem and is deliberately included in its
    ``include_conv=False`` candidate list too.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    out = []
    for node in graph.node:
        match = _match_matmul_like(node)
        if match is not None:
            x_name, w_name, weight_transposed = match
            w_init = initializer_map.get(w_name)
            if (
                w_init is None
                or w_init.data_type != onnx.TensorProto.FLOAT
                or len(w_init.dims) != 2
            ):
                continue
            out.append((node, x_name, w_name, weight_transposed, False))
            continue
        if include_conv:
            conv_match = _match_conv_weight_only(
                node, initializer_map, allow_grouped=allow_grouped_conv
            )
            if conv_match is not None:
                x_name, w_name = conv_match
                out.append((node, x_name, w_name, False, True))
                continue
        attn_match = _match_attention_weight_only(node, initializer_map)
        if attn_match is not None:
            x_name, w_name = attn_match
            out.append((node, x_name, w_name, False, False))
    return out


def _prune_weight(
    w_init: onnx.TensorProto,
    weight_transposed: bool,
    importance_of_nk,
    is_conv: bool = False,
) -> None:
    w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
    if is_conv:
        # [out_channels, in_channels, kH, kW] -> [N, K], the same
        # output-channel-first, flattened-receptive-field reshape
        # :func:`_apply_chains` already uses for Conv importance.
        n, cin, kh, kw = w.shape
        w_nk = w.reshape(n, cin * kh * kw)
        mask = importance_of_nk(w_nk)
        w_pruned_nk = np.where(mask, w_nk, 0.0)
        w_new = w_pruned_nk.reshape(n, cin, kh, kw).astype(np.float32)
    else:
        dim0, dim1 = w.shape
        w_nk = w if weight_transposed else w.T  # [N, K], output channel first
        mask = importance_of_nk(w_nk)
        w_pruned_nk = np.where(mask, w_nk, 0.0)
        w_new = w_pruned_nk if weight_transposed else w_pruned_nk.T
        w_new = w_new.reshape(dim0, dim1).astype(np.float32)
    w_init.CopyFrom(onnx.numpy_helper.from_array(w_new, name=w_init.name))


# --- Conv im2col-unfolded activation statistics (Wanda only) -----------


@dataclass(frozen=True)
class _ConvSpatialAttrs:
    kh: int
    kw: int
    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int
    stride_h: int
    stride_w: int


def _conv_spatial_attrs(
    node: onnx.NodeProto, w_init: onnx.TensorProto
) -> Optional[_ConvSpatialAttrs]:
    """Extracts the padding/stride a Conv node's calibration input needs to
    be correctly im2col-unfolded for Wanda's per-``(in_channel, kh, kw)``
    activation norm (see this module's own docstring). Declines
    (``None``) on any attribute combination not confidently handled:

    - non-default ``auto_pad`` -- its ``SAME_*``/``VALID`` padding depends
      on the input's own spatial size, not something fixed per node the
      way an explicit ``pads`` (or the schema's all-zero default) is;
    - a non-all-ones ``dilations`` -- a dilated receptive field's
      ``(kh, kw)`` offsets aren't evenly spaced in the padded input the
      way ``numpy.lib.stride_tricks.sliding_window_view`` assumes below.

    Per this module's own docstring, it's better to leave such a layer's
    activation norm unobserved (falling back to plain magnitude for it,
    same as any layer whose calibration activation was never a usable
    shape) than to guess at either.
    """
    kh, kw = int(w_init.dims[2]), int(w_init.dims[3])
    auto_pad = "NOTSET"
    pads: Optional[List[int]] = None
    strides: Optional[List[int]] = None
    dilations: Optional[List[int]] = None
    for attr in node.attribute:
        if attr.name == "auto_pad":
            auto_pad = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
        elif attr.name == "pads":
            pads = list(attr.ints)
        elif attr.name == "strides":
            strides = list(attr.ints)
        elif attr.name == "dilations":
            dilations = list(attr.ints)
        elif attr.name == "kernel_shape":
            ks = list(attr.ints)
            if len(ks) != 2 or ks[0] != kh or ks[1] != kw:
                return None  # weight/attribute mismatch -- don't guess

    if auto_pad not in ("NOTSET", ""):
        return None
    if dilations is not None and dilations != [1, 1]:
        return None
    if pads is None:
        pads = [0, 0, 0, 0]  # ONNX Conv schema default
    if len(pads) != 4 or any(p < 0 for p in pads):
        return None
    if strides is None:
        strides = [1, 1]  # ONNX Conv schema default
    if len(strides) != 2 or any(s <= 0 for s in strides):
        return None

    return _ConvSpatialAttrs(
        kh=kh,
        kw=kw,
        pad_top=pads[0],
        pad_left=pads[1],
        pad_bottom=pads[2],
        pad_right=pads[3],
        stride_h=strides[0],
        stride_w=strides[1],
    )


def _conv_patch_sq_sum(
    x: np.ndarray, attrs: _ConvSpatialAttrs
) -> Tuple[Optional[np.ndarray], int]:
    """Sum of squares of the im2col-unfolded activation patch value at
    every ``(in_channel, kh, kw)`` receptive-field offset -- Wanda's
    ``||X_j||_2`` statistic, generalized from "input feature ``j``" (a
    MatMul/Gemm column) to "receptive-field offset ``j``" (a Conv column
    of the reshaped ``[out_channels, in_channels*kH*kW]`` weight, see this
    module's own docstring) -- reduced over the batch and every output
    spatial position, for one calibration batch's raw ``[N, Cin, H, W]``
    Conv input `x`. Returns ``(sq_sum, count)`` with `sq_sum` shaped
    ``[Cin, kh, kw]`` (flattening it in that order matches
    :func:`_prune_weight`'s own ``w.reshape(n, cin * kh * kw)``), or
    ``(None, 0)`` if `x` isn't a plausible 4-D NCHW activation for this
    Conv's own kernel once padded (too small, or not rank-4 at all --
    the same "no usable calibration signal" case
    :func:`apply_wanda_pruning`'s MatMul/Gemm branch already declines
    with a plain ``x.ndim != 2`` check).

    Uses ``numpy.lib.stride_tricks.sliding_window_view`` on the
    zero-padded input rather than materializing an explicit im2col
    matrix: a view of every ``(kh, kw)`` window at every unit-stride
    position, then subsampled by the Conv's own stride -- exactly the
    positions the real Conv itself would read from, at zero extra
    calibration-data copies for the (potentially large) unstrided
    intermediate.
    """
    if x.ndim != 4:
        return None, 0
    n = x.shape[0]
    xp = np.pad(
        x,
        (
            (0, 0),
            (0, 0),
            (attrs.pad_top, attrs.pad_bottom),
            (attrs.pad_left, attrs.pad_right),
        ),
    )
    if xp.shape[2] < attrs.kh or xp.shape[3] < attrs.kw:
        return None, 0
    # [N, Cin, Hfull, Wfull, kh, kw], Hfull/Wfull at unit stride.
    windows = np.lib.stride_tricks.sliding_window_view(
        xp, (attrs.kh, attrs.kw), axis=(2, 3)
    )
    windows = windows[:, :, :: attrs.stride_h, :: attrs.stride_w, :, :]
    h_out, w_out = windows.shape[2], windows.shape[3]
    count = n * h_out * w_out
    if count == 0:
        return None, 0
    sq_sum = np.sum(np.square(windows), axis=(0, 2, 3))  # [Cin, kh, kw]
    return sq_sum, count


def _conv_group_relative_norm(
    norm_flat: np.ndarray, cout: int, cin_per_group: int, kh: int, kw: int, group: int
) -> Optional[np.ndarray]:
    """Expands a Conv layer's flat per-``(in_channel, kh, kw)``-offset
    activation norm (:func:`_conv_patch_sq_sum`'s accumulated statistic,
    ``[Cin, kh, kw]`` flattened to length ``Cin*kh*kw`` -- `Cin` here being
    the raw, *full* input channel count :func:`_conv_patch_sq_sum` unfolds
    from, i.e. ``cin_per_group * group``) into a ``[out_channels,
    in_channels/group * kh * kw]`` array matching the shape of that Conv's
    own reshaped weight (:func:`_prune_weight`'s ``w.reshape(n,
    cin*kh*kw)``) -- one row per output filter, ready to multiply
    elementwise against ``|W|``.

    This is the piece that makes Wanda's Conv support correct for a
    grouped/depthwise Conv rather than silently wrong for every group but
    the first: `norm_flat` is computed once from the *raw, full-channel*
    input (an ordinary Conv's `X`, `_conv_patch_sq_sum` never sees `group`
    at all), but output filter ``i`` of a grouped Conv only ever reads its
    *own* group's global input-channel slice, ``[g * cin_per_group, (g+1)
    * cin_per_group)`` where ``g = i // (out_channels/group)`` (ONNX's
    grouped-Conv weight layout: the first ``out_channels/group`` filters
    belong to group 0, the next block to group 1, and so on) -- so "local
    receptive-field offset ``j``" means a *different* global input channel
    depending on the filter's own group, and reusing one shared
    (group-0-shaped) norm row for every filter would silently score every
    other group's filters against the wrong channels' activation
    statistics. This function slices `norm_flat`'s ``[Cin, kh, kw]`` shape
    along its channel axis once per group and repeats that group's own
    slice across exactly the filter rows that belong to it, so each row
    the caller gets back already carries its own filter's own group's
    statistic -- the "unfold once, select each filter's own group-relative
    channel slice" approach (see this module's own docstring), avoiding a
    second, per-group im2col unfold of the same input.

    For an ordinary (``group=1``) Conv this collapses to the previous
    behavior exactly: one group spanning every channel, broadcast to every
    output filter row identically. Returns ``None`` if `norm_flat`'s length
    doesn't match ``cin_per_group * group * kh * kw`` -- the "no usable
    calibration signal" case, same as this module's other norm-shape
    checks (e.g. a probe whose captured input channel count doesn't match
    the weight this Conv node claims, an already-declined-elsewhere kind
    of malformed model this function simply doesn't guess at either).
    """
    cin_full = cin_per_group * group
    if norm_flat.shape[0] != cin_full * kh * kw:
        return None
    norm_full = norm_flat.reshape(cin_full, kh, kw)
    filters_per_group = cout // group
    rows = np.empty((cout, cin_per_group * kh * kw), dtype=norm_flat.dtype)
    for g in range(group):
        block = norm_full[g * cin_per_group : (g + 1) * cin_per_group].reshape(-1)
        rows[g * filters_per_group : (g + 1) * filters_per_group, :] = block
    return rows


def apply_magnitude_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    n: Optional[int] = None,
    m: Optional[int] = None,
) -> onnx.ModelProto:
    """Zeros the least-magnitude entries of every MatMul/vanilla-Gemm
    layer's constant 2-D float32 weight (this includes
    ``com.microsoft::GroupQueryAttention``'s separate Q/K/V projections,
    ordinary MatMul/Gemm nodes in their own right), every 2-D ``Conv``
    layer's constant 4-D float32 weight -- ordinary (``group=1``),
    depthwise, and general grouped Conv alike, see this module's own
    docstring for why grouping needs no special-casing for this technique
    -- and every ``com.microsoft::Attention`` node's constant 2-D float32
    merged QKV weight -- the data-free pruning baseline (Han et al., 2015).
    See this module's own docstring for how importance is grouped
    (including the Conv reshape convention), why the merged QKV weight
    needs no special handling here beyond matching it (:func:`_candidates`,
    via :func:`_match_attention_weight_only`), and why structured
    (shape-changing) pruning isn't offered here.

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of each row's (or, for Conv, each
            output filter's) entries to zero, ignored when ``n``/``m`` are
            given
    :param n: keep the ``n`` highest-magnitude entries per group of ``m``
            (semi-structured N:M pruning, e.g. NVIDIA's 2:4). Must be given
            together with ``m``.
    :param m: group size for N:M pruning; see ``n``
    :returns: ``model`` with every matched layer's weight zeroed in place
            to the target pattern; layers with a non-constant weight, a
            non-2-D MatMul/Gemm weight, or a non-4-D Conv weight are left
            untouched
    """
    _validate_pattern(sparsity, n, m)
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    initializer_map = {t.name: t for t in out.graph.initializer}

    for _, _, w_name, weight_transposed, is_conv in _candidates(out.graph):
        w_init = initializer_map[w_name]

        def importance_of_nk(w_nk, n=n, m=m, sparsity=sparsity):
            importance = np.abs(w_nk)
            return (
                _nm_mask(importance, n, m)
                if n is not None
                else _sparsity_mask(importance, sparsity)
            )

        _prune_weight(w_init, weight_transposed, importance_of_nk, is_conv=is_conv)

    return out


def apply_wanda_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    n: Optional[int] = None,
    m: Optional[int] = None,
    epsilon: float = 1e-8,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Wanda pruning (Sun et al., 2023): zeros the least-important entries
    of every MatMul/vanilla-Gemm layer's constant 2-D float32 weight (this
    includes ``com.microsoft::GroupQueryAttention``'s separate Q/K/V
    projections, ordinary MatMul/Gemm nodes in their own right), every 2-D
    ``Conv`` layer's constant 4-D float32 weight -- ordinary (``group=1``),
    depthwise, and general grouped Conv alike -- and every
    ``com.microsoft::Attention`` node's constant 2-D float32 merged QKV
    weight, using ``|W_ij| * ||X_j||_2`` (weight magnitude times its
    reduction-dimension entry's activation norm over calibration data) as
    the importance metric instead of plain ``|W|``. See this module's own
    docstring for the technique -- including what ``X_j`` means for a Conv
    column (one ``(in_channel, kh, kw)`` receptive-field offset, not a
    whole input channel), how a grouped/depthwise Conv's activation norm is
    kept group-relative rather than shared across every filter
    (:func:`_conv_group_relative_norm`), which Conv attribute combinations
    this confidently handles, and why ``Attention``'s merged weight -- whose
    own activation input is rank-3, not the plain 2-D tensor this metric
    requires -- always falls back to plain magnitude here rather than being
    calibrated too -- and :func:`apply_magnitude_pruning` for the
    calibration-free baseline this upgrades.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            input channel's (or, for Conv, each receptive-field offset's)
            activation norm on. Each batch is a ``{input_name: np.ndarray}``
            dict matching ``model``'s graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of each row's (or, for Conv, each
            output filter's) entries to zero, ignored when ``n``/``m`` are
            given
    :param n: keep the ``n`` highest-importance entries per group of ``m``
            (semi-structured N:M pruning, e.g. NVIDIA's 2:4). Must be given
            together with ``m``.
    :param m: group size for N:M pruning; see ``n``
    :param epsilon: floor applied to the accumulated per-channel activation
            norm, avoiding every entry of an all-zero channel tying at
            exactly-zero importance
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight zeroed in place
            to the target pattern; a MatMul/Gemm layer with a non-constant
            or non-2-D weight, a Conv layer with a non-4-D weight, or any
            matched layer whose activation input isn't usable (not a plain
            2-D tensor for MatMul/Gemm or Attention -- Attention's own
            ``X`` input is always rank-3, so it always falls into this
            case; not a 4-D NCHW tensor, or a Conv attribute combination
            :func:`_conv_spatial_attrs` declines, for Conv) falls back to
            plain magnitude pruning (no activation norm was ever observed)
    """
    _validate_pattern(sparsity, n, m)
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}

    candidates = _candidates(graph)
    if not candidates:
        return out

    probe_names = sorted({x_name for _, x_name, _, _, _ in candidates})
    probe_model = _add_probe_outputs(out, probe_names)

    # Conv attributes are per-node (two Convs can share an input tensor
    # with different kernels/strides), so the per-node Conv statistic is
    # keyed by its own weight name, not the (possibly shared) input name
    # the plain MatMul/Gemm statistic below is keyed by.
    conv_attrs: Dict[str, Optional[_ConvSpatialAttrs]] = {
        w_name: _conv_spatial_attrs(node, initializer_map[w_name])
        for node, _, w_name, _, is_conv in candidates
        if is_conv
    }

    sq_sum: Dict[str, np.ndarray] = {}
    count: Dict[str, int] = {}
    conv_sq_sum: Dict[str, np.ndarray] = {}
    conv_count: Dict[str, int] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            if x.ndim != 2:
                continue
            s = np.square(x).sum(axis=0)
            sq_sum[name] = s if name not in sq_sum else sq_sum[name] + s
            count[name] = count.get(name, 0) + x.shape[0]
        for _, x_name, w_name, _, is_conv in candidates:
            if not is_conv:
                continue
            attrs = conv_attrs[w_name]
            if attrs is None:
                continue
            x = np.asarray(result[x_name], dtype=np.float64)
            s, cnt = _conv_patch_sq_sum(x, attrs)
            if s is None:
                continue
            conv_sq_sum[w_name] = (
                s if w_name not in conv_sq_sum else conv_sq_sum[w_name] + s
            )
            conv_count[w_name] = conv_count.get(w_name, 0) + cnt

    act_norm: Dict[str, np.ndarray] = {
        name: np.sqrt(s / max(count[name], 1)) for name, s in sq_sum.items()
    }
    conv_act_norm: Dict[str, np.ndarray] = {
        name: np.sqrt(s / max(conv_count[name], 1)).reshape(-1)
        for name, s in conv_sq_sum.items()
    }

    for node, x_name, w_name, weight_transposed, is_conv in candidates:
        w_init = initializer_map[w_name]
        # `norm` is always kept 2-D here, broadcastable elementwise against
        # `w_nk` ([out_channels, K]) inside importance_of_nk below: shape
        # (1, K) for a MatMul/Gemm layer (one shared norm row for every
        # output channel, the same broadcast the plain
        # ``norm[np.newaxis, :]`` used to do directly), or -- for Conv --
        # shape (out_channels, K), already expanded per output filter's own
        # group by :func:`_conv_group_relative_norm` (trivially identical
        # across every row when ``group=1``, genuinely different per group
        # otherwise -- see that function's own docstring for why a single
        # shared row would be wrong for a grouped/depthwise Conv).
        norm: Optional[np.ndarray]
        if is_conv:
            flat_norm = conv_act_norm.get(w_name)
            norm = None
            if flat_norm is not None:
                cout, cin_per_group, kh, kw = (int(d) for d in w_init.dims)
                norm = _conv_group_relative_norm(
                    flat_norm, cout, cin_per_group, kh, kw, _conv_group(node)
                )
        else:
            norm_flat = act_norm.get(x_name)
            norm = norm_flat[np.newaxis, :] if norm_flat is not None else None

        def importance_of_nk(w_nk, norm=norm, n=n, m=m, sparsity=sparsity):
            if (
                norm is None
                or norm.shape[-1] != w_nk.shape[1]
                or (norm.shape[0] != 1 and norm.shape[0] != w_nk.shape[0])
            ):
                importance = np.abs(w_nk)  # fall back to plain magnitude
            else:
                importance = np.abs(w_nk) * np.maximum(norm, epsilon)
            return (
                _nm_mask(importance, n, m)
                if n is not None
                else _sparsity_mask(importance, sparsity)
            )

        _prune_weight(w_init, weight_transposed, importance_of_nk, is_conv=is_conv)

    return out


def weight_sparsity(model: Union[str, onnx.ModelProto]) -> float:
    """Fraction of exact-zero entries across every matched MatMul/vanilla-
    Gemm layer (including ``com.microsoft::GroupQueryAttention``'s separate
    Q/K/V projections), 2-D Conv layer -- ordinary (``group=1``), depthwise,
    and general grouped alike, since :func:`_candidates`'s default
    `allow_grouped_conv` matches all three -- or ``com.microsoft::Attention``
    merged-QKV-weight layer's constant weight -- a quick way to confirm a
    pruning call reached its target, or to measure an already-sparse model.
    Shares :func:`_candidates` with every ``apply_*_pruning`` function
    above, so it automatically reports across whatever layer types they
    match, with no separate list to keep in sync.
    Returns ``0.0`` if no matching layer is present.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    zeros = 0
    total = 0
    initializer_map = {t.name: t for t in model.graph.initializer}
    for _, _, w_name, _, _ in _candidates(model.graph):
        w = onnx.numpy_helper.to_array(initializer_map[w_name])
        zeros += int(np.count_nonzero(w == 0))
        total += w.size

    return zeros / total if total else 0.0


# --- SparseGPT ----------------------------------------------------------


def _sparsegpt_prune_columns(
    w_nk: np.ndarray,
    h: np.ndarray,
    sparsity: float,
    n: Optional[int],
    m: Optional[int],
    percdamp: float,
    proc_block_size: int,
) -> np.ndarray:
    """Returns SparseGPT-pruned values for ``w_nk`` ([N, K], output channel
    first), a direct port of the reference implementation's own
    ``fasterprune`` (https://github.com/IST-DASLab/sparsegpt/blob/master/
    sparsegpt.py). Unlike :func:`_prune_weight`'s ``importance_of_nk``
    callbacks, this returns fully-formed replacement values, not a mask --
    every *kept* entry may also change, having accumulated Hessian-based
    compensation for every *pruned* entry processed before it.
    """
    n_rows, k = w_nk.shape
    diag = np.arange(k)
    dead = h[diag, diag] == 0.0

    w_work = w_nk.copy()
    w_work[:, dead] = 0.0
    w_pruned = np.zeros_like(w_work)

    if n is None and sparsity <= 0.0:
        return w_nk.copy()  # true no-op, rather than the reference's own
        # "always drop the single lowest-scoring entry" edge case at
        # sparsity == 0.0 -- matching every other apply_*_pruning function
        # in this module, all of which treat sparsity=0.0 as a no-op.

    hinv = _inverse_hessian_cholesky(h, percdamp)

    for i1 in range(0, k, proc_block_size):
        i2 = min(i1 + proc_block_size, k)
        count = i2 - i1
        w1 = w_work[:, i1:i2].copy()
        err1 = np.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        hinv1_diag = np.diag(hinv1)

        if n is None:
            score = np.square(w1) / np.square(hinv1_diag)[np.newaxis, :]
            thresh = np.sort(score.reshape(-1))[int(score.size * sparsity)]
            mask1 = score <= thresh
        else:
            mask1 = np.zeros_like(w1, dtype=bool)

        for i in range(count):
            if n is not None and m is not None and i % m == 0:
                group_end = min(i + m, count)
                group_score = (
                    np.square(w1[:, i:group_end])
                    / np.square(hinv1_diag[i:group_end])[np.newaxis, :]
                )
                prune_count = min(group_end - i, m - n)
                mask1[:, i:group_end] = False
                if prune_count > 0:
                    drop_local = np.argsort(group_score, axis=1)[:, :prune_count]
                    np.put_along_axis(mask1[:, i:group_end], drop_local, True, axis=1)

            w_col = w1[:, i]
            d = hinv1_diag[i]
            q_col = np.where(mask1[:, i], 0.0, w_col)
            w_pruned[:, i1 + i] = q_col

            err = (w_col - q_col) / d
            err1[:, i] = err
            if i + 1 < count:
                w1[:, i + 1 :] -= np.outer(err, hinv1[i, i + 1 :])

        if i2 < k:
            w_work[:, i2:] -= err1 @ hinv[i1:i2, i2:]

    return w_pruned


def _conv_im2col_patches(
    x: np.ndarray, attrs: _ConvSpatialAttrs
) -> Optional[np.ndarray]:
    """Returns the ``[n_positions, Cin*kh*kw]`` im2col-unfolded patch
    matrix for one calibration batch's raw ``[N, Cin, H, W]`` Conv input
    `x` -- every output spatial position's full receptive-field patch,
    flattened in the same ``(in_channel, kh, kw)`` row-major order
    :func:`_prune_weight`'s own ``w.reshape(n, cin*kh*kw)`` uses (verified
    against a nested-loop oracle by
    ``test_sparsegpt_conv_hessian_matches_naive_nested_loop_oracle``).
    SparseGPT's Conv Hessian is ``H = patches.T @ patches``, this
    function's own return value being the only new piece: everything else
    (the zero-padded ``numpy.lib.stride_tricks.sliding_window_view``
    unfolding, the attribute handling) mirrors
    :func:`_conv_patch_sq_sum` exactly, reusing the same
    :class:`_ConvSpatialAttrs`/:func:`_conv_spatial_attrs` Wanda's own
    Conv support already built -- see this module's own docstring for why
    a *diagonal-only* per-offset norm (Wanda's ``_conv_patch_sq_sum``) is
    not enough here and the *full* cross-covariance this returns is
    needed instead. Returns ``None`` on the same "not usable" conditions
    :func:`_conv_patch_sq_sum` declines (not a rank-4 activation, or too
    small once padded for this kernel).
    """
    if x.ndim != 4:
        return None
    n, cin = x.shape[0], x.shape[1]
    xp = np.pad(
        x,
        (
            (0, 0),
            (0, 0),
            (attrs.pad_top, attrs.pad_bottom),
            (attrs.pad_left, attrs.pad_right),
        ),
    )
    if xp.shape[2] < attrs.kh or xp.shape[3] < attrs.kw:
        return None
    # [N, Cin, Hfull, Wfull, kh, kw], Hfull/Wfull at unit stride.
    windows = np.lib.stride_tricks.sliding_window_view(
        xp, (attrs.kh, attrs.kw), axis=(2, 3)
    )
    windows = windows[:, :, :: attrs.stride_h, :: attrs.stride_w, :, :]
    h_out, w_out = windows.shape[2], windows.shape[3]
    n_positions = n * h_out * w_out
    if n_positions == 0:
        return None
    # [N, Cin, Hout, Wout, kh, kw] -> [N, Hout, Wout, Cin, kh, kw]: moves
    # every output position to the leading axes and (in_channel, kh, kw)
    # to the trailing ones, so the final reshape's row-major flatten of
    # those trailing axes matches w.reshape(n, cin*kh*kw)'s own column
    # order exactly.
    patches = np.transpose(windows, (0, 2, 3, 1, 4, 5))
    return patches.reshape(n_positions, cin * attrs.kh * attrs.kw)


def apply_sparsegpt_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    n: Optional[int] = None,
    m: Optional[int] = None,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """SparseGPT (Frantar & Alistarh, 2023): zeros the least-important
    entries of every MatMul/vanilla-Gemm layer's constant 2-D float32
    weight (this includes ``com.microsoft::GroupQueryAttention``'s separate
    Q/K/V projections, ordinary MatMul/Gemm nodes in their own right) and
    every ``com.microsoft::Attention`` node's constant 2-D float32 merged
    QKV weight, the same unstructured-or-N:M patterns
    :func:`apply_magnitude_pruning`/:func:`apply_wanda_pruning` offer, but
    -- unlike either -- using a sequential, Hessian-error-compensating
    algorithm ported from GPTQ (:mod:`onnxsim.gptq`, same authors, same
    Cholesky-factored inverse Hessian) rather than a one-shot static
    importance score. See this module's own docstring for the technique,
    including the one deliberate departure from every other function here:
    for unstructured sparsity, the pruning threshold is shared across every
    output row within each ``proc_block_size``-wide column block (the
    reference implementation's own behavior), not chosen per row.

    Also matches, unlike :func:`apply_magnitude_pruning`/
    :func:`apply_wanda_pruning`, only *ordinary* (``group=1``) 2-D ``Conv``
    layers (``_candidates(graph, allow_grouped_conv=False)``) -- see this
    module's own docstring for the full ``[K, K]`` im2col cross-covariance
    Hessian this needed (materially more machinery than Wanda's per-offset
    norm, and -- unlike everything else this function ports from the
    reference implementation -- with no correct reference to work from at
    all: the official implementation's own ``add_batch``
    (https://github.com/IST-DASLab/sparsegpt) never actually unfolds a
    ``nn.Conv2d`` activation, only reshapes the *weight*, since its own
    driver scripts never exercise a Conv layer), how it's verified, why --
    unlike magnitude/Wanda above -- extending it to depthwise/general
    grouped Conv is deliberately left undone (a per-group Hessian and a
    per-group column-processing loop, not a mechanical extension of the
    matching alone), and how ``H`` accumulates batch by batch rather than
    ever materializing every calibration batch's unfolded patches at once.
    A Conv layer whose ``auto_pad``/``dilations`` aren't a combination
    :func:`_conv_spatial_attrs` confidently handles is left completely
    untouched, same as a layer with no observed calibration activation at
    all -- there is still no data-free fallback for SparseGPT.

    ``Attention``'s merged QKV weight has no analogous gap and is
    deliberately matched here too (unconditionally -- see
    :func:`_candidates`'s own docstring): its own input is a plain
    ``[*, K]`` activation (the same ``X`` any ordinary MatMul reads), not an
    im2col-unfolded receptive field, so ``H = X^T X`` is exactly as correct
    for it as for every other MatMul/Gemm layer already matched here, with
    no new machinery needed. See this module's own docstring for the fuller
    reasoning, including why it would be inconsistent to exclude it while
    ``GroupQueryAttention``'s separate Q/K/V MatMuls remain (and always
    were) in scope.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            layer's Hessian from. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of entries to zero (shared per column
            block, not per row -- see above), ignored when ``n``/``m`` are
            given
    :param n: keep the ``n`` highest-importance entries per group of ``m``
            (semi-structured N:M pruning, e.g. NVIDIA's 2:4, per-row exactly
            as :func:`apply_magnitude_pruning`). Must be given together
            with ``m``.
    :param m: group size for N:M pruning; see ``n``
    :param percdamp: Hessian damping factor (fraction of the mean diagonal
            added to every diagonal entry before inversion), matching
            :func:`onnxsim.apply_gptq`'s own default
    :param proc_block_size: column-processing block size -- both the
            lazy-update granularity (how many columns' errors accumulate
            locally before a full cross-block update, matching
            :func:`onnxsim.apply_gptq`'s ``proc_block_size``) and, for
            unstructured sparsity only, the width each shared per-block
            threshold is computed over
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight rewritten in
            place to the target pattern -- every surviving entry may also
            change value, having accumulated compensation for entries
            pruned before it; a MatMul/Gemm layer with no observed 2-D
            calibration activation (dead input, or every batch's
            activation isn't plain 2-D/higher-rank-with-a-trailing-
            feature-axis), or a Conv layer with no observed usable 4-D
            activation (dead input, or an ``auto_pad``/``dilations``
            combination :func:`_conv_spatial_attrs` declines), is left
            completely untouched -- unlike Wanda, there is no data-free
            fallback for a technique whose entire mechanism is the Hessian
    """
    _validate_pattern(sparsity, n, m)
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}

    # allow_grouped_conv=False: unlike apply_magnitude_pruning/
    # apply_wanda_pruning, this function's Conv Hessian machinery
    # (_conv_im2col_patches, H = patches.T @ patches) is only correct for
    # group=1 -- see this function's own docstring and this module's own
    # docstring for why extending it to grouped/depthwise Conv needs a real
    # per-group Hessian this pass does not attempt.
    candidates = _candidates(graph, allow_grouped_conv=False)
    if not candidates:
        return out

    probe_names = sorted({x_name for _, x_name, _, _, _ in candidates})
    probe_model = _add_probe_outputs(out, probe_names)

    # Conv attributes are per-node (two Convs can share an input tensor
    # with different kernels/strides), so the Conv Hessian below is keyed
    # by its own weight name, mirroring apply_wanda_pruning's own
    # conv_attrs/conv_act_norm.
    conv_attrs: Dict[str, Optional[_ConvSpatialAttrs]] = {
        w_name: _conv_spatial_attrs(node, initializer_map[w_name])
        for node, _, w_name, _, is_conv in candidates
        if is_conv
    }

    activations: Dict[str, List[np.ndarray]] = {
        x_name: [] for _, x_name, _, _, is_conv in candidates if not is_conv
    }
    # Unlike the MatMul/Gemm activations above (each layer's whole
    # calibration set concatenated once, below -- small enough per layer
    # to keep entirely in memory), each Conv layer's H accumulates
    # incrementally, one calibration batch's own im2col-unfolded patch
    # matrix at a time: a full [n_positions, K] patch matrix can be large,
    # so no batch's patches outlive the H += they fold into. See this
    # module's own docstring.
    conv_h: Dict[str, np.ndarray] = {}

    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in activations:
            x = np.asarray(result[name], dtype=np.float64)
            if x.ndim < 2:
                continue
            activations[name].append(x.reshape(-1, x.shape[-1]))
        for _, x_name, w_name, _, is_conv in candidates:
            if not is_conv:
                continue
            attrs = conv_attrs[w_name]
            if attrs is None:
                continue
            x_conv = np.asarray(result[x_name], dtype=np.float64)
            patches = _conv_im2col_patches(x_conv, attrs)
            if patches is None:
                continue
            h_batch = patches.T @ patches
            conv_h[w_name] = (
                h_batch if w_name not in conv_h else conv_h[w_name] + h_batch
            )

    for _, x_name, w_name, weight_transposed, is_conv in candidates:
        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)

        if is_conv:
            cout, cin, kh, kw = w.shape
            h = conv_h.get(w_name)
            if h is None:
                continue
            w_nk = w.reshape(cout, cin * kh * kw)
            w_pruned_nk = _sparsegpt_prune_columns(
                w_nk, h, sparsity, n, m, percdamp, proc_block_size
            )
            w_new = w_pruned_nk.reshape(cout, cin, kh, kw).astype(np.float32)
        else:
            acts = activations[x_name]
            if not acts:
                continue
            x = np.concatenate(acts, axis=0)
            dim0, dim1 = w.shape
            w_nk = w if weight_transposed else w.T  # [N, K]
            if x.shape[1] != w_nk.shape[1]:
                continue

            h = x.T @ x
            w_pruned_nk = _sparsegpt_prune_columns(
                w_nk, h, sparsity, n, m, percdamp, proc_block_size
            )
            w_new = w_pruned_nk if weight_transposed else w_pruned_nk.T
            w_new = w_new.reshape(dim0, dim1).astype(np.float32)

        w_init.CopyFrom(onnx.numpy_helper.from_array(w_new, name=w_init.name))

    return out


# --- Structured (channel) pruning -------------------------------------------

# Shape-preserving, channel-order-preserving elementwise ops that may sit
# between a producer and consumer without blocking the chain: unary
# activations (single input, single output, no other operand to worry
# about) and Add/Mul against a constant per-channel bias/scale.
_UNARY_PASS_THROUGH = {
    "Relu",
    "LeakyRelu",
    "Elu",
    "Selu",
    "Sigmoid",
    "Tanh",
    "Softplus",
    "Softsign",
    "Gelu",
    "HardSigmoid",
    "Mish",
    "Identity",
    "Cast",
    # com.microsoft::QuickGelu(X) = X * Sigmoid(alpha * X) (alpha an
    # attribute, default 1.702, not a second *input* -- confirmed against
    # onnxruntime's own schema, contrib_defs.cc, and by direct execution,
    # see this module's own docstring): a single-input, single-output,
    # purely elementwise activation exactly like every other entry in this
    # set, just from a different domain -- membership here is by op_type
    # alone for every entry, never by domain (a same-named op in an
    # unrelated custom domain has always been a theoretical risk this set
    # accepts, not one unique to this entry). Being unary, it needs no
    # dedicated hop machinery at all: adding it here alone already extends
    # every walker that already consults `_UNARY_PASS_THROUGH` --
    # `_walk_to_consumer`/`_walk_to_conv_consumer` (forward), their two
    # backward counterparts, *and* `_trace_gate_producer_backward`'s own
    # gated-pair gate-activation matcher -- for free.
    "QuickGelu",
}
_BINARY_CHANNEL_OPS = {"Add", "Mul"}
_MAX_CHAIN_HOPS = 8

# com.microsoft's fused bias-add + Gelu-family activation nodes -- the FFN
# analogue of the SkipLayerNormalization residual fusion above, done by the
# same onnxruntime transformer-optimizer tool: `BiasGelu(A, B) = Gelu(A + B)`
# (erf-based, exactly plain ONNX Gelu's own default `approximate="none"`)
# and `FastGelu(X[, bias]) = Gelu_tanh(X [+ bias])` (the tanh approximation,
# `bias` optional) both fuse an FFN's bias-add into its following activation
# the same way `BiasGelu`'s own name suggests -- confirmed against
# onnxruntime's own schema (`contrib_defs.cc`) and CPU kernel
# (`bias_gelu.cc`'s shared `BiasGelu<T, use_approximation>::AddBiasGelu`,
# which literally computes `value = input[i] + bias[i]` then erf- or
# tanh-based Gelu on `value`) and by direct execution. Neither fusion
# changes *which* Gelu variant is being computed -- FastGelu's tanh
# approximation is a different formula from plain `Gelu`'s erf-based one
# regardless of fusion, already true before this pass ever mattered to
# either -- so, exactly like a bias/scale `Add`/`Mul` hop's own constant,
# what matters here is only that each is a per-channel-independent,
# shape-preserving elementwise op with one extra constant operand to slice,
# not its particular activation math. `BiasGelu`'s own schema makes `B`
# (bias) a *required* second input; `FastGelu`'s marks its own `bias`
# *optional* -- both handled by :func:`_match_fused_bias_gelu` below. Like
# the `_BINARY_CHANNEL_OPS` bias/scale hop these sit alongside, this is a
# MatMul/Gemm-chain-only hop (:func:`_walk_to_consumer`/
# :func:`_walk_matmul_producer_backward`): Conv chains already decline any
# per-channel `Add`/`Mul` hop at all (a real Conv's bias already lives in
# its own third input, see this module's own docstring), and neither
# optimizer fusion targets Conv graphs in practice, so no Conv-side
# analogue is added.
_FUSED_BIAS_GELU_OPS: Dict[str, bool] = {
    "BiasGelu": True,  # bias (input 1) required by BiasGelu's own schema
    "FastGelu": False,  # bias (input 1) optional for FastGelu
}
_FUSED_BIAS_GELU_DOMAIN = "com.microsoft"

_ConsumerMatch = Tuple[onnx.NodeProto, str, bool]  # (node, weight, weight_transposed)


@dataclass(frozen=True)
class _Producer:
    node: onnx.NodeProto
    weight: str
    weight_transposed: bool
    bias: Optional[str]
    # Activation nodes between this producer's raw output and the point it
    # combines with another producer (a gated pair only -- see
    # :func:`_find_gated_chains`; empty for a plain single-producer chain).
    pre_ops: Tuple[onnx.NodeProto, ...] = ()
    # True for a Conv producer: `weight_transposed` is meaningless then
    # (Conv's ``[out_channels, in_channels, kH, kW]`` weight layout is
    # fixed), and output channels always live on axis 0.
    is_conv: bool = False
    # This Conv's own ``group`` attribute (always 1 for a MatMul/Gemm
    # producer, or an ordinary ``group=1`` Conv). > 1 for a general grouped
    # Conv producer (see :func:`_match_conv_producer`) -- output channel
    # axis 0 stays flat/global either way (grouping only splits the *input*
    # axis), so this only changes how :func:`_apply_chains` picks `keep`,
    # never how the producer's own weight is sliced.
    group: int = 1


@dataclass(frozen=True)
class _ConvPassThrough:
    """A depthwise Conv (``group == in_channels == out_channels``) the chain
    walk crossed transparently between a Conv chain's real producer and real
    consumer. A depthwise Conv mixes no channels at all -- output channel
    ``i`` depends only on input channel ``i`` -- so it needs no independent
    importance of its own the way a producer/consumer boundary does; it is
    carried on the matched :class:`_Chain` purely so :func:`_apply_chains`
    can slice its own ``[C, 1, kH, kW]`` weight (and bias, if present) by
    the *same* `keep` index set as the chain's real producer, and update its
    ``group`` attribute to the new channel count. See
    :func:`_walk_to_conv_consumer`.
    """

    node: onnx.NodeProto
    weight: str
    bias: Optional[str]


@dataclass(frozen=True)
class _Chain:
    # One producer for a plain chain; two for a gated (elementwise-product)
    # pair, where both branches must agree on which channels survive.
    producers: Tuple[_Producer, ...]
    chain_ops: Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]
    consumer_node: onnx.NodeProto
    consumer_weight: str
    consumer_weight_transposed: bool
    n_channels: int
    # True for a Conv consumer: input channels always live on axis 1 of its
    # ``[out_channels, in_channels, kH, kW]`` weight, regardless of
    # `consumer_weight_transposed` (unused then).
    consumer_is_conv: bool = False
    # Depthwise Conv hops the chain walk crossed transparently between the
    # real producer and the real consumer (Conv chains only -- see
    # :class:`_ConvPassThrough`; always empty for a MatMul/Gemm chain).
    conv_pass_through: Tuple[_ConvPassThrough, ...] = ()
    # The consumer's own ``group`` attribute (always 1 for a MatMul/Gemm
    # consumer, or an ordinary ``group=1`` Conv). > 1 for a general grouped
    # Conv consumer (see :func:`_match_conv_consumer`) -- unlike the
    # producer side, the consumer's input-channel axis *is*
    # per-group-relative, so this drives both `keep` selection (see
    # :func:`_chain_group`) and the dedicated slicing
    # :func:`_slice_grouped_consumer_conv_weight` performs.
    consumer_group: int = 1


def _consumers_of(graph: onnx.GraphProto) -> Dict[str, List[onnx.NodeProto]]:
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    return consumers


def _match_producer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, bool, Optional[str], int]]:
    """If `node` is a MatMul/vanilla-Gemm with a constant 2-D float32
    weight (and, for Gemm, either no bias or a constant one), returns
    ``(weight_name, weight_transposed, bias_name_or_None, n_channels)``.
    """
    match = _match_matmul_like(node)
    if match is None:
        return None
    _, w_name, weight_transposed = match
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or w_init.data_type != onnx.TensorProto.FLOAT
        or len(w_init.dims) != 2
    ):
        return None
    bias_name = None
    if node.op_type == "Gemm" and len(node.input) == 3:
        bias_name = node.input[2]
        if bias_name not in initializer_map:
            return None  # non-constant bias -- can't safely prune it
    n_channels = w_init.dims[0] if weight_transposed else w_init.dims[1]
    return w_name, weight_transposed, bias_name, n_channels


def _match_fused_bias_gelu(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, Optional[str]]]:
    """If `node` is a ``com.microsoft::BiasGelu``/``FastGelu`` node (see
    `_FUSED_BIAS_GELU_OPS`'s own comment above for the exact fused
    arithmetic and how it was confirmed), returns ``(data_name,
    bias_name_or_None)``: `data_name` is the node's own primary input (`A`/
    `X`, input 0), and `bias_name` is its per-channel bias operand (input 1)
    when present and a constant float initializer shaped like a flat
    per-last-axis vector -- the same self-consistency bar
    :func:`_walk_matmul_producer_backward`'s own `_BINARY_CHANNEL_OPS` hop
    check already uses (the real ``dims[-1] == n_channels`` check is
    deferred to the caller: :func:`_walk_to_consumer` already knows
    `n_channels` and checks immediately; :func:`_walk_matmul_producer_backward`
    doesn't yet, and defers to :func:`_find_matmul_residual_chains` once the
    group's real channel count is known, exactly like that hop's own and
    `_skip_layer_norm_const_names`'s own deferred check).

    Declines (``None``) when `node` isn't one of these ops/domain at all, or
    when its bias is required but missing (`BiasGelu`'s own schema makes its
    `B` input required, unlike `FastGelu`'s optional `bias`) or present but
    non-constant -- never guessed at, the same conservative bar a
    non-constant bias/scale on an ordinary `Add`/`Mul` hop already gets.
    `FastGelu` with its bias genuinely absent (omitted entirely, or present
    as an empty placeholder) returns ``(data_name, None)`` -- no term to
    slice, the same shape a `SkipLayerNormalization` node's own absent
    `beta`/`bias` already gets.
    """
    bias_required = _FUSED_BIAS_GELU_OPS.get(node.op_type)
    if (
        bias_required is None
        or node.domain != _FUSED_BIAS_GELU_DOMAIN
        or not node.input
        or not node.input[0]
        or len(node.output) != 1
    ):
        return None
    data_name = node.input[0]
    has_bias_input = len(node.input) > 1 and bool(node.input[1])
    if not has_bias_input:
        if bias_required:
            return None  # BiasGelu's own schema requires a bias operand
        return data_name, None  # FastGelu with no bias -- plain tanh-Gelu(x)
    bias_name = node.input[1]
    bias_init = initializer_map.get(bias_name)
    if (
        bias_init is None
        or bias_init.data_type != onnx.TensorProto.FLOAT
        or not list(bias_init.dims)
        or int(np.prod(bias_init.dims)) != bias_init.dims[-1]
    ):
        return None  # non-constant bias -- can't safely slice/prune it
    return data_name, bias_name


def _walk_to_consumer(
    start: str,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    n_channels: int,
    max_hops: int,
) -> Tuple[Optional[_ConsumerMatch], Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]]:
    """From tensor `start`, walks forward through shape-preserving
    elementwise ops (an activation, an Add/Mul against a constant
    per-channel bias/scale, or a fused ``com.microsoft::BiasGelu``/
    ``FastGelu`` node -- see `_FUSED_BIAS_GELU_OPS`'s own comment and
    :func:`_match_fused_bias_gelu`) with no other consumer anywhere along
    the way, until a MatMul/vanilla-Gemm consumer is found whose reduction
    dimension matches `n_channels`. Returns ``(None, ())`` if the walk
    runs out of hops, hits a branch, or never reaches such a consumer.
    """
    chain_ops: List[Tuple[onnx.NodeProto, Optional[str]]] = []
    consumer = None
    cur = start
    for _hop in range(max_hops):
        candidates = consumers_of.get(cur, [])
        if len(candidates) != 1:
            break
        nxt = candidates[0]

        cm = _match_matmul_like(nxt)
        if cm is not None and cm[0] == cur:
            _, cw_name, c_weight_transposed = cm
            cw_init = initializer_map.get(cw_name)
            if (
                cw_init is not None
                and cw_init.data_type == onnx.TensorProto.FLOAT
                and len(cw_init.dims) == 2
            ):
                k = cw_init.dims[1] if c_weight_transposed else cw_init.dims[0]
                if k == n_channels:
                    consumer = (nxt, cw_name, c_weight_transposed)
            break

        const_name: Optional[str] = None
        if (
            nxt.op_type in _UNARY_PASS_THROUGH
            and list(nxt.input) == [cur]
            and len(nxt.output) == 1
        ):
            pass
        elif (
            nxt.op_type in _BINARY_CHANNEL_OPS
            and len(nxt.input) == 2
            and cur in nxt.input
            and len(nxt.output) == 1
        ):
            other = nxt.input[1] if nxt.input[0] == cur else nxt.input[0]
            const_init = initializer_map.get(other)
            if (
                const_init is not None
                and const_init.data_type == onnx.TensorProto.FLOAT
                and list(const_init.dims)
                and const_init.dims[-1] == n_channels
                and int(np.prod(const_init.dims)) == n_channels
            ):
                const_name = other
            else:
                break
        elif nxt.op_type in _FUSED_BIAS_GELU_OPS:
            fused = _match_fused_bias_gelu(nxt, initializer_map)
            if fused is None or fused[0] != cur:
                break
            _, bias_name = fused
            if bias_name is not None and (
                initializer_map[bias_name].dims[-1] != n_channels
            ):
                break
            const_name = bias_name
        else:
            break

        out2 = nxt.output[0]
        if len(consumers_of.get(out2, [])) != 1 or out2 in graph_outputs:
            break
        chain_ops.append((nxt, const_name))
        cur = out2

    return consumer, tuple(chain_ops)


def _find_chains(graph: onnx.GraphProto) -> List[_Chain]:
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    graph_outputs = {o.name for o in graph.output}

    def _is_internal(name: str) -> bool:
        # Safe to reshape only if exactly one node reads it and it isn't
        # itself something the caller observes (a graph output).
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    chains = []
    for node in graph.node:
        info = _match_producer(node, initializer_map)
        if info is None:
            continue
        w_name, weight_transposed, bias_name, n_channels = info

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue

        consumer, chain_ops = _walk_to_consumer(
            out_name,
            initializer_map,
            consumers_of,
            graph_outputs,
            n_channels,
            _MAX_CHAIN_HOPS,
        )
        if consumer is None:
            continue

        chains.append(
            _Chain(
                producers=(_Producer(node, w_name, weight_transposed, bias_name),),
                chain_ops=chain_ops,
                consumer_node=consumer[0],
                consumer_weight=consumer[1],
                consumer_weight_transposed=consumer[2],
                n_channels=n_channels,
            )
        )
    return chains


def _conv_group(node: onnx.NodeProto) -> int:
    for attr in node.attribute:
        if attr.name == "group":
            return attr.i
    return 1  # ONNX default


def _match_conv_producer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, Optional[str], int, int]]:
    """If `node` is an ordinary (``group=1``) *or* a general grouped
    (``1 < group < in_channels``, ``group != out_channels`` -- see this
    module's own docstring) 2-D ``Conv`` with a constant 4-D float32
    ``[out_channels, in_channels/group, kH, kW]`` weight (and, if present, a
    constant bias), returns
    ``(weight_name, bias_name_or_None, out_channels, group)``. A depthwise
    Conv (``group == in_channels == out_channels``) never matches: even
    though it *is* given a narrower exception elsewhere in this pass, as a
    transparent pass-through hop the chain walk may cross between two real
    producer/consumer boundaries (see
    :func:`_match_depthwise_conv_pass_through`,
    :func:`_walk_to_conv_consumer`), it is never itself matched as a
    producer -- only a *general* grouped Conv (this function's new case) is.
    A general grouped Conv's `group` output channels never need slicing
    themselves here: axis 0 (`out_channels`) is flat/global regardless of
    grouping (grouping only ever splits the *input* axis), so the caller's
    existing `keep`-index slicing of a producer's own weight/bias needs no
    special-casing for this -- only *which* `keep` indices get chosen (one
    independent top-k per group, see :func:`_apply_chains`) changes.
    """
    if node.op_type != "Conv" or len(node.input) < 2:
        return None
    w_name = node.input[1]
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or w_init.data_type != onnx.TensorProto.FLOAT
        or len(w_init.dims) != 4
    ):
        return None
    group = _conv_group(node)
    if group < 1:
        return None
    out_channels = w_init.dims[0]
    in_channels = w_init.dims[1] * group
    if group > 1 and (
        group >= in_channels  # depthwise (a transparent hop, not a
        # producer) or an unsupported in-channels-per-group == 1 grouping
        or group == out_channels
        or out_channels % group != 0  # groups must stay equal-sized
    ):
        return None
    bias_name = None
    if len(node.input) == 3 and node.input[2]:
        bias_name = node.input[2]
        if bias_name not in initializer_map:
            return None  # non-constant bias -- can't safely prune it
    return w_name, bias_name, out_channels, group


def _match_conv_consumer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, int, int]]:
    """If `node` is an ordinary (``group=1``) *or* a general grouped Conv
    (see :func:`_match_conv_producer`) with a constant 4-D float32 weight,
    returns ``(weight_name, in_channels, group)``. Like
    :func:`_match_conv_producer`, a depthwise Conv never matches here
    either -- it's only ever a transparent pass-through hop the chain walk
    crosses en route to a *real* consumer, never a consumer itself (see
    :func:`_match_depthwise_conv_pass_through`). Unlike the producer side, a
    grouped consumer's input-channel axis (axis 1 of its weight) *is*
    per-group-relative -- weight column ``j`` on an output filter belonging
    to group ``g`` means global input channel ``g * (in_channels / group) +
    j``, not global channel ``j`` -- so slicing it by the chain's `keep`
    indices needs the dedicated
    :func:`_slice_grouped_consumer_conv_weight`, not the flat
    ``w[:, keep, ...]`` an ordinary consumer's weight uses.
    """
    if node.op_type != "Conv" or len(node.input) < 2:
        return None
    w_name = node.input[1]
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or w_init.data_type != onnx.TensorProto.FLOAT
        or len(w_init.dims) != 4
    ):
        return None
    group = _conv_group(node)
    if group < 1:
        return None
    out_channels = w_init.dims[0]
    in_channels = w_init.dims[1] * group
    if group > 1 and (
        group >= in_channels or group == out_channels or out_channels % group != 0
    ):
        return None
    return w_name, in_channels, group


def _match_depthwise_conv_pass_through(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    n_channels: int,
) -> Optional[Tuple[str, Optional[str]]]:
    """If `node` is a depthwise 2-D ``Conv`` (``group == in_channels ==
    out_channels == n_channels``) with a constant ``[n_channels, 1, kH,
    kW]`` float32 weight (and, if present, a constant bias), returns
    ``(weight_name, bias_name_or_None)``. A depthwise Conv mixes no channels
    at all -- output channel ``i`` depends only on input channel ``i`` --
    unlike a general grouped Conv (``group`` neither 1 nor `n_channels`),
    which is not matched here and stays out of scope for this pass entirely
    (see :func:`_match_conv_producer`/:func:`_match_conv_consumer`'s own
    docstrings): only in the depthwise case is every output channel tied
    1:1 to the same-index input channel, which is what lets the chain walk
    (:func:`_walk_to_conv_consumer`) treat it as a transparent pass-through
    hop -- carrying whatever channel-index set survives upstream straight
    through, unchanged -- rather than a producer or consumer of its own.
    """
    if node.op_type != "Conv" or len(node.input) < 2:
        return None
    w_name = node.input[1]
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or w_init.data_type != onnx.TensorProto.FLOAT
        or len(w_init.dims) != 4
        or w_init.dims[0] != n_channels
        or w_init.dims[1] != 1
        or _conv_group(node) != n_channels
    ):
        return None
    bias_name = None
    if len(node.input) == 3 and node.input[2]:
        bias_name = node.input[2]
        b_init = initializer_map.get(bias_name)
        if b_init is None or b_init.data_type != onnx.TensorProto.FLOAT:
            return None  # non-constant bias -- can't safely prune it
    return w_name, bias_name


def _walk_to_conv_consumer(
    start: str,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    n_channels: int,
    max_hops: int,
) -> Tuple[
    Optional[Tuple[onnx.NodeProto, str, int]],
    Tuple[Tuple[onnx.NodeProto, None], ...],
    Tuple[_ConvPassThrough, ...],
]:
    """The Conv analogue of :func:`_walk_to_consumer`: from tensor `start`,
    walks forward through unary shape-preserving activations (see
    `_UNARY_PASS_THROUGH`) and depthwise Conv hops (see
    :func:`_match_depthwise_conv_pass_through` -- transparent to the
    channel-index mapping, but each still needs its own weight/bias sliced
    and its ``group`` attribute updated, so they're returned separately as
    `conv_pass_through` rather than folded into `chain_ops`) with no other
    consumer anywhere along the way, until an ordinary (``group=1``) *or*
    general grouped Conv consumer is found whose input channel count
    matches `n_channels` (see :func:`_match_conv_consumer`). A depthwise
    Conv is only ever a transparent hop, never a match for the consumer
    role itself -- one sitting last before a graph output or a branch
    simply ends the walk with no consumer found, same as any other
    unmatched topology. Unlike the MatMul/Gemm walk, no per-channel
    ``Add``/``Mul`` op is recognized -- see this module's own docstring for
    why that's out of scope for Conv chains.
    """
    chain_ops: List[Tuple[onnx.NodeProto, None]] = []
    conv_pass_through: List[_ConvPassThrough] = []
    consumer: Optional[Tuple[onnx.NodeProto, str, int]] = None
    cur = start
    for _hop in range(max_hops):
        candidates = consumers_of.get(cur, [])
        if len(candidates) != 1:
            break
        nxt = candidates[0]

        if nxt.op_type == "Conv" and nxt.input[0] == cur:
            depthwise = _match_depthwise_conv_pass_through(
                nxt, initializer_map, n_channels
            )
            if depthwise is not None:
                out2 = nxt.output[0]
                if len(consumers_of.get(out2, [])) != 1 or out2 in graph_outputs:
                    break
                dw_weight, dw_bias = depthwise
                conv_pass_through.append(_ConvPassThrough(nxt, dw_weight, dw_bias))
                cur = out2
                continue

            match = _match_conv_consumer(nxt, initializer_map)
            if match is not None and match[1] == n_channels:
                consumer = (nxt, match[0], match[2])
            break

        if not (
            nxt.op_type in _UNARY_PASS_THROUGH
            and list(nxt.input) == [cur]
            and len(nxt.output) == 1
        ):
            break

        out2 = nxt.output[0]
        if len(consumers_of.get(out2, [])) != 1 or out2 in graph_outputs:
            break
        chain_ops.append((nxt, None))
        cur = out2

    return consumer, tuple(chain_ops), tuple(conv_pass_through)


def _find_conv_chains(graph: onnx.GraphProto) -> List[_Chain]:
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    graph_outputs = {o.name for o in graph.output}

    def _is_internal(name: str) -> bool:
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    chains = []
    for node in graph.node:
        info = _match_conv_producer(node, initializer_map)
        if info is None:
            continue
        w_name, bias_name, n_channels, producer_group = info

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue

        consumer, chain_ops, conv_pass_through = _walk_to_conv_consumer(
            out_name,
            initializer_map,
            consumers_of,
            graph_outputs,
            n_channels,
            _MAX_CHAIN_HOPS,
        )
        if consumer is None:
            continue
        consumer_node, consumer_weight, consumer_group = consumer

        if (
            producer_group > 1
            and consumer_group > 1
            and producer_group != consumer_group
        ):
            # Both sides grouped, but with a different group count: the two
            # sides' block boundaries wouldn't generally align (a channel
            # surviving as "the k-th of the producer's own group" has no
            # well-defined membership in any of the consumer's
            # differently-sized groups), so this composition needs real
            # cross-chain bookkeeping this pass doesn't attempt -- declined
            # outright, same as any other topology left unmatched rather
            # than guessed at. See this module's own docstring.
            continue

        chains.append(
            _Chain(
                producers=(
                    _Producer(
                        node,
                        w_name,
                        False,
                        bias_name,
                        is_conv=True,
                        group=producer_group,
                    ),
                ),
                chain_ops=chain_ops,
                consumer_node=consumer_node,
                consumer_weight=consumer_weight,
                consumer_weight_transposed=False,
                n_channels=n_channels,
                consumer_is_conv=True,
                conv_pass_through=conv_pass_through,
                consumer_group=consumer_group,
            )
        )
    return chains


# --- Conv residual (Add-merged) chains -------------------------------------
#
# A bounded slice of the general dependency-graph-grouping problem this
# module's own docstring otherwise disclaims (see its "residual/skip
# connection" paragraph): a channel-preserving `Add(a, b)` where *both*
# operands are non-constant tensors -- `y = Add(x, f(x))`, the shape every
# residual/skip connection takes -- forces whichever real Conv producer(s)
# feed `a` and `b` to be pruned to the exact same channel-index set, since
# they're about to be summed elementwise. `_walk_to_conv_consumer`'s own
# forward walk just breaks at such an `Add` (an ordinary Conv chain has no
# way to represent "two producers must agree"), so this is a *separate*
# finder -- `_find_conv_residual_chains` below -- built entirely on top of
# the existing `_Chain`/`_apply_chains` machinery rather than a change to
# `_walk_to_conv_consumer`/`_find_conv_chains` themselves: every `_Chain` it
# produces still has exactly one (real, `group=1`) consumer and some tuple
# of producers, precisely the shape `_apply_chains` (and both importance
# callbacks, `_plain_structured_importance`'s already-generic root-sum-
# square combination included) already knows how to ride a shared `keep`
# index set through -- only *finding* that tuple of producers needs new
# code.
#
# The union-find grouping :func:`_walk_conv_producer_backward` and
# :func:`_find_conv_residual_chains` build together covers not just a
# single `Add(x, f(x))` but a whole *chain* of such merges transitively
# sharing one spine channel count -- "many residual blocks share one
# spine" -- by walking backward from each `Add` operand and, on hitting
# *another* eligible `Add`'s raw output, unioning that `Add`'s own group in
# rather than stopping. What it does **not** do, on purpose, is relax the
# single-consumer bar every other hop in this module already holds every
# intermediate tensor to: a real multi-block ResNet stage's post-block
# tensor is read *twice* (once by the next block's own first Conv, once
# as-is by that block's own `Add`), and that fan-out is exactly the
# "more than one ... consumer anywhere along it" case this pass declines
# rather than guesses at -- so an *interior* block of a real deep ResNet
# stage is left completely untouched by this mechanism (the same
# conservative "no branch-following" boundary as everywhere else in this
# module), while a lone residual connection (a single `Add`, each branch
# its own uninterrupted Conv/depthwise-pass-through chain back to a real
# producer -- e.g. a projection-shortcut block, or the very last block of a
# stage before something else consumes its output) or a genuinely linear
# stack of `Add`-only combinations (no intervening branch at all) is
# pruned, with the same oracle-verified numeric guarantee as every other
# chain kind here. See this module's own docstring for how this changes
# (without fully retiring) the "left untouched" residual paragraph above.


def _is_eligible_add_merge(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> bool:
    """True for an ``Add`` node :func:`_find_conv_residual_chains`/
    :func:`_find_matmul_residual_chains` may treat as a residual merge
    point: exactly two distinct, non-constant operands.
    A per-channel bias/scale ``Add`` (one operand a constant initializer --
    already out of scope for Conv chains generally, see this module's own
    docstring) or a degenerate ``Add(x, x)`` never qualifies -- neither is a
    "two independent producers must agree" merge point at all.
    """
    return (
        node.op_type == "Add"
        and len(node.input) == 2
        and len(node.output) == 1
        and node.input[0] != node.input[1]
        and node.input[0] not in initializer_map
        and node.input[1] not in initializer_map
    )


def _match_conv_pass_through_self(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, Optional[str]]]:
    """The depthwise-Conv pass-through check :func:`_walk_conv_producer_backward`
    uses: unlike :func:`_match_depthwise_conv_pass_through`, which validates
    a hop against an externally supplied `n_channels`, the backward residual
    walk doesn't know its group's shared channel count yet at the point it
    first crosses a hop (it's still walking toward whichever real producer
    -- or other ``Add`` -- eventually establishes it), so this checks the
    node's own weight is self-consistently depthwise-shaped (``dims[0] ==
    group``, ``dims[1] == 1``) by calling that same matcher with the node's
    own ``dims[0]`` as the "expected" count -- trivially satisfying that one
    check and leaving every other one intact. :func:`_find_conv_residual_chains`
    re-validates every such hop against the group's real, established
    channel count once the whole group is resolved.
    """
    if node.op_type != "Conv" or len(node.input) < 2:
        return None
    w_init = initializer_map.get(node.input[1])
    if (
        w_init is None
        or w_init.data_type != onnx.TensorProto.FLOAT
        or len(w_init.dims) != 4
    ):
        return None
    return _match_depthwise_conv_pass_through(node, initializer_map, w_init.dims[0])


def _walk_conv_producer_backward(
    start: str,
    node_by_output: Dict[str, onnx.NodeProto],
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    max_hops: int,
) -> Tuple[
    str,
    Optional[Union[Tuple[_Producer, int], onnx.NodeProto]],
    Tuple[_ConvPassThrough, ...],
    Tuple[onnx.NodeProto, ...],
]:
    """The backward counterpart of :func:`_walk_to_conv_consumer`, used only
    by :func:`_find_conv_residual_chains` to resolve one operand of an
    ``Add`` merge point back to whatever produces it. Walks backward from
    tensor `start` through unary pass-through activations and
    self-consistently-depthwise Conv hops (see
    :func:`_match_conv_pass_through_self`), requiring every tensor crossed
    -- `start` itself included -- to have exactly one consumer, the same
    safety bar every other hop in this module already holds every
    intermediate tensor to. This is exactly what makes a real multi-block
    ResNet's shared "post-block" tensor -- read both by the next block's
    own first Conv and, unchanged, by that block's own ``Add`` -- resolve to
    ``"fail"`` rather than being silently pruned wrong: whichever of its two
    readers this walk is approached from finds more than one consumer on
    the very first hop and stops.

    Returns one of:

    - ``("producer", (producer, n_channels), pass_through, unary_ops)`` --
      resolved all the way back to a real ``group=1`` Conv producer;
    - ``("add", add_node, pass_through, unary_ops)`` -- resolved to another
      eligible ``Add`` merge node's raw output instead (the "many residual
      blocks share one spine" case: the caller unions this group with that
      ``Add``'s own rather than treating it as a separate producer);
    - ``("fail", None, (), ())`` -- a graph input, a non-Conv/non-``Add``
      producer, a branch, or the hop limit -- the caller declines the whole
      group this operand belongs to, rather than guessing.
    """
    pass_through: List[_ConvPassThrough] = []
    unary_ops: List[onnx.NodeProto] = []
    cur = start
    for _hop in range(max_hops):
        if len(consumers_of.get(cur, [])) != 1 or cur in graph_outputs:
            return "fail", None, (), ()
        node = node_by_output.get(cur)
        if node is None or len(node.output) != 1 or node.output[0] != cur:
            return "fail", None, (), ()

        prod_info = _match_conv_producer(node, initializer_map)
        if prod_info is not None:
            w_name, bias_name, n_channels, producer_group = prod_info
            if producer_group != 1:
                # General grouped Conv is deliberately out of scope for the
                # residual/Add-merge case -- see this function's own
                # docstring ("a real group=1 Conv producer"). Combining
                # per-group independent top-k with cross-branch
                # combined-importance ranking isn't verified here.
                return "fail", None, (), ()
            producer = _Producer(node, w_name, False, bias_name, is_conv=True)
            return (
                "producer",
                (producer, n_channels),
                tuple(reversed(pass_through)),
                tuple(reversed(unary_ops)),
            )

        dw = _match_conv_pass_through_self(node, initializer_map)
        if dw is not None:
            dw_weight, dw_bias = dw
            pass_through.append(_ConvPassThrough(node, dw_weight, dw_bias))
            cur = node.input[0]
            continue

        if node.op_type in _UNARY_PASS_THROUGH and len(node.input) == 1:
            unary_ops.append(node)
            cur = node.input[0]
            continue

        if _is_eligible_add_merge(node, initializer_map):
            return (
                "add",
                node,
                tuple(reversed(pass_through)),
                tuple(reversed(unary_ops)),
            )

        return "fail", None, (), ()

    return "fail", None, (), ()


def _find_conv_residual_chains(graph: onnx.GraphProto) -> List[_Chain]:
    """Finds Conv residual/skip-connection groups -- see the section comment
    above. For every maximal union-find group of transitively-connected
    eligible ``Add`` merge points (:func:`_is_eligible_add_merge`), resolves
    every member's two operands via :func:`_walk_conv_producer_backward`:
    each must reach either a real ``group=1`` Conv producer (a "leaf" of the
    group) or another `Add` already in the same group. If *any* operand,
    anywhere in the group, fails to resolve that way, or the leaf
    producers' channel counts don't all agree, the *entire* group is
    declined -- never partially pruned. A correctly-resolved group must
    reduce to exactly one "sink" `Add` (the one whose own output isn't
    itself consumed by another `Add` in the group); the chain walk
    continues forward from there via the ordinary :func:`_walk_to_conv_consumer`
    to find the group's one real downstream consumer, exactly as any other
    Conv chain does.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    node_by_output = {out: node for node in graph.node for out in node.output}
    graph_outputs = {o.name for o in graph.output}

    def _is_internal(name: str) -> bool:
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    eligible_adds = [
        node for node in graph.node if _is_eligible_add_merge(node, initializer_map)
    ]
    if not eligible_adds:
        return []
    add_index = {id(node): i for i, node in enumerate(eligible_adds)}

    parent = list(range(len(eligible_adds)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    Edge = Tuple[
        str,
        Optional[Union[Tuple[_Producer, int], onnx.NodeProto]],
        Tuple[_ConvPassThrough, ...],
        Tuple[onnx.NodeProto, ...],
    ]
    edge_results: Dict[int, List[Edge]] = {}
    poisoned: Set[int] = set()
    for idx, add_node in enumerate(eligible_adds):
        results: List[Edge] = []
        for operand in add_node.input:
            edge = _walk_conv_producer_backward(
                operand,
                node_by_output,
                initializer_map,
                consumers_of,
                graph_outputs,
                _MAX_CHAIN_HOPS,
            )
            results.append(edge)
            kind, payload = edge[0], edge[1]
            if kind == "fail":
                poisoned.add(idx)
            elif kind == "add":
                assert isinstance(payload, onnx.NodeProto)
                j = add_index.get(id(payload))
                if j is None:
                    poisoned.add(idx)  # defensive -- shouldn't happen
                else:
                    union(idx, j)
        edge_results[idx] = results

    groups: Dict[int, List[int]] = {}
    for idx in range(len(eligible_adds)):
        groups.setdefault(find(idx), []).append(idx)

    chains: List[_Chain] = []
    for members in groups.values():
        if any(i in poisoned for i in members):
            continue

        leaf_producers: List[_Producer] = []
        n_channels_set: Set[int] = set()
        pass_through: List[_ConvPassThrough] = []
        unary_ops: List[onnx.NodeProto] = []
        referenced: Set[int] = set()

        for idx in members:
            for kind, payload, pt, uops in edge_results[idx]:
                pass_through.extend(pt)
                unary_ops.extend(uops)
                if kind == "producer":
                    assert payload is not None and not isinstance(
                        payload, onnx.NodeProto
                    )
                    producer, n_channels = payload
                    leaf_producers.append(producer)
                    n_channels_set.add(n_channels)
                elif kind == "add":
                    assert isinstance(payload, onnx.NodeProto)
                    referenced.add(add_index[id(payload)])

        if len(n_channels_set) != 1:
            continue  # branches disagree on channel count -- decline
        n_channels = next(iter(n_channels_set))

        # Every depthwise pass-through hop was only self-consistently
        # checked when first crossed (see _match_conv_pass_through_self);
        # now that the group's real channel count is known, re-validate.
        if any(
            initializer_map[hop.weight].dims[0] != n_channels for hop in pass_through
        ):
            continue

        sinks = [idx for idx in members if idx not in referenced]
        if len(sinks) != 1:
            continue  # not a single linear chain of merges -- decline
        sink_add = eligible_adds[sinks[0]]

        if len({p.weight for p in leaf_producers}) != len(leaf_producers):
            continue  # degenerate -- the same producer named twice

        sink_out = sink_add.output[0]
        if not _is_internal(sink_out):
            continue

        consumer, fwd_chain_ops, fwd_pass_through = _walk_to_conv_consumer(
            sink_out,
            initializer_map,
            consumers_of,
            graph_outputs,
            n_channels,
            _MAX_CHAIN_HOPS,
        )
        if consumer is None:
            continue
        consumer_node, consumer_weight, consumer_group = consumer
        if consumer_group != 1:
            # As with the producer side above, a general grouped Conv
            # consumer is out of scope here: _chain_group's per-group
            # top-k assumes every producer feeds the consumer's full
            # (ungrouped) channel range, an assumption a multi-producer
            # residual group doesn't verify against per-group slicing.
            continue

        chain_ops = (
            tuple((op, None) for op in unary_ops)
            + tuple((eligible_adds[i], None) for i in members)
            + fwd_chain_ops
        )

        chains.append(
            _Chain(
                producers=tuple(leaf_producers),
                chain_ops=chain_ops,
                consumer_node=consumer_node,
                consumer_weight=consumer_weight,
                consumer_weight_transposed=False,
                n_channels=n_channels,
                consumer_is_conv=True,
                conv_pass_through=tuple(pass_through) + fwd_pass_through,
            )
        )
    return chains


# --- MatMul/Gemm residual (Add-merged) chains -------------------------------
#
# The MatMul/Gemm analogue of the Conv residual/Add-merge grouping above --
# same union-find-over-eligible-`Add`-merge-points construction, same
# provably-safe special case (`y = Add(a, b)`, two non-constant operands,
# forces whichever real producer(s) feed `a`/`b` to agree on one shared
# channel-index set), same single-consumer safety bar on every hop of every
# branch, and the same reasoning for why an *interior* block of a deep
# residual stack (its own "post-block" tensor read both by the next block
# and directly by that block's own `Add`) is declined while a lone residual
# connection or a linear stack of `Add`-only merges is reached. See
# `_find_conv_residual_chains`'s own section comment above for the shared
# half of that reasoning; only what's different for MatMul/Gemm is covered
# here. `_is_eligible_add_merge` above is reused unchanged -- it was never
# Conv-specific to begin with (it only inspects the `Add` node's own
# operands against `initializer_map`), so no MatMul-specific variant is
# needed.
#
# This is exactly the shape every current transformer block's residual
# stream takes -- `x = x + SelfAttn(LN(x))`, `x = x + MLP(LN(x))` -- the
# single most valuable gap this closes versus the Conv-only residual
# support above, since a MatMul/Gemm residual chain was previously declined
# outright (see this module's own docstring's prior "MatMul/Gemm
# residuals ... remains out of scope" sentence, now narrowed).
#
# One real structural difference from the Conv version, not just a
# find-and-replace of "Conv" with "MatMul/Gemm": `_walk_to_consumer` (the
# MatMul/Gemm forward walk) allows a *wider* hop set than
# `_walk_to_conv_consumer` does -- not just unary activations, but also a
# per-channel bias/scale `Add`/`Mul` against a *constant* initializer (see
# this module's own docstring's "shape-preserving elementwise ops ...
# and for MatMul/Gemm also a bias/scale add/mul" phrase, and
# `_walk_to_consumer`'s own `_BINARY_CHANNEL_OPS` branch). There is no
# MatMul/Gemm analogue of a depthwise-Conv pass-through hop at all -- nothing
# in the MatMul/Gemm producer/consumer vocabulary mixes channels
# transparently the way a depthwise Conv does -- so `_walk_matmul_producer_backward`
# below mirrors *that* wider hop set symmetrically instead: unary
# pass-through activations (as before) plus a per-channel `Add`/`Mul` against
# a constant, walked backward through to whichever tensor it combined with.
# Distinguishing that per-channel bias/scale hop from an eligible residual
# merge is the crux of the whole backward walk, and it falls out for free
# from `_is_eligible_add_merge`'s own definition: an `Add` with exactly one
# constant operand can never be an eligible merge (it requires *both*
# operands non-constant), so it is unambiguously a bias hop instead, and a
# `Mul` is never a merge candidate at all (only `Add` is, per
# `_is_eligible_add_merge`'s own op-type check) -- a per-channel `Mul` hop and
# a residual `Add` merge are never the same node under any input shape.
#
# Because the backward walk doesn't yet know the group's real, shared
# channel count at the point it first crosses a bias/scale hop (the same
# situation `_match_conv_pass_through_self` documents for a depthwise Conv
# hop), it only self-consistently checks the constant is float and
# effectively a flat per-last-axis vector (`prod(dims) == dims[-1]`) when
# first crossed, deferring the real `dims[-1] == n_channels` check to
# `_find_matmul_residual_chains` once the group's producers establish it --
# exactly the same defer-then-revalidate split the depthwise Conv hop above
# uses for its own group-count check.
#
# Two compositions this was checked against and found *not* safe to handle
# silently, so both are declined outright (the group is poisoned, left
# untouched) rather than guessed at, the same conservative choice
# `_find_conv_chains` makes for "both sides grouped with a different group
# count":
#
# - **A gated (SwiGLU/GeGLU-style) pair feeding directly into a residual
#   branch with no downstream projection in between** -- `Add(x, Mul(gate,
#   up))` rather than the usual `Add(x, MatMul(Mul(gate, up), Wd))`. A `Mul`
#   node reached while walking backward always has *two* non-constant
#   operands to choose between (unlike a bias/scale `Mul`, which has exactly
#   one constant operand and is walked through as an ordinary hop above) --
#   there is no way to pick "the" single upstream producer through it without
#   silently assuming which operand is "the" branch and dropping the other's
#   contribution to the combined channel-index set entirely, so this case
#   simply isn't reachable by any hop this walk recognizes and falls straight
#   through to `"fail"`. The far more common shape -- a gated FFN's own
#   *output* projection (`Wd`) feeding a residual `Add` -- needs no special
#   handling at all: `Wd` is an entirely ordinary MatMul/Gemm producer to
#   `_match_producer`, gated combine or not upstream of it, exactly the
#   `x = x + MLP(LN(x))` shape this feature exists for. See
#   `test_structured_pruning_matmul_residual_add_declines_on_gated_branch_with_no_projection`.
# - **A residual branch whose backward walk would need to cross a fused
#   self-attention op boundary** (`com.microsoft::Attention`,
#   `GroupQueryAttention`, or the plain `ai.onnx` `Attention` -- see the
#   "Attention-head pruning" section far below) to reach a real producer --
#   e.g. `Add(x, GroupQueryAttention(q, k, v))` with no output projection
#   MatMul between the attention op and the `Add`. None of those ops is a
#   MatMul/Gemm (`_match_producer` never matches them), an `Add` (never an
#   eligible-merge candidate), or one of `_UNARY_PASS_THROUGH`'s shape-preserving
#   activations, so a residual branch that bottoms out at one is simply
#   unrecognized by any hop this walk knows and falls through to `"fail"` --
#   the same outcome as any other unmatched topology, not a special case that
#   needed its own check. The realistic version of this pattern -- an
#   attention block's own output-projection MatMul (`Wo`) feeding the
#   residual `Add`, with `GroupQueryAttention`/`Attention` sitting further
#   upstream of `Wo` -- needs no special handling either, for the same reason
#   the gated-FFN case above doesn't: the backward walk starts at the `Add`
#   operand and stops at the very first node it finds (`Wo`, an ordinary
#   MatMul/Gemm producer), never looking any further upstream at what feeds
#   `Wo`. See
#   `test_structured_pruning_matmul_residual_add_declines_on_bare_gqa_shortcut`.
#
# A bare `Add` is not, in practice, what the realistic target for this whole
# residual mechanism -- a transformer already run through onnxruntime's own
# transformer-optimizer tool, the same optimization pass that produces the
# `com.microsoft::Attention`/`GroupQueryAttention` fused ops this module
# already targets elsewhere -- actually has at each residual connection: that
# optimizer fuses `Add(input, skip)` (plus an optional per-channel bias
# `Add`) and the *following* `LayerNorm`/`SimplifiedLayerNormalization` into
# one `com.microsoft::SkipLayerNormalization`/`SkipSimplifiedLayerNormalization`
# node (`skip_layer_norm.cc`'s own `ComputeJob`, confirmed against
# onnxruntime's schema (`bert_defs.cc`) and by direct execution --
# `sum = input + skip (+ bias)`; `SkipLayerNormalization` computes ordinary
# LayerNorm on `sum` (population mean/variance, `* gamma + beta` if `beta`
# is given); `SkipSimplifiedLayerNormalization` -- the RMSNorm variant
# LLaMA-style models use -- drops `beta`/mean-centering entirely:
# `sum / sqrt(mean(sum**2) + epsilon) * gamma`). So a fully-optimized
# transformer typically has *no* bare `Add` at its residual connections at
# all, and without also recognizing this fused node the feature above would
# rarely fire on the models it exists for. `_match_matmul_residual_merge`
# closes that gap: such a node is simultaneously (1) the residual merge
# point itself -- its first two inputs, `input`/`skip`, are exactly `Add`'s
# two operands, walked backward the same way -- *and* (2) a per-channel
# affine hop on top of that sum, since `gamma` (and `beta`/`bias`, if
# given) scale/shift each surviving channel independently and so must be
# sliced by the group's own `keep` set precisely like a bias/scale `Add`/
# `Mul` hop's own constant already is. Rather than inventing a new shape for
# that, `_match_matmul_residual_merge` returns those constants as two or
# three synthetic `(node, const_name)` `chain_ops` entries against the same
# node -- `_Chain.chain_ops` already tolerates more than one entry per node
# (nothing about it assumes one entry per distinct node), so
# `_apply_chains`'s existing per-hop constant-slicing loop, its touched-role
# conflict tracking, and its stale-`value_info` cleanup all pick every one
# of them up with no changes of their own. `beta` (`SkipLayerNormalization`
# only) and `bias` are the op's own optional inputs -- simply absent
# becomes no slice needed for that term; *present but non-constant* (like
# `gamma`, required and always checked) declines the node outright, the
# same as a non-constant bias on a MatMul/Gemm producer. The op's optional
# `mean`/`inv_std_var` outputs are training-only bookkeeping onnxruntime's
# own CPU kernel never actually populates (`skip_layer_norm.cc`'s `Compute`
# only ever writes outputs 0 and 3); a real inference-exported graph should
# never wire them anywhere, but if one somehow does, this pass has no basis
# for whether pruning keeps those values meaningful for whatever reads them,
# so it declines outright rather than guessing, same as everywhere else in
# this pass. The optional fourth output, `input_skip_bias_sum` (the raw,
# pre-normalization sum), gets the same "declines if consumed" treatment,
# for a different reason: this pass never reads it itself, and in the
# common post-LN transformer shape a later residual connection's own `skip`
# operand is simply the *previous* `SkipLayerNormalization`'s ordinary
# (normalized) `output` -- not that raw sum -- so the "resolves to another
# eligible merge node's raw output" backward-walk case
# :func:`_walk_matmul_producer_backward` already handles for a chain of
# bare `Add`s covers a chain of `SkipLayerNormalization` nodes for free.
# But if *something else* in the graph reads `input_skip_bias_sum`
# directly, pruning still changes its width -- it's a plain runtime sum of
# `input`/`skip`, so it naturally comes out however wide those two end up
# post-pruning, no static array to reslice -- and this pass has no way to
# confirm that other consumer expects the new width rather than the
# original one (confirmed concretely: a second graph output reading it
# directly ends up with a shape mismatch against its own originally
# declared shape). So it's declined whenever consumed by anything, the
# same conservative bar `mean`/`inv_std_var` get, just for a shape reason
# rather than a values-still-meaningful one.


_SKIP_LAYER_NORM_OPS = ("SkipLayerNormalization", "SkipSimplifiedLayerNormalization")
_SKIP_LAYER_NORM_DOMAIN = "com.microsoft"


def _skip_layer_norm_const_names(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
    """If every constant input a ``com.microsoft::SkipLayerNormalization``/
    ``SkipSimplifiedLayerNormalization`` `node` needs sliced -- `gamma`
    (input 2, required), plus `beta` (input 3, ``SkipLayerNormalization``
    only) and `bias` (input 4, or input 3 for the simplified/RMSNorm
    variant, which has no `beta`), both optional -- is present exactly as
    the node's own input list says, and, whenever present, a constant float
    initializer shaped like a flat per-channel vector (``prod(dims) ==
    dims[-1]``, the same self-consistency bar
    :func:`_walk_matmul_producer_backward`'s own bias/scale hop check
    already uses -- the real ``dims[-1] == n_channels`` check is deferred to
    :func:`_find_matmul_residual_chains` once the group's real channel
    count is known, exactly like that hop's own), returns
    ``(gamma_name, beta_name_or_None, bias_name_or_None)``. `beta`/`bias`
    simply absent from the node's own input list (as opposed to present but
    non-constant) becomes ``None`` -- the corresponding term the kernel
    itself omits, confirmed against onnxruntime's own ``skip_layer_norm.cc``
    kernel and by direct execution (see this section's own comment).
    Declines (``None``) on a non-constant `gamma`, a *present* but
    non-constant `beta`/`bias`, or the same underlying tensor named for two
    of `gamma`/`beta`/`bias` at once (double-slicing it in
    :func:`_apply_chains`'s own per-hop loop would corrupt it) -- none of
    these is guessed at.
    """
    simplified = node.op_type == "SkipSimplifiedLayerNormalization"

    def _const_vec(name: str) -> bool:
        init = initializer_map.get(name)
        return (
            init is not None
            and init.data_type == onnx.TensorProto.FLOAT
            and bool(list(init.dims))
            and int(np.prod(init.dims)) == init.dims[-1]
        )

    if len(node.input) < 3 or not node.input[2] or not _const_vec(node.input[2]):
        return None  # gamma is required
    gamma_name = node.input[2]

    beta_name: Optional[str] = None
    bias_idx = 3
    if not simplified:
        bias_idx = 4
        if len(node.input) > 3 and node.input[3]:
            if not _const_vec(node.input[3]):
                return None
            beta_name = node.input[3]

    bias_name: Optional[str] = None
    if len(node.input) > bias_idx and node.input[bias_idx]:
        if not _const_vec(node.input[bias_idx]):
            return None
        bias_name = node.input[bias_idx]

    names = [n for n in (gamma_name, beta_name, bias_name) if n is not None]
    if len(set(names)) != len(names):
        return None  # tied gamma/beta/bias -- double-slicing would corrupt it

    return gamma_name, beta_name, bias_name


def _match_matmul_residual_merge(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
) -> Optional[Tuple[Tuple[str, str], Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]]]:
    """The MatMul/Gemm residual finder's own eligible-merge-point check:
    `node` is either a bare ``Add`` (:func:`_is_eligible_add_merge`, reused
    unchanged, with no extra `chain_ops` of its own -- exactly today's
    behavior) *or* a ``com.microsoft::SkipLayerNormalization``/
    ``SkipSimplifiedLayerNormalization`` node (see this section's own
    comment above for the exact fused arithmetic and how it was confirmed).
    Its first two inputs (`input`, `skip`) play exactly the role `Add`'s two
    operands do -- same "two independent branches must agree on one
    channel-index set" merge point, same eligibility bar (distinct, both
    non-constant) -- while its constant `gamma`/`beta`/`bias` inputs (see
    :func:`_skip_layer_norm_const_names`) are a per-channel affine hop
    riding the very same node, so this returns them as extra
    ``(node, const_name)`` entries for the caller to fold into the resolved
    chain's own `chain_ops`, reusing :func:`_apply_chains`'s existing
    per-hop constant slicing verbatim -- the same way a bias/scale
    ``Add``/``Mul`` hop's own single constant already does, just two or
    three entries against the same node instead of one.

    Declines (``None``) the same way :func:`_skip_layer_norm_const_names`
    does for a non-constant/tied `gamma`/`beta`/`bias`, and additionally
    whenever any of the op's optional secondary outputs -- `mean`/
    `inv_std_var` (training-only; onnxruntime's own CPU kernel never
    actually writes them) *or* `input_skip_bias_sum` (the raw pre-norm sum)
    -- are actually consumed by anything else in the graph. `mean`/
    `inv_std_var`: this pass has no basis for whether pruning keeps those
    still meaningful for whatever reads them. `input_skip_bias_sum` is
    different in kind -- this pass never reads it itself, and its *shape*
    (not its meaningfulness) is what's at risk: it naturally comes out
    however wide `input`/`skip` end up post-pruning (a plain runtime sum of
    two already-consistently-pruned tensors, nothing to reslice), but any
    *other* consumer of it outside this chain has no idea that width just
    changed and may expect the original one -- confirmed concretely: a
    second graph output reading it directly ends up with a shape mismatch
    against its own originally-declared shape once pruned. So this output
    is held to the same "not consumed elsewhere" bar as `mean`/
    `inv_std_var`, not because its value would be wrong, but because
    nothing here can confirm whatever reads it still expects the resulting
    shape -- the same "no basis to guess a shape survives" reasoning this
    module already applies to fan-out generally.
    """
    if _is_eligible_add_merge(node, initializer_map):
        return (node.input[0], node.input[1]), ()

    if (
        node.domain != _SKIP_LAYER_NORM_DOMAIN
        or node.op_type not in _SKIP_LAYER_NORM_OPS
    ):
        return None
    if len(node.input) < 3:
        return None
    input_name, skip_name = node.input[0], node.input[1]
    if (
        not input_name
        or not skip_name
        or input_name == skip_name
        or input_name in initializer_map
        or skip_name in initializer_map
    ):
        return None

    const_names = _skip_layer_norm_const_names(node, initializer_map)
    if const_names is None:
        return None
    gamma_name, beta_name, bias_name = const_names

    for out_idx in (1, 2, 3):  # mean, inv_std_var, input_skip_bias_sum
        if len(node.output) > out_idx and node.output[out_idx]:
            out_name = node.output[out_idx]
            if consumers_of.get(out_name) or out_name in graph_outputs:
                return None

    extra_ops = tuple(
        (node, name) for name in (gamma_name, beta_name, bias_name) if name is not None
    )
    return (input_name, skip_name), extra_ops


def _walk_matmul_producer_backward(
    start: str,
    node_by_output: Dict[str, onnx.NodeProto],
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    max_hops: int,
) -> Tuple[
    str,
    Optional[Union[Tuple[_Producer, int], onnx.NodeProto]],
    Tuple[Tuple[onnx.NodeProto, Optional[str]], ...],
]:
    """The backward counterpart of :func:`_walk_to_consumer`, used only by
    :func:`_find_matmul_residual_chains` to resolve one operand of an
    eligible merge node (see :func:`_match_matmul_residual_merge`) back to
    whatever produces it -- the MatMul/Gemm analogue of
    :func:`_walk_conv_producer_backward` (see this function's own section
    comment above for how the two differ: a wider hop set mirroring
    `_walk_to_consumer`'s own per-channel bias/scale ``Add``/``Mul`` hop,
    and no depthwise-pass-through analogue at all). Requires every tensor
    crossed -- `start` itself included -- to have exactly one consumer, the
    same safety bar every other hop in this module already holds every
    intermediate tensor to; this is what makes a shared "post-block" tensor
    (read by both the next stage and directly by a merge node) resolve to
    ``"fail"`` rather than being silently pruned wrong. The usual
    exactly-one-output check on every node crossed is relaxed to "its own
    *first* output is `cur`" rather than "it has exactly one output" purely
    to let a multi-output ``SkipLayerNormalization``-family node (`mean`/
    `inv_std_var`/`input_skip_bias_sum`, all beyond its primary `output`)
    through to :func:`_match_matmul_residual_merge`'s own check below --
    every other node type this walk ever matches (`MatMul`/`Gemm`, a unary
    activation, `Add`/`Mul`) already has exactly one output per its own
    ONNX schema, so this is a no-op relaxation for them.

    Returns one of:

    - ``("producer", (producer, n_channels), chain_ops)`` -- resolved all
      the way back to a real MatMul/vanilla-Gemm producer;
    - ``("add", merge_node, chain_ops)`` -- resolved to another eligible
      merge node's raw output instead -- a bare ``Add`` or a
      ``SkipLayerNormalization``-family node alike (the caller unions this
      group with that node's own rather than treating it as a separate
      producer);
    - ``("fail", None, ())`` -- a graph input, an unrecognized producer
      (attention-op boundary, gated-Mul combine, ...), a branch, or the hop
      limit -- the caller declines the whole group this operand belongs to.

    `chain_ops` mirrors :class:`_Chain`'s own field exactly (each entry a
    ``(node, const_name_or_None)`` pair, in forward -- producer-to-merge --
    order), so it can be concatenated directly into the resolved chain's
    `chain_ops` the same way `_find_chains` builds them for an ordinary
    single-producer chain.
    """
    chain_ops: List[Tuple[onnx.NodeProto, Optional[str]]] = []
    cur = start
    for _hop in range(max_hops):
        if len(consumers_of.get(cur, [])) != 1 or cur in graph_outputs:
            return "fail", None, ()
        node = node_by_output.get(cur)
        if node is None or not node.output or node.output[0] != cur:
            return "fail", None, ()

        prod_info = _match_producer(node, initializer_map)
        if prod_info is not None:
            w_name, weight_transposed, bias_name, n_channels = prod_info
            producer = _Producer(node, w_name, weight_transposed, bias_name)
            return "producer", (producer, n_channels), tuple(reversed(chain_ops))

        if node.op_type in _UNARY_PASS_THROUGH and len(node.input) == 1:
            chain_ops.append((node, None))
            cur = node.input[0]
            continue

        if node.op_type in _BINARY_CHANNEL_OPS and len(node.input) == 2:
            a_name, b_name = node.input
            a_const = a_name in initializer_map
            b_const = b_name in initializer_map
            if a_const != b_const:
                const_name, other = (a_name, b_name) if a_const else (b_name, a_name)
                const_init = initializer_map[const_name]
                if (
                    const_init.data_type == onnx.TensorProto.FLOAT
                    and list(const_init.dims)
                    and int(np.prod(const_init.dims)) == const_init.dims[-1]
                ):
                    chain_ops.append((node, const_name))
                    cur = other
                    continue
                return "fail", None, ()
            # Both operands constant (degenerate) or both non-constant: for
            # `Add` the latter is exactly `_is_eligible_add_merge`'s own
            # shape, handled below; for `Mul` it's a gated (SwiGLU/GeGLU)
            # combine point this walk doesn't try to pick a branch through
            # (see this section's own comment) -- either way, falling
            # through to the merge check (which requires `Add` or a
            # ``SkipLayerNormalization``-family node specifically) or
            # `"fail"` is correct.

        if node.op_type in _FUSED_BIAS_GELU_OPS:
            fused = _match_fused_bias_gelu(node, initializer_map)
            if fused is not None:
                data_name, bias_name = fused
                chain_ops.append((node, bias_name))
                cur = data_name
                continue
            return "fail", None, ()

        merge = _match_matmul_residual_merge(
            node, initializer_map, consumers_of, graph_outputs
        )
        if merge is not None:
            return "add", node, tuple(reversed(chain_ops))

        return "fail", None, ()

    return "fail", None, ()


def _find_matmul_residual_chains(graph: onnx.GraphProto) -> List[_Chain]:
    """Finds MatMul/Gemm residual/skip-connection groups -- see this
    section's own comment above and :func:`_find_conv_residual_chains`'s
    (this function mirrors that one's union-find structure exactly, over
    :func:`_walk_matmul_producer_backward` instead of
    :func:`_walk_conv_producer_backward`). Every eligible merge point
    (:func:`_match_matmul_residual_merge` -- a bare ``Add`` or a
    ``SkipLayerNormalization``-family node) contributes its own extra
    `chain_ops` (empty for `Add`; `gamma`/`beta`/`bias` for the
    normalization-fused case) up front, before any union-find grouping, so
    every member of a resolved group -- not just its "sink" -- has its own
    per-channel constants, if any, folded into the final chain the same
    way. For every maximal union-find group of transitively-connected
    eligible merge points, resolves every member's two operands: each must
    reach either a real MatMul/vanilla-Gemm producer (a "leaf" of the
    group) or another eligible merge node already in the same group. If
    *any* operand, anywhere in the group, fails to resolve that way, or the
    leaf producers' channel counts don't all agree, the *entire* group is
    declined -- never partially pruned. A correctly-resolved group must
    reduce to exactly one "sink" merge node (the one whose own output isn't
    itself consumed by another merge node in the group); the chain walk
    continues forward from there via the ordinary :func:`_walk_to_consumer`
    to find the group's one real downstream consumer, exactly as any other
    MatMul/Gemm chain does -- unaffected by which kind of node the sink
    happens to be, since it only ever reads the sink's own raw output
    tensor name.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    node_by_output = {out: node for node in graph.node for out in node.output}
    graph_outputs = {o.name for o in graph.output}

    def _is_internal(name: str) -> bool:
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    Merge = Tuple[
        onnx.NodeProto,
        Tuple[str, str],
        Tuple[Tuple[onnx.NodeProto, Optional[str]], ...],
    ]
    merges: List[Merge] = []
    for node in graph.node:
        match = _match_matmul_residual_merge(
            node, initializer_map, consumers_of, graph_outputs
        )
        if match is not None:
            operands, extra_ops = match
            merges.append((node, operands, extra_ops))
    if not merges:
        return []
    merge_index = {id(m[0]): i for i, m in enumerate(merges)}

    parent = list(range(len(merges)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    Edge = Tuple[
        str,
        Optional[Union[Tuple[_Producer, int], onnx.NodeProto]],
        Tuple[Tuple[onnx.NodeProto, Optional[str]], ...],
    ]
    edge_results: Dict[int, List[Edge]] = {}
    poisoned: Set[int] = set()
    for idx, (merge_node, operands, _extra_ops) in enumerate(merges):
        results: List[Edge] = []
        for operand in operands:
            edge = _walk_matmul_producer_backward(
                operand,
                node_by_output,
                initializer_map,
                consumers_of,
                graph_outputs,
                _MAX_CHAIN_HOPS,
            )
            results.append(edge)
            kind, payload = edge[0], edge[1]
            if kind == "fail":
                poisoned.add(idx)
            elif kind == "add":
                assert isinstance(payload, onnx.NodeProto)
                j = merge_index.get(id(payload))
                if j is None:
                    poisoned.add(idx)  # defensive -- shouldn't happen
                else:
                    union(idx, j)
        edge_results[idx] = results

    groups: Dict[int, List[int]] = {}
    for idx in range(len(merges)):
        groups.setdefault(find(idx), []).append(idx)

    chains: List[_Chain] = []
    for members in groups.values():
        if any(i in poisoned for i in members):
            continue

        leaf_producers: List[_Producer] = []
        n_channels_set: Set[int] = set()
        pre_chain_ops: List[Tuple[onnx.NodeProto, Optional[str]]] = []
        referenced: Set[int] = set()

        for idx in members:
            pre_chain_ops.extend(merges[idx][2])  # this merge node's own extra_ops
            for kind, payload, ops in edge_results[idx]:
                pre_chain_ops.extend(ops)
                if kind == "producer":
                    assert payload is not None and not isinstance(
                        payload, onnx.NodeProto
                    )
                    producer, n_channels = payload
                    leaf_producers.append(producer)
                    n_channels_set.add(n_channels)
                elif kind == "add":
                    assert isinstance(payload, onnx.NodeProto)
                    referenced.add(merge_index[id(payload)])

        if len(n_channels_set) != 1:
            continue  # branches disagree on channel count -- decline
        n_channels = next(iter(n_channels_set))

        # Every bias/scale hop's constant (an Add/Mul hop's own, or a fused
        # BiasGelu/FastGelu hop's own bias -- see _match_fused_bias_gelu),
        # and every SkipLayerNorm-family merge's own gamma/beta/bias, was
        # only self-consistently checked when first crossed/matched (see
        # _walk_matmul_producer_backward, _match_matmul_residual_merge); now
        # that the group's real channel count is known, re-validate it
        # actually matches -- mirroring the depthwise-Conv-hop re-validation
        # in _find_conv_residual_chains.
        if any(
            const_name is not None
            and initializer_map[const_name].dims[-1] != n_channels
            for _, const_name in pre_chain_ops
        ):
            continue

        sinks = [idx for idx in members if idx not in referenced]
        if len(sinks) != 1:
            continue  # not a single linear chain of merges -- decline
        sink_node = merges[sinks[0]][0]

        if len({p.weight for p in leaf_producers}) != len(leaf_producers):
            continue  # degenerate -- the same producer named twice

        sink_out = sink_node.output[0]
        if not _is_internal(sink_out):
            continue

        consumer, fwd_chain_ops = _walk_to_consumer(
            sink_out,
            initializer_map,
            consumers_of,
            graph_outputs,
            n_channels,
            _MAX_CHAIN_HOPS,
        )
        if consumer is None:
            continue

        chain_ops = (
            tuple(pre_chain_ops)
            + tuple((merges[i][0], None) for i in members)
            + fwd_chain_ops
        )

        chains.append(
            _Chain(
                producers=tuple(leaf_producers),
                chain_ops=chain_ops,
                consumer_node=consumer[0],
                consumer_weight=consumer[1],
                consumer_weight_transposed=consumer[2],
                n_channels=n_channels,
            )
        )
    return chains


def _trace_gate_producer_backward(
    tensor_name: str,
    node_by_output: Dict[str, onnx.NodeProto],
    producer_infos: Dict[str, Tuple[onnx.NodeProto, str, bool, Optional[str], int]],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    max_hops: int,
) -> Optional[
    Tuple[
        Tuple[onnx.NodeProto, str, bool, Optional[str], int], Tuple[onnx.NodeProto, ...]
    ]
]:
    """Walks backward from `tensor_name` through unary activation ops
    (Sigmoid, Gelu, ...) until it resolves to a matmul-like producer's raw
    output -- the mirror image of :func:`_walk_to_consumer`'s forward walk,
    used to recognize a gate branch's own activation (e.g. SwiGLU's
    ``silu(gate)`` when exported as separate Sigmoid/Mul-by-a-second-
    operand rather than a single node -- see :func:`_find_gated_chains`).
    Every tensor walked through, `tensor_name` itself included, must have
    exactly one consumer and not be a graph output: the same safety bar
    the forward walk holds every intermediate tensor to.
    """
    pre_ops: List[onnx.NodeProto] = []
    cur = tensor_name
    for _ in range(max_hops):
        if len(consumers_of.get(cur, [])) != 1 or cur in graph_outputs:
            return None
        if cur in producer_infos:
            return producer_infos[cur], tuple(reversed(pre_ops))
        producer_node = node_by_output.get(cur)
        if producer_node is None:
            return None
        if not (
            producer_node.op_type in _UNARY_PASS_THROUGH
            and len(producer_node.input) == 1
            and len(producer_node.output) == 1
        ):
            return None
        pre_ops.append(producer_node)
        cur = producer_node.input[0]
    return None


def _find_gated_chains(graph: onnx.GraphProto) -> List[_Chain]:
    """Finds gated FFN blocks -- SwiGLU/GeGLU-style ``down(act(gate(x)) *
    up(x))``, the FFN architecture most current LLMs use (Llama, Mistral,
    Qwen, Gemma, ...) -- that :func:`_find_chains` cannot see at all,
    because it only ever follows a *single* producer's output. Two
    matmul-like producers (gate and up) whose outputs, each optionally
    through its own activation, combine via one of:

    - a plain elementwise ``Mul`` of two non-constant operands (covers an
      unactivated GLU, or any activation expressed as ordinary unary ops
      -- e.g. GeGLU's ``Gelu``); or
    - ONNX's native fused ``SwiGLU(a, b[, alpha]) = swish(a) * b`` node
      (opset 28+), whose swish lives entirely inside the op, so ``a``/``b``
      must be the two producers' raw outputs with nothing in between,

    with no other consumer anywhere along either branch or at the combine
    point, into exactly one downstream MatMul/vanilla-Gemm's reduction
    dimension, are pruned together: both branches must drop the *same*
    output-channel indices, since they're about to be multiplied
    elementwise. A gate activation decomposed into more than one node
    (e.g. SiLU exported as the self-referencing ``x * Sigmoid(x)`` rather
    than a single ``Sigmoid``/native ``Swish``) isn't recognized -- that
    block is safely left untouched, not guessed at.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    graph_outputs = {o.name for o in graph.output}
    node_by_output = {out: node for node in graph.node for out in node.output}

    def _is_internal(name: str) -> bool:
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    producer_infos: Dict[str, Tuple[onnx.NodeProto, str, bool, Optional[str], int]] = {}
    for node in graph.node:
        info = _match_producer(node, initializer_map)
        if info is not None:
            w_name, weight_transposed, bias_name, n_channels = info
            producer_infos[node.output[0]] = (
                node,
                w_name,
                weight_transposed,
                bias_name,
                n_channels,
            )

    def _producer(info, pre_ops) -> _Producer:
        node, w_name, weight_transposed, bias_name, _n = info
        return _Producer(node, w_name, weight_transposed, bias_name, pre_ops)

    chains: List[_Chain] = []
    for node in graph.node:
        if node.op_type == "Mul" and len(node.input) == 2 and len(node.output) == 1:
            a_name, b_name = node.input
            if (
                a_name == b_name
                or a_name in initializer_map
                or b_name in initializer_map
            ):
                continue
            trace_a = _trace_gate_producer_backward(
                a_name,
                node_by_output,
                producer_infos,
                consumers_of,
                graph_outputs,
                _MAX_CHAIN_HOPS,
            )
            trace_b = _trace_gate_producer_backward(
                b_name,
                node_by_output,
                producer_infos,
                consumers_of,
                graph_outputs,
                _MAX_CHAIN_HOPS,
            )
            if trace_a is None or trace_b is None:
                continue
            info_a, pre_a = trace_a
            info_b, pre_b = trace_b
        elif (
            node.op_type == "SwiGLU" and len(node.input) == 2 and len(node.output) == 1
        ):
            a_name, b_name = node.input
            if a_name in initializer_map or b_name in initializer_map:
                continue
            if not (_is_internal(a_name) and _is_internal(b_name)):
                continue
            info_a_lookup = producer_infos.get(a_name)
            info_b_lookup = producer_infos.get(b_name)
            if info_a_lookup is None or info_b_lookup is None:
                continue
            info_a, pre_a = info_a_lookup, ()
            info_b, pre_b = info_b_lookup, ()
        else:
            continue

        node_a, n_a = info_a[0], info_a[4]
        node_b, n_b = info_b[0], info_b[4]
        if node_a is node_b or n_a != n_b:
            continue

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue

        consumer, chain_ops = _walk_to_consumer(
            out_name, initializer_map, consumers_of, graph_outputs, n_a, _MAX_CHAIN_HOPS
        )
        if consumer is None:
            continue

        chains.append(
            _Chain(
                producers=(_producer(info_a, pre_a), _producer(info_b, pre_b)),
                chain_ops=chain_ops,
                consumer_node=consumer[0],
                consumer_weight=consumer[1],
                consumer_weight_transposed=consumer[2],
                n_channels=n_a,
            )
        )
    return chains


def _slice_producer_weight(
    w_init: onnx.TensorProto,
    weight_transposed: bool,
    keep: np.ndarray,
    is_conv: bool = False,
) -> None:
    w = onnx.numpy_helper.to_array(w_init)
    if is_conv:
        # [out_channels, in_channels, kH, kW]: output channel is always axis 0.
        w_new = w[keep, ...]
    else:
        # [N, K] storage (transB=1): output channel is axis 0. [K, N]
        # storage (the common case): output channel is axis 1.
        w_new = w[keep, :] if weight_transposed else w[:, keep]
    w_init.CopyFrom(onnx.numpy_helper.from_array(w_new, name=w_init.name))


def _slice_consumer_weight(
    w_init: onnx.TensorProto,
    weight_transposed: bool,
    keep: np.ndarray,
    is_conv: bool = False,
) -> None:
    w = onnx.numpy_helper.to_array(w_init)
    if is_conv:
        # [out_channels, in_channels, kH, kW]: input channel is always axis 1.
        w_new = w[:, keep, ...]
    else:
        # [N, K] storage (transB=1): reduction dim is axis 1. [K, N] storage:
        # reduction dim is axis 0.
        w_new = w[:, keep] if weight_transposed else w[keep, :]
    w_init.CopyFrom(onnx.numpy_helper.from_array(w_new, name=w_init.name))


def _slice_grouped_consumer_conv_weight(
    w_init: onnx.TensorProto,
    keep: np.ndarray,
    group: int,
    n_channels: int,
) -> None:
    """Slices a *general grouped* Conv consumer's ``[out_channels,
    in_channels/group, kH, kW]`` weight by a global (whole-``in_channels``)
    `keep` index set. Unlike :func:`_slice_consumer_weight`'s flat ``w[:,
    keep, ...]`` (correct only for an ordinary ``group=1`` consumer, whose
    axis 1 truly spans every input channel), a grouped consumer's axis 1 is
    only `in_channels/group` wide and is *per-group-relative*: weight
    column ``j`` on output filter ``o`` means global input channel
    ``(o // out_per_group) * block + j`` -- `block` (`n_channels // group`)
    input channels per group, not `j` itself. So each output-filter group
    needs its own local slice of `keep` -- that group's own retained
    channels, translated from global indices back to local ones by
    subtracting the group's own block offset -- rather than one shared
    index set applied uniformly across the whole axis.

    This is well-defined only because whatever produced `keep` already
    guarantees a *uniform count* of survivors per `group`-sized block (see
    :func:`_chain_group`/:func:`_apply_chains`): both producer-grouped and
    consumer-grouped selection independently keep the same count from every
    block by construction, and the "both sides grouped" composition this
    pass supports requires a matching `group` count on both ends (see
    :func:`_find_conv_chains`), so the producer's own blocks and this
    consumer's own blocks are always the exact same partition of
    `n_channels` -- never a case where one side's block boundaries split a
    count unevenly relative to the other's.
    """
    w = onnx.numpy_helper.to_array(w_init)
    out_channels = w.shape[0]
    out_per_group = out_channels // group
    block = n_channels // group
    parts = []
    for gi in range(group):
        lo, hi = gi * block, (gi + 1) * block
        local_keep = keep[(keep >= lo) & (keep < hi)] - lo
        filt_lo, filt_hi = gi * out_per_group, (gi + 1) * out_per_group
        parts.append(w[filt_lo:filt_hi, local_keep, ...])
    w_new = np.concatenate(parts, axis=0)
    w_init.CopyFrom(onnx.numpy_helper.from_array(w_new, name=w_init.name))


def _slice_last_axis(init: onnx.TensorProto, keep: np.ndarray) -> None:
    arr = onnx.numpy_helper.to_array(init)
    new = np.take(arr, keep, axis=-1)
    init.CopyFrom(onnx.numpy_helper.from_array(new, name=init.name))


def _plain_structured_importance(
    chain: _Chain, w_arrays_nk: List[np.ndarray]
) -> np.ndarray:
    # Combined (root-sum-square) importance across every producer in this
    # chain: for a plain chain this is just that producer's own L2 norm;
    # for a gated pair, both branches must agree on which channels survive,
    # so their per-channel norms are combined first.
    squared_norm = np.zeros(chain.n_channels, dtype=np.float64)
    for w_nk in w_arrays_nk:
        squared_norm += np.square(np.linalg.norm(w_nk, axis=1))
    return np.sqrt(squared_norm)


def _chain_group(chain: _Chain) -> int:
    """The `group` count that governs this chain's `keep`-index selection
    (see :func:`_apply_chains`): 1 for every chain this pass already
    supported before general grouped Conv (a MatMul/Gemm chain, a gated
    pair, or an ordinary ``group=1`` Conv producer/consumer -- all leave
    every `group` field at its default), and > 1 whenever either side of a
    Conv chain is a general grouped Conv.

    Worked example for why the producer side takes priority when *only* the
    producer is grouped: a grouped producer (`group=g`, `g` output-channel
    blocks of `out_channels/g` filters each) feeding an ordinary `group=1`
    consumer needs every one of the producer's own `g` blocks pruned to a
    uniform count so `out_channels % g == 0` survives -- a requirement the
    consumer itself doesn't share (an ordinary consumer accepts any subset
    of surviving input channels), so the producer's own grouping is what
    the shared `keep` selection must honor. Symmetrically, an ordinary
    `group=1` producer feeding a grouped consumer (`group=g_c`) has no
    grouping constraint of its own -- any subset of its output channels is
    individually a valid producer-side cut -- so it's the *consumer's* `g_c`
    blocks that constrain which subset is safe to choose, making the
    consumer's `group` the one that governs `keep` selection there. When
    both sides are grouped, :func:`_find_conv_chains` already declined the
    chain unless `producer_group == consumer_group`, so either field gives
    the same answer.
    """
    if len(chain.producers) == 1 and chain.producers[0].group > 1:
        return chain.producers[0].group
    return chain.consumer_group


def _apply_chains(
    graph: onnx.GraphProto,
    chains: List[_Chain],
    sparsity: float,
    compute_importance,
) -> None:
    """Shared body for :func:`apply_structured_pruning` and
    :func:`apply_structured_wanda_pruning`: resolves cross-chain touched-role
    conflicts, computes each surviving chain's target channel count, calls
    ``compute_importance(chain, w_arrays_nk) -> np.ndarray[n_channels]`` for
    the ranking, and performs the actual slicing plus stale ``value_info``
    cleanup. Mutates ``graph`` in place.

    For a chain with :func:`_chain_group` (`group`) > 1 -- a general grouped
    Conv producer or consumer, see this module's own docstring -- `keep` is
    chosen independently *within each of `group` equal-sized blocks* of the
    channel-importance vector, keeping the same count from every block
    (`_chain_group`'s own docstring works through why one side's `group`
    always suffices to pick the block boundaries both roles need to honor).
    This reduces to today's single whole-vector top-k exactly when
    `group == 1` -- the code below keeps that as a literal separate branch,
    not a `group=1` special case of the block formula, so every
    already-supported chain's rounding (and therefore its exact `keep`
    selection) stays byte-identical to before this function learned about
    grouped Conv at all.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    # A weight legitimately plays both roles across two different chains --
    # e.g. the middle layer of a 3-layer MLP is the *consumer* of the first
    # chain (its reduction/input axis gets pruned) and the *producer* of the
    # second (its own output axis gets pruned), two independent axes of the
    # same tensor. Only collapse when the *same role* is claimed twice (a
    # tied/shared weight), tracked separately per role; bias/scale constants
    # only ever play one role, so a single shared set is enough for those.
    producer_touched: Set[str] = set()
    consumer_touched: Set[str] = set()
    const_touched: Set[str] = set()
    conv_hop_touched: Set[str] = set()
    stale_value_info: Set[str] = set()

    for chain in chains:
        producer_weights = {p.weight for p in chain.producers}
        if len(producer_weights) != len(chain.producers):
            continue  # degenerate (a gated pair naming the same weight twice)

        conv_hop_weights = {h.weight for h in chain.conv_pass_through}
        if len(conv_hop_weights) != len(chain.conv_pass_through):
            continue  # degenerate (the same depthwise weight named twice)

        consts = {p.bias for p in chain.producers if p.bias is not None}
        consts.update(
            const_name for _, const_name in chain.chain_ops if const_name is not None
        )
        if (
            (producer_weights & producer_touched)
            or chain.consumer_weight in consumer_touched
            or (consts & const_touched)
            or (conv_hop_weights & conv_hop_touched)
        ):
            continue  # a shared/tied initializer another chain already resized

        n = chain.n_channels
        group = _chain_group(chain)
        if group > 1:
            block = n // group
            per_group_keep = max(1, round(block * (1.0 - sparsity)))
            keep_count = per_group_keep * group
        else:
            keep_count = max(1, n - round(n * sparsity))
        if keep_count >= n:
            continue  # rounds down to nothing for this layer -- no-op

        w_arrays_nk = []
        for p in chain.producers:
            w = onnx.numpy_helper.to_array(initializer_map[p.weight]).astype(np.float64)
            if p.is_conv:
                w_nk = w.reshape(w.shape[0], -1)  # [out_channels, in_channels*kH*kW]
            else:
                w_nk = w if p.weight_transposed else w.T  # [N, K]
            w_arrays_nk.append(w_nk)
        importance = compute_importance(chain, w_arrays_nk)
        if group > 1:
            # One independent top-k per block -- see _chain_group and this
            # function's own docstring. Blocks are contiguous and already
            # increasing, and each block's own local top-k is sorted
            # ascending before its offset is added back, so the
            # concatenation is sorted ascending overall too, same as the
            # group=1 branch's own `keep` invariant.
            keep = np.concatenate(
                [
                    np.sort(
                        np.argsort(-importance[gi * block : (gi + 1) * block])[
                            :per_group_keep
                        ]
                    )
                    + gi * block
                    for gi in range(group)
                ]
            )
        else:
            keep = np.sort(np.argsort(-importance)[:keep_count])

        for p in chain.producers:
            _slice_producer_weight(
                initializer_map[p.weight], p.weight_transposed, keep, is_conv=p.is_conv
            )
            if p.bias is not None:
                _slice_last_axis(initializer_map[p.bias], keep)
        for _, const_name in chain.chain_ops:
            if const_name is not None:
                _slice_last_axis(initializer_map[const_name], keep)
        for hop in chain.conv_pass_through:
            # Same `keep` index set as the real producer -- a depthwise
            # Conv's own channel i is exactly upstream channel i, so its
            # weight (output-channel axis 0, like any Conv producer) and
            # bias slice identically, and `group` (== in_channels ==
            # out_channels for a depthwise Conv) drops to the new count
            # right alongside them.
            _slice_producer_weight(
                initializer_map[hop.weight], False, keep, is_conv=True
            )
            if hop.bias is not None:
                _slice_last_axis(initializer_map[hop.bias], keep)
            found_group = False
            for attr in hop.node.attribute:
                if attr.name == "group":
                    attr.i = keep_count
                    found_group = True
                    break
            if not found_group:
                hop.node.attribute.append(
                    onnx.helper.make_attribute("group", keep_count)
                )
        if chain.consumer_is_conv and chain.consumer_group > 1:
            _slice_grouped_consumer_conv_weight(
                initializer_map[chain.consumer_weight], keep, chain.consumer_group, n
            )
        else:
            _slice_consumer_weight(
                initializer_map[chain.consumer_weight],
                chain.consumer_weight_transposed,
                keep,
                is_conv=chain.consumer_is_conv,
            )

        producer_touched.update(producer_weights)
        consumer_touched.add(chain.consumer_weight)
        const_touched.update(consts)
        conv_hop_touched.update(conv_hop_weights)
        for p in chain.producers:
            stale_value_info.add(p.node.output[0])
            stale_value_info.update(pre_op.output[0] for pre_op in p.pre_ops)
        stale_value_info.update(
            chain_node.output[0] for chain_node, _ in chain.chain_ops
        )
        stale_value_info.update(hop.node.output[0] for hop in chain.conv_pass_through)

    if stale_value_info:
        kept = [vi for vi in graph.value_info if vi.name not in stale_value_info]
        del graph.value_info[:]
        graph.value_info.extend(kept)


def apply_structured_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
) -> onnx.ModelProto:
    """Removes whole output channels from MatMul/vanilla-Gemm layers --
    real structural pruning (smaller weight tensors, smaller matmuls on any
    runtime, not just one with sparse-kernel support), as opposed to
    :func:`apply_magnitude_pruning`/:func:`apply_wanda_pruning`'s value-only
    zeroing. See this module's own docstring for the technique, its L2-norm
    importance metric, and why it's restricted to an unambiguous single
    producer -> consumer topology rather than general dependency-graph
    pruning. :func:`apply_structured_wanda_pruning` is the calibrated
    upgrade of this same technique, exactly as :func:`apply_wanda_pruning`
    is to :func:`apply_magnitude_pruning`.

    For every MatMul/vanilla-Gemm node (the "producer") whose output feeds,
    through zero or more shape-preserving elementwise ops (an activation,
    or an Add/Mul against a constant per-channel bias/scale) with no other
    consumer anywhere along that path, into exactly one downstream
    MatMul/vanilla-Gemm's reduction dimension (the "consumer"): ranks the
    producer's output channels by L2 norm of their own weight row, drops
    the lowest-``sparsity``-fraction of them, and removes the corresponding
    rows/columns from the producer's weight (and bias, if it has a constant
    one) and every intermediate per-channel constant, and the matching
    columns/rows from the consumer's weight -- a shape change that leaves
    the two layers' composition mathematically unaffected for every
    surviving channel.

    The same cut applies to 2-D ``Conv`` producer -> consumer pairs -- each
    output filter's whole ``[in_channels/group, kH, kW]`` kernel ranked by
    its own L2 norm, exactly Li et al.'s original filter-pruning criterion
    -- joined by unary activations and/or depthwise Conv hops (``group ==
    in_channels == out_channels``: one filter per channel, no cross-channel
    mixing, so it's crossed transparently -- its own weight/bias sliced by
    the producer's channel indices and its ``group`` attribute shrunk to
    match, but it contributes no importance of its own and can't itself be
    the producer or consumer -- see this module's own docstring). No
    per-channel Add/Mul between two Convs (a Conv already carries its own
    bias, and ``BatchNormalization`` is expected to already be fused into
    the preceding Conv by the time this pass runs).

    A *general* grouped Conv (``group`` neither 1 nor its channel count) is
    also matched, as a producer and/or a consumer, ranking/pruning each of
    its ``group`` channel blocks independently (see this module's own
    docstring for exactly why that's safe and how the two roles differ). A
    grouped producer paired with an ordinary consumer, an ordinary producer
    paired with a grouped consumer, and both sides grouped *with the same
    ``group`` count* are all supported; both sides grouped with a
    *different* ``group`` count is declined and the chain is left
    completely untouched, same as any other topology this pass can't prove
    safe to cut.

    Also handles the gated FFN pattern most current LLMs use in place of a
    plain two-layer MLP (SwiGLU/GeGLU: ``down(act(gate(x)) * up(x))``, see
    :func:`_find_gated_chains`) -- two producers (gate and up) combined by
    an elementwise product feed one consumer; both branches are ranked by
    combined (root-sum-square) importance and pruned to the *same*
    surviving channel indices, since they're about to be multiplied. This
    gated form is MatMul/Gemm-only -- Conv chains don't take part in it.

    Also handles a bounded slice of the Conv residual/skip-connection case
    (see :func:`_find_conv_residual_chains` and this module's own
    docstring): a channel-preserving ``Add(a, b)`` with two non-constant
    operands -- every residual connection's shape -- forces whichever real
    Conv producer(s) feed `a` and `b` (found by walking backward through the
    same unary-activation/depthwise-pass-through hops the forward walk
    already allows, transitively through any further such `Add` merges
    sharing the same spine) to be pruned to one shared channel-index set,
    ranked the same combined (root-sum-square) way as a gated pair. Bounded,
    not general DepGraph: every branch is still held to the same
    single-consumer bar as everywhere else in this pass, so a real
    multi-block ResNet stage's shared "post-block" tensor -- read by both
    the next block's own first Conv and, unchanged, that block's own `Add`
    -- is a fan-out this declines rather than guesses at; a lone residual
    connection whose branches don't fan out elsewhere, or a linear stack of
    `Add`-only merges, is what this reaches.

    The MatMul/Gemm analogue of that same residual/skip-connection case is
    also handled (see :func:`_find_matmul_residual_chains` and this
    module's own docstring) -- the transformer-block residual stream shape
    (``x = x + SelfAttn(LN(x))``, ``x = x + MLP(LN(x))``) that was
    previously declined outright. Same union-find grouping over eligible
    merge points, same single-consumer bar on every hop of every branch,
    same combined (root-sum-square) importance ranking; the one real
    difference is the backward walk mirrors :func:`_walk_to_consumer`'s own
    *wider* MatMul/Gemm hop set (unary activations plus a per-channel
    bias/scale ``Add``/``Mul`` against a constant) rather than the Conv
    walk's narrower one, since there is no depthwise-Conv-style pass-through
    analogue for MatMul/Gemm at all. A bare ``Add`` merge point is only one
    recognized shape -- since onnxruntime's own transformer-optimizer tool
    typically fuses each residual ``Add`` (plus an optional per-channel bias
    ``Add``) together with the *following* LayerNorm/RMSNorm into one
    ``com.microsoft::SkipLayerNormalization``/
    ``SkipSimplifiedLayerNormalization`` node instead, that fused node is
    recognized as an eligible merge point too (:func:`_match_matmul_residual_merge`):
    its ``input``/``skip`` inputs play ``Add``'s own two-operand role, while
    its ``gamma`` (required) and, if present, ``beta``
    (``SkipLayerNormalization`` only) and ``bias`` are sliced by the group's
    own surviving channel indices alongside everything else, confirmed
    against onnxruntime's own kernel source and by direct execution -- see
    :func:`_find_matmul_residual_chains`'s own section comment for the exact
    fused arithmetic. A gated (SwiGLU/GeGLU) combine feeding a residual
    branch with no downstream projection in between, and a residual branch
    that would need to cross a fused self-attention op boundary
    (``com.microsoft::Attention``/``GroupQueryAttention``/``ai.onnx``
    ``Attention``) to reach a real producer, are both declined rather than
    guessed at -- see the section comment above
    :func:`_walk_matmul_producer_backward` for why neither is actually
    reachable by any hop this walk recognizes, and why the far more common
    shapes (a gated FFN's own output projection, or an attention block's own
    output-projection MatMul, feeding the residual `Add`/`SkipLayerNormalization`)
    need no special handling at all. A non-constant (or, for ``beta``/``bias``,
    present-but-non-constant) ``gamma``/``beta``/``bias``, or a
    ``SkipLayerNormalization``-family node whose optional ``mean``/
    ``inv_std_var`` outputs are actually consumed elsewhere, is declined the
    same conservative way.

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of each matched producer's output
            channels to remove (at least one channel is always kept)
    :returns: ``model`` with every matched chain's tensors resized in
            place; anything not matching that exact topology (branching,
            a non-constant bias, a consumer whose reduction dimension
            doesn't line up, ...) is left completely untouched
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains = (
        _find_chains(graph)
        + _find_gated_chains(graph)
        + _find_conv_chains(graph)
        + _find_conv_residual_chains(graph)
        + _find_matmul_residual_chains(graph)
    )
    if chains:
        _apply_chains(graph, chains, sparsity, _plain_structured_importance)

    return out


def apply_structured_wanda_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    epsilon: float = 1e-8,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """The calibrated upgrade of :func:`apply_structured_pruning`, exactly
    as :func:`apply_wanda_pruning` is to :func:`apply_magnitude_pruning`:
    same real structural channel removal, same topology matching (a single
    producer, a gated pair, or a bounded Conv *or* MatMul/Gemm
    residual/merge group -- an ``Add`` or, for MatMul/Gemm, also a
    ``SkipLayerNormalization``-family node, see :func:`apply_structured_pruning`'s
    own docstring) -> zero or more shape-preserving elementwise ops and, for
    a Conv chain, depthwise Conv hops -> one consumer,
    MatMul/Gemm or Conv, general grouped Conv included on either side, see
    :func:`apply_structured_pruning`'s own docstring) including the same
    depthwise-Conv pass-through sliced by the producer's channel indices
    alone -- it contributes no activation norm of its own to the ranking
    either, being transparent to the chain's channel-index mapping just as
    it is to plain L2-norm importance -- but each chain's
    output channels are ranked by ``||W_row||_2 * ||X||_2`` -- L2 norm of
    that channel's own weight row (or, for Conv, whole filter), times the
    L2 norm of the *activation* actually flowing through that channel over
    calibration data (captured right where the chain feeds into its
    consumer, reduced over every axis but the channel one -- the last axis
    for a MatMul/Gemm consumer, axis 1 of ``[N, C, H, W]`` for a Conv
    consumer) -- instead of weight magnitude alone. This is the same
    protection Wanda's element-wise metric gives unstructured pruning,
    transplanted to whole channels: a channel whose weight is individually
    unremarkable but which gates a consistently high-magnitude activation
    is kept over one with a larger weight norm but a near-dead activation.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            chain's consumer-side activation norm on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of each matched chain's output
            channels to remove (at least one channel is always kept)
    :param epsilon: floor applied to the accumulated per-channel activation
            norm, avoiding every channel of an all-zero activation tying at
            exactly the weight-only importance
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched chain's tensors resized in
            place; anything not matching that exact topology falls back to
            :func:`apply_structured_pruning`'s plain L2-norm ranking if no
            matching activation was ever observed for that chain's consumer
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains = (
        _find_chains(graph)
        + _find_gated_chains(graph)
        + _find_conv_chains(graph)
        + _find_conv_residual_chains(graph)
        + _find_matmul_residual_chains(graph)
    )
    if not chains:
        return out

    # The channel axis of the activation feeding each chain's consumer: a
    # MatMul/Gemm's reduction dimension is its input's last axis, while a
    # Conv's input channel dimension is always axis 1 of [N, C, H, W]. Two
    # chains can't disagree on a shared probe name -- a tensor has exactly
    # one producer node, so it feeds one consumer type.
    channel_axis: Dict[str, int] = {
        chain.consumer_node.input[0]: (1 if chain.consumer_is_conv else -1)
        for chain in chains
    }
    probe_names = sorted(channel_axis)
    probe_model = _add_probe_outputs(out, probe_names)

    sq_sum: Dict[str, np.ndarray] = {}
    count: Dict[str, int] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            axis = channel_axis[name]
            axis = axis if axis >= 0 else x.ndim + axis
            if axis < 0 or axis >= x.ndim:
                continue
            # Sum of squares over every axis but the channel one -- correct
            # for any activation rank, not just the 2-D case.
            reduce_axes = tuple(i for i in range(x.ndim) if i != axis)
            s = np.square(x).sum(axis=reduce_axes) if reduce_axes else np.square(x)
            cnt = int(np.prod(x.shape, dtype=np.int64)) // x.shape[axis]
            sq_sum[name] = s if name not in sq_sum else sq_sum[name] + s
            count[name] = count.get(name, 0) + cnt

    act_norm: Dict[str, np.ndarray] = {
        name: np.sqrt(s / max(count[name], 1)) for name, s in sq_sum.items()
    }

    def _wanda_structured_importance(
        chain: _Chain, w_arrays_nk: List[np.ndarray]
    ) -> np.ndarray:
        base = _plain_structured_importance(chain, w_arrays_nk)
        norm = act_norm.get(chain.consumer_node.input[0])
        if norm is None or norm.shape[0] != chain.n_channels:
            return base  # no matching activation observed -- fall back to |W|
        return base * np.maximum(norm, epsilon)

    _apply_chains(graph, chains, sparsity, _wanda_structured_importance)
    return out


# --- Attention-head pruning -----------------------------------------------

# Three fused self-attention ops are matched here -- two from the
# ``com.microsoft`` domain, produced by onnxsim's own fusion passes from a
# decomposed self-attention block, plus the standard ``ai.onnx`` op -- each
# pruned at the granularity its own kernel contract allows:
#
# - `Attention` (onnxsim/passes/fuse_attention.h): a single merged QKV
#   weight/bias ([hidden_size, Nq+Nk+Nv] / [Nq+Nk+Nv]) plus
#   `num_heads`/`qkv_hidden_sizes` attributes, one `num_heads` shared by
#   Q/K/V alike. Every head owns an equally-sized, independent column block
#   of that merged weight, so individual heads can be dropped one at a time
#   -- see :func:`_apply_one_plain_attention_chain`.
# - `GroupQueryAttention` (onnxsim/passes/fuse_gqa.h): separate, un-merged
#   Q/K/V projections (ordinary MatMul/vanilla-Gemm nodes feeding directly
#   into the op, not weights the op itself owns) plus independent
#   `num_heads` (query heads)/`kv_num_heads` (key/value heads) attributes,
#   `num_heads` a positive multiple of `kv_num_heads`. A contiguous *group*
#   of `num_heads / kv_num_heads` query heads shares each KV head via the
#   kernel's own internal broadcast -- GQA's real-world purpose is fewer KV
#   heads than query heads, exactly the shape Llama 2/3, Mistral, Qwen, and
#   most current open-weight models export. Because every surviving KV head
#   must keep exactly the same number of query heads mapped to it (the
#   kernel requires `num_heads % kv_num_heads == 0` after pruning just as
#   before), an individual query head cannot be dropped in isolation the
#   way plain `Attention` pruning does -- only a *whole KV group* (that KV
#   head's own K/V column block, together with every query head mapped to
#   it) is ever removed at once, ranked by the combined importance of the
#   group's whole Q+K+V block -- see :func:`_apply_one_gqa_chain` and
#   :func:`_gqa_group_importance`. A non-empty constant past-KV-cache input
#   (which would itself need slicing along the `kv_num_heads` axis) or a
#   packed-QKV/missing-required-input node this module cannot prove safe to
#   leave alone is declined outright -- see :func:`_match_gqa_producer`.
# - the plain ``ai.onnx`` `Attention` (opset 24+, domain ``""``, schema
#   confirmed against this environment's installed ``onnx==1.22.0`` via
#   ``onnx.defs.get_schema("Attention", domain="")`` -- it is fully defined
#   there, unlike the still-under-development op ``fuse_attention.h``'s own
#   comment warns about for a different opset/op; see
#   :func:`_match_onnx_attention_producer`'s own docstring for the exact
#   attributes/inputs read off that schema): structurally the same shape as
#   `GroupQueryAttention` -- separate, un-merged Q/K/V projections plus
#   independent `q_num_heads`/`kv_num_heads` attributes, `q_num_heads` a
#   positive multiple of `kv_num_heads` (the op's schema doc names this same
#   MHA/GQA/MQA taxonomy explicitly) -- close enough a cousin that this pass
#   reuses :class:`_GQAChain`, :func:`_apply_one_gqa_chain`, and
#   :func:`_gqa_group_importance` for it outright rather than a parallel
#   implementation (see :func:`_find_separate_qkv_chains`, the two matchers'
#   shared caller). It differs in three ways this pass accounts for: (1) its
#   query-head-count attribute is named `q_num_heads`, not `num_heads` --
#   :class:`_GQAChain` carries which attribute name to write back
#   (`num_heads_attr`); (2) it schema-allows V its own `head_size`
#   independent of Q/K's (confirmed via the op's own backend test suite,
#   e.g. ``test_attention_3d_diff_heads_sizes``) -- since
#   :func:`_apply_one_gqa_chain`'s shared slicing assumes one uniform
#   `head_size` the way `fuse_gqa.h` itself requires, a node whose V head
#   size actually differs is declined rather than mis-sliced; (3) its
#   optional `attn_mask`/`past_key`/`past_value` inputs (a different set
#   from `GroupQueryAttention`'s own `past_key`/`past_value`/`seqlens_k`/
#   `total_sequence_length`/`cos_cache`/`sin_cache`) get the same
#   non-empty-constant-is-declined, dynamic-is-left-alone treatment
#   :func:`_match_gqa_producer` already applies -- see
#   :func:`_match_onnx_attention_producer`. Verified here via actual
#   execution (``onnx.checker`` plus onnxruntime, both of which handle this
#   op in this environment -- see ``tests/test_pruning.py``'s own "plain
#   ai.onnx Attention" section), the same oracle-vs-onnxruntime bar every
#   other function in this module is held to; no structural-only fallback
#   was needed.
_ATTENTION_DOMAIN = "com.microsoft"


@dataclass(frozen=True)
class _AttentionChain:
    node: onnx.NodeProto
    weight: str
    bias: Optional[str]
    num_heads: int
    nq: int
    nk: int
    nv: int
    chain_ops: Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]
    consumer_node: onnx.NodeProto
    consumer_weight: str
    consumer_weight_transposed: bool


@dataclass(frozen=True)
class _GQAChain:
    node: onnx.NodeProto
    q_weight: str
    q_bias: Optional[str]
    q_weight_transposed: bool
    k_weight: str
    k_bias: Optional[str]
    k_weight_transposed: bool
    v_weight: str
    v_bias: Optional[str]
    v_weight_transposed: bool
    num_heads: int
    kv_num_heads: int
    head_size: int
    chain_ops: Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]
    consumer_node: onnx.NodeProto
    consumer_weight: str
    consumer_weight_transposed: bool
    # Which attribute on `.node` holds the query head count:
    # ``com.microsoft::GroupQueryAttention`` names it `num_heads`, the plain
    # ``ai.onnx::Attention`` op (see :func:`_match_onnx_attention_producer`)
    # names the same concept `q_num_heads` -- both share `kv_num_heads`
    # verbatim, so only this one name needs to travel with the chain for
    # :func:`_apply_one_gqa_chain`'s shared write-back to target the right
    # attribute on either op.
    num_heads_attr: str = "num_heads"


# Either kind of matched attention block, sharing enough of a common shape
# (a `.node`, a `.consumer_node`/`.consumer_weight`, `.chain_ops`) that
# :func:`_apply_attention_chains`'s own bookkeeping (touched-role tracking,
# stale value_info cleanup) and the activation-probing setup in
# :func:`apply_attention_head_wanda_pruning` treat both uniformly, only
# dispatching on which one a given chain is for the actual slicing. A
# matched plain ``ai.onnx::Attention`` node is represented as a
# :class:`_GQAChain` too (see that class's own `num_heads_attr` field) --
# it is a *third* matched node type, not a third dataclass, since its
# separate-Q/K/V-producer shape and whole-KV-group pruning unit are
# identical to `GroupQueryAttention`'s own.
_AttnLikeChain = Union[_AttentionChain, _GQAChain]


def _match_attention_producer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, Optional[str], int, int, int, int]]:
    """If `node` is a ``com.microsoft::Attention`` node with a constant 2-D
    float32 merged QKV weight ``[K, Nq+Nk+Nv]`` (and, if present, a
    constant 1-D float32 merged bias), returns
    ``(weight_name, bias_name_or_None, num_heads, Nq, Nk, Nv)``.
    """
    if node.domain != _ATTENTION_DOMAIN or node.op_type != "Attention":
        return None
    if len(node.input) < 2:
        return None
    w_name = node.input[1]
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or w_init.data_type != onnx.TensorProto.FLOAT
        or len(w_init.dims) != 2
    ):
        return None
    total_n = w_init.dims[1]

    bias_name = None
    if len(node.input) >= 3 and node.input[2]:
        bias_name = node.input[2]
        b_init = initializer_map.get(bias_name)
        if (
            b_init is None
            or b_init.data_type != onnx.TensorProto.FLOAT
            or list(b_init.dims) != [total_n]
        ):
            return None

    num_heads = None
    qkv_hidden_sizes: Optional[List[int]] = None
    for attr in node.attribute:
        if attr.name == "num_heads":
            num_heads = attr.i
        elif attr.name == "qkv_hidden_sizes":
            qkv_hidden_sizes = list(attr.ints)
    if not num_heads or num_heads <= 0:
        return None

    if qkv_hidden_sizes is not None:
        if len(qkv_hidden_sizes) != 3:
            return None
        nq, nk, nv = qkv_hidden_sizes
    else:
        # Schema default: Q/K/V evenly split the merged width.
        if total_n % 3 != 0:
            return None
        nq = nk = nv = total_n // 3
    if (
        nq <= 0
        or nk <= 0
        or nv <= 0
        or nq + nk + nv != total_n
        or nq % num_heads
        or nk % num_heads
        or nv % num_heads
    ):
        return None

    return w_name, bias_name, num_heads, nq, nk, nv


def _reshape_last_dim(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[int]:
    """If `node` is a ``Reshape`` whose target-shape input is a constant
    int64 tensor, returns its last entry (or ``None`` if that entry is a
    wildcard/inferred ``-1`` or ``0``, or the shape can't be read at all).
    """
    if node.op_type != "Reshape" or len(node.input) != 2:
        return None
    shape_init = initializer_map.get(node.input[1])
    if shape_init is None or shape_init.data_type != onnx.TensorProto.INT64:
        return None
    dims = onnx.numpy_helper.to_array(shape_init)
    if dims.size == 0:
        return None
    last = int(dims[-1])
    return last if last > 0 else None


def _walk_to_attention_consumer(
    start: str,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    nv: int,
) -> Tuple[Optional[_ConsumerMatch], Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]]:
    """From `Attention`'s raw (V-hidden-size-wide) output tensor `start`,
    optionally through a single ``Reshape`` hop whose target shape's last
    entry is provably still `nv` (the shape onnxsim's own `fuse_attention`
    pass always appends, reusing the original ``ctx`` reshape's own target
    -- see fuse_attention.h's own doc comment; a hand-authored or
    differently-sourced graph is still handled the same way as long as it
    matches this same shape), to a MatMul/vanilla-Gemm consumer (the output
    projection) whose reduction dimension matches `nv`. Declines (``None``)
    on anything else -- a branch, an activation, a mismatched Reshape --
    rather than guessing. When a Reshape hop is matched, its second (shape)
    input must be single-use too -- the caller overwrites that constant's
    last entry to the post-pruning `nv` in place, which would corrupt any
    other reader of the same tensor.
    """
    candidates = consumers_of.get(start, [])
    if len(candidates) != 1:
        return None, ()
    node = candidates[0]
    chain_ops: Tuple[Tuple[onnx.NodeProto, Optional[str]], ...] = ()
    cur = start

    if node.op_type == "Reshape" and node.input[:1] == [cur]:
        last_dim = _reshape_last_dim(node, initializer_map)
        if last_dim != nv:
            return None, ()
        shape_name = node.input[1]
        if len(consumers_of.get(shape_name, [])) != 1:
            return None, ()  # shared shape constant -- mutating it isn't safe
        out_name = node.output[0]
        if len(consumers_of.get(out_name, [])) != 1 or out_name in graph_outputs:
            return None, ()
        chain_ops = ((node, shape_name),)
        cur = out_name
        node = consumers_of[cur][0]

    cm = _match_matmul_like(node)
    if cm is None or cm[0] != cur:
        return None, chain_ops
    _, cw_name, c_weight_transposed = cm
    cw_init = initializer_map.get(cw_name)
    if (
        cw_init is None
        or cw_init.data_type != onnx.TensorProto.FLOAT
        or len(cw_init.dims) != 2
    ):
        return None, chain_ops
    k = cw_init.dims[1] if c_weight_transposed else cw_init.dims[0]
    if k != nv:
        return None, chain_ops
    return (node, cw_name, c_weight_transposed), chain_ops


def _find_attention_chains(graph: onnx.GraphProto) -> List[_AttentionChain]:
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    graph_outputs = {o.name for o in graph.output}

    def _is_internal(name: str) -> bool:
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    chains = []
    for node in graph.node:
        info = _match_attention_producer(node, initializer_map)
        if info is None:
            continue
        w_name, bias_name, num_heads, nq, nk, nv = info

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue

        consumer, chain_ops = _walk_to_attention_consumer(
            out_name, initializer_map, consumers_of, graph_outputs, nv
        )
        if consumer is None:
            continue

        chains.append(
            _AttentionChain(
                node=node,
                weight=w_name,
                bias=bias_name,
                num_heads=num_heads,
                nq=nq,
                nk=nk,
                nv=nv,
                chain_ops=chain_ops,
                consumer_node=consumer[0],
                consumer_weight=consumer[1],
                consumer_weight_transposed=consumer[2],
            )
        )
    return chains


def _match_gqa_producer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[int, int]]:
    """If `node` is a ``com.microsoft::GroupQueryAttention`` node this
    module can safely act on, returns ``(num_heads, kv_num_heads)``.

    Requires: separate, non-empty query/key/value inputs (rules out the
    op's packed-QKV calling convention, where key/value are left empty and
    Q/K/V instead live concatenated in `query` -- a different tensor layout
    this function doesn't attempt to slice) and the `seqlens_k`/
    `total_sequence_length` inputs `GroupQueryAttention`'s schema requires
    even for a plain forward pass (both independent of head count, so never
    need touching themselves -- their presence is checked only as a sign
    this is a real, complete GQA node rather than a partially-constructed
    one); `num_heads`/`kv_num_heads` attributes with `num_heads` a positive
    multiple of `kv_num_heads`; and, for each of the optional `past_key`/
    `past_value` inputs (indices 3/4) that is actually connected, that it
    is *not* a non-empty constant -- such a tensor holds real per-KV-head
    cache data laid out along the `kv_num_heads` axis this function would
    need to slice but doesn't attempt to, so a node with one is declined
    outright rather than pruned incorrectly. A *dynamic* (non-constant)
    past_key/past_value -- an ordinary graph input or intermediate
    activation, not a weight -- is left alone and does not block the match:
    it is the caller's own runtime data, not something this rewrite could
    corrupt by leaving untouched. cos_cache/sin_cache (indices 7/8, for
    rotary position embedding), if present, are also always left alone:
    both are `[max_sequence_length, rotary_dim/2]`, broadcast identically
    across every head, so a head/group count change can never invalidate
    them.
    """
    if node.domain != _ATTENTION_DOMAIN or node.op_type != "GroupQueryAttention":
        return None
    if len(node.input) < 7 or not (node.input[0] and node.input[1] and node.input[2]):
        return None

    num_heads = kv_num_heads = None
    for attr in node.attribute:
        if attr.name == "num_heads":
            num_heads = attr.i
        elif attr.name == "kv_num_heads":
            kv_num_heads = attr.i
    if not num_heads or not kv_num_heads or num_heads <= 0 or kv_num_heads <= 0:
        return None
    if num_heads % kv_num_heads != 0:
        return None

    for idx in (3, 4):  # past_key, past_value
        if len(node.input) <= idx or not node.input[idx]:
            continue
        past_init = initializer_map.get(node.input[idx])
        if past_init is not None and int(np.prod(past_init.dims)) > 0:
            return None  # non-empty KV-cache constant -- would need slicing

    return num_heads, kv_num_heads


def _match_onnx_attention_producer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[int, int]]:
    """If `node` is a plain ``ai.onnx`` ``Attention`` node (domain ``""``,
    opset 24+ -- confirmed via ``onnx.defs.get_schema("Attention",
    domain="")`` against this environment's installed ``onnx==1.22.0``, see
    this module's own "Attention-head pruning" section comment) this module
    can safely act on, returns ``(q_num_heads, kv_num_heads)``.

    Structurally the closest cousin of :func:`_match_gqa_producer`: three
    separate, un-merged query/key/value inputs (``Q``, ``K``, ``V`` at
    indices 0/1/2, all required by the schema) rather than one merged
    weight, plus independent `q_num_heads`/`kv_num_heads` attributes with
    `q_num_heads` a positive multiple of `kv_num_heads` -- the same
    MHA/GQA/MQA taxonomy the op's own schema doc names explicitly. Both
    attributes are schema-*optional* (inferable from a rank-4 ``Q``/``K``
    input's own head axis, per ``onnx.reference.ops.op_attention``'s
    reference kernel), but this function requires both given explicitly:
    the topology this pass matches -- ``Q``/``K``/``V`` arriving directly
    from a MatMul/vanilla-Gemm projection's raw (rank-3,
    ``[batch, seq, hidden]``) output, the same shape
    :func:`_match_gqa_producer` already assumes for `GroupQueryAttention`
    -- is exactly the case the reference kernel itself asserts both
    attributes for, so a node relying on rank-4-inferred head counts isn't
    a shape this pass tracks and is declined rather than guessed at.

    The optional `attn_mask`/`past_key`/`past_value` inputs (indices 3/4/5
    -- a different set from `GroupQueryAttention`'s own, and this op has no
    `seqlens_k`/`total_sequence_length` equivalent to require) get the same
    treatment :func:`_match_gqa_producer` gives `past_key`/`past_value`:
    each is declined outright if it is connected to a non-empty constant
    (real per-head mask or KV-cache data this function doesn't attempt to
    slice), but left alone -- and does not block the match -- if dynamic
    (an ordinary graph input or intermediate activation, the caller's own
    runtime data). `nonpad_kv_seqlen` (index 6), like `GroupQueryAttention`'s
    own `seqlens_k`, is `[batch_size]`-shaped and independent of head count,
    so its presence never blocks a match either.
    """
    if node.domain != "" or node.op_type != "Attention":
        return None
    if len(node.input) < 3 or not (node.input[0] and node.input[1] and node.input[2]):
        return None

    q_num_heads = kv_num_heads = None
    for attr in node.attribute:
        if attr.name == "q_num_heads":
            q_num_heads = attr.i
        elif attr.name == "kv_num_heads":
            kv_num_heads = attr.i
    if not q_num_heads or not kv_num_heads or q_num_heads <= 0 or kv_num_heads <= 0:
        return None
    if q_num_heads % kv_num_heads != 0:
        return None

    for idx in (3, 4, 5):  # attn_mask, past_key, past_value
        if len(node.input) <= idx or not node.input[idx]:
            continue
        const_init = initializer_map.get(node.input[idx])
        if const_init is not None and int(np.prod(const_init.dims)) > 0:
            return None  # non-empty constant -- would need slicing

    return q_num_heads, kv_num_heads


def _find_separate_qkv_chains(
    graph: onnx.GraphProto,
    match_producer,
    num_heads_attr: str,
) -> List[_GQAChain]:
    """Shared body for :func:`_find_gqa_chains` and
    :func:`_find_onnx_attention_chains`: both match a fused attention node
    fed by three separate, un-merged Q/K/V MatMul/vanilla-Gemm projections
    (as opposed to :func:`_find_attention_chains`'s single merged-QKV-weight
    ``com.microsoft::Attention``) and prune it at whole-KV-group granularity
    (see :func:`_apply_one_gqa_chain`/:func:`_gqa_group_importance`),
    differing only in which node/attributes `match_producer` recognizes
    (:func:`_match_gqa_producer` or :func:`_match_onnx_attention_producer`)
    and which attribute on the matched node holds the query head count
    (`num_heads_attr` -- see :class:`_GQAChain`'s own field of that name).
    """
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    graph_outputs = {o.name for o in graph.output}
    node_by_output = {out: node for node in graph.node for out in node.output}

    def _is_internal(name: str) -> bool:
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    chains = []
    for node in graph.node:
        info = match_producer(node, initializer_map)
        if info is None:
            continue
        num_heads, kv_num_heads = info

        q_name, k_name, v_name = node.input[0], node.input[1], node.input[2]
        if q_name == k_name or q_name == v_name or k_name == v_name:
            continue  # degenerate -- can't independently slice a shared producer

        producer_infos = []
        matched = True
        for in_name in (q_name, k_name, v_name):
            if not _is_internal(in_name):
                matched = False
                break
            prod_node = node_by_output.get(in_name)
            if prod_node is None:
                matched = False
                break
            pinfo = _match_producer(prod_node, initializer_map)
            if pinfo is None:
                matched = False
                break
            producer_infos.append(pinfo)
        if not matched:
            continue

        (wq, wq_t, bq, nq), (wk, wk_t, bk, nk), (wv, wv_t, bv, nv) = producer_infos
        if (
            wq == wk
            or wq == wv
            or wk == wv
            or nq % num_heads
            or nk % kv_num_heads
            or nv % kv_num_heads
        ):
            continue
        head_size = nq // num_heads
        if (
            head_size <= 0
            or nk // kv_num_heads != head_size
            or nv // kv_num_heads != head_size
        ):
            # `fuse_gqa.h` requires equal Q/K/V head_size, and this is the
            # same restriction :func:`_match_onnx_attention_producer`'s own
            # docstring narrows the plain ai.onnx op's own, more permissive
            # schema down to (that op genuinely allows V its own head_size,
            # see this module's "Attention-head pruning" section comment) --
            # a node whose V head size actually differs is declined here
            # rather than mis-sliced by this shared, uniform-head_size body.
            continue

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue

        # Both matched ops' raw output is Nq (= num_heads * head_size) wide
        # -- unlike plain `com.microsoft::Attention` (V-hidden-size-wide),
        # each is head-count-and-size symmetric with the query side (per
        # fuse_gqa.h's own "Y = GroupQueryAttention(...)" shape comment, and
        # -- since this function only ever matches the equal-head_size case
        # above -- the ai.onnx op's own "q_num_heads * head_size_v" output
        # width, see its reference kernel) -- so
        # `_walk_to_attention_consumer`'s generic "raw output width"
        # parameter is passed `nq` here, not an analogue of `nv`.
        consumer, chain_ops = _walk_to_attention_consumer(
            out_name, initializer_map, consumers_of, graph_outputs, nq
        )
        if consumer is None:
            continue

        chains.append(
            _GQAChain(
                node=node,
                q_weight=wq,
                q_bias=bq,
                q_weight_transposed=wq_t,
                k_weight=wk,
                k_bias=bk,
                k_weight_transposed=wk_t,
                v_weight=wv,
                v_bias=bv,
                v_weight_transposed=wv_t,
                num_heads=num_heads,
                kv_num_heads=kv_num_heads,
                head_size=head_size,
                chain_ops=chain_ops,
                consumer_node=consumer[0],
                consumer_weight=consumer[1],
                consumer_weight_transposed=consumer[2],
                num_heads_attr=num_heads_attr,
            )
        )
    return chains


def _find_gqa_chains(graph: onnx.GraphProto) -> List[_GQAChain]:
    return _find_separate_qkv_chains(graph, _match_gqa_producer, "num_heads")


def _find_onnx_attention_chains(graph: onnx.GraphProto) -> List[_GQAChain]:
    """The plain ``ai.onnx::Attention`` analogue of :func:`_find_gqa_chains`
    -- see :func:`_find_separate_qkv_chains` (the shared body) and
    :func:`_match_onnx_attention_producer` (what's matched and why it's
    declined otherwise).
    """
    return _find_separate_qkv_chains(
        graph, _match_onnx_attention_producer, "q_num_heads"
    )


def _plain_attention_head_importance(
    chain: _AttentionChain,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    dq: int,
    dk: int,
    dv: int,
) -> np.ndarray:
    # Combined (Frobenius-norm) importance of each head's full Q+K+V
    # weight block -- the Li et al. filter-norm criterion this module uses
    # everywhere else, applied to a whole head's block of columns (across
    # every input row) at once instead of a single output channel/filter.
    importance = np.zeros(chain.num_heads, dtype=np.float64)
    for h in range(chain.num_heads):
        block = np.concatenate(
            [
                wq[:, h * dq : (h + 1) * dq],
                wk[:, h * dk : (h + 1) * dk],
                wv[:, h * dv : (h + 1) * dv],
            ],
            axis=1,
        )
        importance[h] = np.linalg.norm(block)
    return importance


def _head_column_indices(keep_heads: np.ndarray, head_size: int) -> np.ndarray:
    return np.concatenate(
        [np.arange(h * head_size, (h + 1) * head_size) for h in keep_heads]
    )


def _gqa_group_importance(
    chain: _GQAChain, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray
) -> np.ndarray:
    # Combined (Frobenius-norm) importance of each *KV group's* whole
    # block: the group's own K/V head columns plus every query head mapped
    # to it -- the GQA analogue of :func:`_plain_attention_head_importance`,
    # at group instead of individual-head granularity, since a lone query
    # head can't be pruned out from under a shared KV head in isolation
    # (see this module's own "Attention-head pruning" section comment).
    d = chain.head_size
    group_size = chain.num_heads // chain.kv_num_heads
    importance = np.zeros(chain.kv_num_heads, dtype=np.float64)
    for kv in range(chain.kv_num_heads):
        q_block = np.concatenate(
            [
                wq[:, h * d : (h + 1) * d]
                for h in range(kv * group_size, (kv + 1) * group_size)
            ],
            axis=1,
        )
        k_block = wk[:, kv * d : (kv + 1) * d]
        v_block = wv[:, kv * d : (kv + 1) * d]
        importance[kv] = np.linalg.norm(
            np.concatenate([q_block, k_block, v_block], axis=1)
        )
    return importance


def _apply_one_plain_attention_chain(
    initializer_map: Dict[str, onnx.TensorProto],
    chain: _AttentionChain,
    sparsity: float,
    compute_importance,
) -> Optional[Tuple[Set[str], str, Set[str]]]:
    """Applies whole-head pruning to one matched ``Attention`` block in
    place: every dropped head removes a *contiguous* ``head_size``-wide
    column block from the single merged QKV weight (and the matching row
    block from the consumer), not an arbitrary top-k column subset. Returns
    ``(producer_weight_names, consumer_weight_name, stale_output_names)`` on
    success, or ``None`` if `sparsity` rounds to no heads dropped for this
    block (a no-op, left for the caller to skip).
    """
    h = chain.num_heads
    keep_count = max(1, h - round(h * sparsity))
    if keep_count >= h:
        return None

    dq, dk, dv = chain.nq // h, chain.nk // h, chain.nv // h
    w_init = initializer_map[chain.weight]
    w = onnx.numpy_helper.to_array(w_init).astype(np.float64)  # [K, Nq+Nk+Nv]
    wq = w[:, : chain.nq]
    wk = w[:, chain.nq : chain.nq + chain.nk]
    wv = w[:, chain.nq + chain.nk :]

    importance = compute_importance(chain, wq, wk, wv, dq, dk, dv)
    keep_heads = np.sort(np.argsort(-importance)[:keep_count])

    q_idx = _head_column_indices(keep_heads, dq)
    k_idx = _head_column_indices(keep_heads, dk) + chain.nq
    v_idx_local = _head_column_indices(keep_heads, dv)
    v_idx = v_idx_local + chain.nq + chain.nk
    all_idx = np.concatenate([q_idx, k_idx, v_idx])

    w_arr = onnx.numpy_helper.to_array(w_init)
    w_init.CopyFrom(onnx.numpy_helper.from_array(w_arr[:, all_idx], name=w_init.name))
    if chain.bias is not None:
        _slice_last_axis(initializer_map[chain.bias], all_idx)

    found_qkv = False
    for attr in chain.node.attribute:
        if attr.name == "num_heads":
            attr.i = keep_count
        elif attr.name == "qkv_hidden_sizes":
            found_qkv = True
            del attr.ints[:]
            attr.ints.extend([keep_count * dq, keep_count * dk, keep_count * dv])
    if not found_qkv:
        chain.node.attribute.append(
            onnx.helper.make_attribute(
                "qkv_hidden_sizes",
                [keep_count * dq, keep_count * dk, keep_count * dv],
            )
        )

    _slice_consumer_weight(
        initializer_map[chain.consumer_weight],
        chain.consumer_weight_transposed,
        v_idx_local,
    )

    for _, shape_name in chain.chain_ops:
        if shape_name is not None:
            shape_init = initializer_map[shape_name]
            dims = onnx.numpy_helper.to_array(shape_init).copy()
            dims[-1] = keep_count * dv
            shape_init.CopyFrom(
                onnx.numpy_helper.from_array(dims, name=shape_init.name)
            )

    stale = {chain.node.output[0]}
    stale.update(op.output[0] for op, _ in chain.chain_ops)
    return {chain.weight}, chain.consumer_weight, stale


def _apply_one_gqa_chain(
    initializer_map: Dict[str, onnx.TensorProto],
    chain: _GQAChain,
    sparsity: float,
    compute_group_importance,
) -> Optional[Tuple[Set[str], str, Set[str]]]:
    """Applies whole-KV-group pruning to one matched ``GroupQueryAttention``
    or plain ``ai.onnx::Attention`` block (see :class:`_GQAChain`'s own
    `num_heads_attr` field for how the two are told apart when writing the
    new query head count back) in place: every dropped group removes one
    *contiguous* ``head_size``-wide column block from each of K's and V's
    own separate weights, together with the ``num_heads / kv_num_heads``
    query-head-sized blocks mapped to that group from Q's own separate
    weight (and the matching row block from the consumer) -- never an
    individual query head in isolation. Returns
    ``(producer_weight_names, consumer_weight_name, stale_output_names)`` on
    success, or ``None`` if `sparsity` rounds to no groups dropped for this
    block (a no-op, left for the caller to skip).
    """
    h = chain.kv_num_heads
    keep_count = max(1, h - round(h * sparsity))
    if keep_count >= h:
        return None

    d = chain.head_size
    group_size = chain.num_heads // chain.kv_num_heads

    # `_gqa_group_importance` (and any caller-supplied Wanda variant of it)
    # indexes columns as the head axis, mirroring
    # `_apply_one_plain_attention_chain`'s own `wq`/`wk`/`wv` convention --
    # so each array is brought to ``[K, N]`` (reduction dim first, head
    # columns last), the *opposite* of `_prune_weight`'s "output channel
    # first" `[N, K]` convention used elsewhere in this module: only
    # transpose when the raw storage is already `[N, K]` (Gemm transB=1).
    wq_init = initializer_map[chain.q_weight]
    wk_init = initializer_map[chain.k_weight]
    wv_init = initializer_map[chain.v_weight]
    wq_kn = onnx.numpy_helper.to_array(wq_init).astype(np.float64)
    wk_kn = onnx.numpy_helper.to_array(wk_init).astype(np.float64)
    wv_kn = onnx.numpy_helper.to_array(wv_init).astype(np.float64)
    if chain.q_weight_transposed:
        wq_kn = wq_kn.T  # [K, Nq]
    if chain.k_weight_transposed:
        wk_kn = wk_kn.T  # [K, Nkv]
    if chain.v_weight_transposed:
        wv_kn = wv_kn.T  # [K, Nkv]

    importance = compute_group_importance(chain, wq_kn, wk_kn, wv_kn)
    keep_groups = np.sort(np.argsort(-importance)[:keep_count])

    keep_q_heads = np.concatenate(
        [np.arange(g * group_size, (g + 1) * group_size) for g in keep_groups]
    )
    q_idx = _head_column_indices(keep_q_heads, d)
    kv_idx = _head_column_indices(keep_groups, d)

    _slice_producer_weight(wq_init, chain.q_weight_transposed, q_idx)
    _slice_producer_weight(wk_init, chain.k_weight_transposed, kv_idx)
    _slice_producer_weight(wv_init, chain.v_weight_transposed, kv_idx)
    if chain.q_bias is not None:
        _slice_last_axis(initializer_map[chain.q_bias], q_idx)
    if chain.k_bias is not None:
        _slice_last_axis(initializer_map[chain.k_bias], kv_idx)
    if chain.v_bias is not None:
        _slice_last_axis(initializer_map[chain.v_bias], kv_idx)

    new_kv_num_heads = keep_count
    new_num_heads = keep_count * group_size
    for attr in chain.node.attribute:
        if attr.name == chain.num_heads_attr:
            attr.i = new_num_heads
        elif attr.name == "kv_num_heads":
            attr.i = new_kv_num_heads

    _slice_consumer_weight(
        initializer_map[chain.consumer_weight],
        chain.consumer_weight_transposed,
        q_idx,
    )

    for _, shape_name in chain.chain_ops:
        if shape_name is not None:
            shape_init = initializer_map[shape_name]
            dims = onnx.numpy_helper.to_array(shape_init).copy()
            dims[-1] = new_num_heads * d
            shape_init.CopyFrom(
                onnx.numpy_helper.from_array(dims, name=shape_init.name)
            )

    stale = {chain.node.output[0]}
    stale.update(op.output[0] for op, _ in chain.chain_ops)
    return (
        {chain.q_weight, chain.k_weight, chain.v_weight},
        chain.consumer_weight,
        stale,
    )


def _apply_attention_chains(
    graph: onnx.GraphProto,
    chains: List[_AttnLikeChain],
    sparsity: float,
    compute_importance,
    compute_group_importance,
) -> None:
    """Shared body for :func:`apply_attention_head_pruning` and
    :func:`apply_attention_head_wanda_pruning`, mirroring
    :func:`_apply_chains`'s own shape (cross-chain touched-role
    bookkeeping, stale ``value_info`` cleanup) but dispatching each chain to
    :func:`_apply_one_plain_attention_chain` (a matched ``Attention``
    block) or :func:`_apply_one_gqa_chain` (a matched
    ``GroupQueryAttention`` block) for the actual per-chain slicing, since
    the two ops' weight layouts (one merged QKV tensor vs. three separate
    Q/K/V producers) and pruning unit (individual head vs. whole KV group)
    are different enough that sharing that part wouldn't simplify either.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    producer_touched: Set[str] = set()
    consumer_touched: Set[str] = set()
    stale_value_info: Set[str] = set()

    for chain in chains:
        if isinstance(chain, _GQAChain):
            producer_names = {chain.q_weight, chain.k_weight, chain.v_weight}
        else:
            producer_names = {chain.weight}
        if (
            producer_names & producer_touched
            or chain.consumer_weight in consumer_touched
        ):
            continue

        applied: Optional[Tuple[Set[str], str, Set[str]]]
        if isinstance(chain, _GQAChain):
            applied = _apply_one_gqa_chain(
                initializer_map, chain, sparsity, compute_group_importance
            )
        else:
            applied = _apply_one_plain_attention_chain(
                initializer_map, chain, sparsity, compute_importance
            )
        if applied is None:
            continue

        touched_producers, touched_consumer, stale = applied
        producer_touched.update(touched_producers)
        consumer_touched.add(touched_consumer)
        stale_value_info.update(stale)

    if stale_value_info:
        kept = [vi for vi in graph.value_info if vi.name not in stale_value_info]
        del graph.value_info[:]
        graph.value_info.extend(kept)


def apply_attention_head_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
) -> onnx.ModelProto:
    """Removes whole attention heads -- or, for grouped-query attention,
    whole KV groups -- from every matched ``com.microsoft::Attention``,
    ``com.microsoft::GroupQueryAttention``, or plain ``ai.onnx::Attention``
    node (the fused self-attention blocks onnxsim's own
    ``fuse_attention``/``fuse_gqa`` optimizer passes produce, plus the
    standard ONNX op those two contrib ops are converging towards -- see
    this module's own "Attention-head pruning" section comment for the real
    schema each was confirmed against and how) whose output feeds,
    optionally through a single shape-preserving ``Reshape``, exactly one
    downstream MatMul/vanilla-Gemm's reduction dimension (the output
    projection) -- the attention analogue of :func:`apply_structured_pruning`,
    at head (or KV-group) instead of single-channel granularity.

    For each matched plain ``com.microsoft::Attention`` block: ranks every
    head by the combined Frobenius norm of its own
    ``[hidden_size, head_size]`` Q, K, and V weight columns, drops the
    lowest-``sparsity``-fraction of heads (at least one head is always
    kept), and removes the corresponding column blocks from the merged QKV
    weight (and bias, if present), decrementing
    ``num_heads``/``qkv_hidden_sizes`` accordingly, and the matching row
    block from the output projection's weight -- mathematically unaffected
    for every surviving head, the same guarantee
    :func:`apply_structured_pruning` gives per channel.

    For each matched ``GroupQueryAttention`` or plain ``ai.onnx::Attention``
    block: ranks every *KV group* (a KV head and the
    ``num_heads / kv_num_heads`` query heads the kernel maps to it) by the
    combined Frobenius norm of that group's own Q+K+V weight block across
    Q's, K's, and V's own separate producer weights, drops the
    lowest-``sparsity``-fraction of groups (at least one group is always
    kept), and removes the corresponding column blocks from all three
    producers (and their biases, if present) together with the matching row
    block from the output projection's weight, decrementing the query head
    count (``num_heads`` for `GroupQueryAttention`, ``q_num_heads`` for the
    plain ``ai.onnx`` op) and ``kv_num_heads`` by the number of groups
    dropped -- so their ratio (query heads per KV head) is unchanged,
    keeping every surviving KV head mapped to exactly the same number of
    query heads the kernel requires. An individual query head is never
    dropped on its own: only a whole group, since neither kernel has a way
    to keep a KV head alive for some, but not all, of the query heads that
    shared it. A plain ``ai.onnx::Attention`` node whose V head size
    genuinely differs from Q/K's own (a shape that op's schema allows but
    `GroupQueryAttention` never produces) is declined rather than mis-sliced
    -- see :func:`_match_onnx_attention_producer`.

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of each matched block's heads (or, for
            GroupQueryAttention/plain ai.onnx Attention, KV groups) to
            remove (at least one is always kept)
    :returns: ``model`` with every matched block's tensors resized in
            place; anything not matching that exact topology (a
            non-constant weight, a packed-QKV GroupQueryAttention node, a
            GroupQueryAttention/plain ai.onnx Attention node with a
            non-empty constant past-KV-cache or attention-mask input, an
            ai.onnx Attention node with differing Q/K/V head sizes or
            without explicit ``q_num_heads``/``kv_num_heads`` attributes, a
            consumer whose reduction dimension doesn't line up, ...) is left
            completely untouched
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains: List[_AttnLikeChain] = [
        *_find_attention_chains(graph),
        *_find_gqa_chains(graph),
        *_find_onnx_attention_chains(graph),
    ]
    if chains:
        _apply_attention_chains(
            graph,
            chains,
            sparsity,
            _plain_attention_head_importance,
            _gqa_group_importance,
        )

    return out


def apply_attention_head_wanda_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    epsilon: float = 1e-8,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """The calibrated upgrade of :func:`apply_attention_head_pruning`,
    exactly as :func:`apply_structured_wanda_pruning` is to
    :func:`apply_structured_pruning`: same real head (or, for
    GroupQueryAttention/plain ai.onnx Attention, whole-KV-group) removal,
    same topology matching (see :func:`apply_attention_head_pruning`'s own
    docstring for the three matched op types), but each unit's importance
    is ``||W||_F * ||X||_2`` -- the plain Frobenius-norm weight score times
    the combined (root-sum-square) activation norm of that unit's own slice
    of the *output projection's* input, captured over calibration data --
    instead of weight magnitude alone. For a plain ``com.microsoft::Attention``
    block this is per head, exactly as before; for a ``GroupQueryAttention``
    or plain ``ai.onnx::Attention`` block the activation norm is combined
    (root-sum-square) over every query head a KV group owns, mirroring how
    :func:`_gqa_group_importance` combines that same group's weight norm
    across Q+K+V -- both matched separate-Q/K/V-producer ops share that one
    importance function (and this one calibrated wrapper around it)
    unmodified.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            block's output-projection-side activation norm on. Each batch
            is a ``{input_name: np.ndarray}`` dict matching ``model``'s
            graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of each matched block's heads (or, for
            GroupQueryAttention/plain ai.onnx Attention, KV groups) to
            remove (at least one is always kept)
    :param epsilon: floor applied to the accumulated per-unit activation
            norm, avoiding every unit of an all-zero activation tying at
            exactly the weight-only importance
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched block's tensors resized in
            place; anything not matching that exact topology falls back to
            :func:`apply_attention_head_pruning`'s plain Frobenius-norm
            ranking if no matching activation was ever observed for that
            block's consumer
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains: List[_AttnLikeChain] = [
        *_find_attention_chains(graph),
        *_find_gqa_chains(graph),
        *_find_onnx_attention_chains(graph),
    ]
    if not chains:
        return out

    probe_names = sorted({chain.consumer_node.input[0] for chain in chains})
    probe_model = _add_probe_outputs(out, probe_names)

    sq_sum: Dict[str, np.ndarray] = {}
    count: Dict[str, int] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            if x.ndim < 1:
                continue
            reduce_axes = tuple(range(x.ndim - 1))
            s = np.square(x).sum(axis=reduce_axes) if reduce_axes else np.square(x)
            cnt = int(np.prod(x.shape[:-1], dtype=np.int64)) if x.ndim > 1 else 1
            sq_sum[name] = s if name not in sq_sum else sq_sum[name] + s
            count[name] = count.get(name, 0) + cnt

    act_norm: Dict[str, np.ndarray] = {
        name: np.sqrt(s / max(count[name], 1)) for name, s in sq_sum.items()
    }

    def _wanda_attention_head_importance(chain, wq, wk, wv, dq, dk, dv):
        base = _plain_attention_head_importance(chain, wq, wk, wv, dq, dk, dv)
        norm = act_norm.get(chain.consumer_node.input[0])
        if norm is None or norm.shape[0] != chain.nv:
            return base  # no matching activation observed -- fall back to plain
        act_head = np.array(
            [
                np.linalg.norm(norm[h * dv : (h + 1) * dv])
                for h in range(chain.num_heads)
            ]
        )
        return base * np.maximum(act_head, epsilon)

    def _wanda_gqa_group_importance(chain, wq, wk, wv):
        base = _gqa_group_importance(chain, wq, wk, wv)
        norm = act_norm.get(chain.consumer_node.input[0])
        width = chain.num_heads * chain.head_size
        if norm is None or norm.shape[0] != width:
            return base  # no matching activation observed -- fall back to plain
        d = chain.head_size
        group_size = chain.num_heads // chain.kv_num_heads
        act_group = np.array(
            [
                np.linalg.norm(norm[kv * group_size * d : (kv + 1) * group_size * d])
                for kv in range(chain.kv_num_heads)
            ]
        )
        return base * np.maximum(act_group, epsilon)

    _apply_attention_chains(
        graph,
        chains,
        sparsity,
        _wanda_attention_head_importance,
        _wanda_gqa_group_importance,
    )
    return out
