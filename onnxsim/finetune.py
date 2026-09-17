"""Layer-wise MatMul/Gemm weight fine-tuning that recovers accuracy lost to
*structured* (channel) or *attention-head* pruning -- the pruning analogue
of :mod:`onnxsim.adaround`'s own "recover accuracy lost to quantization"
role, reusing the same established idiom this codebase already uses there
(and in :mod:`onnxsim.bias_correction`): real activations captured from a
reference model, a closed-form fit against them, no generic autodiff or
backprop-through-the-graph framework of any kind.

:func:`apply_pruning_finetune` takes the *pre-pruning* float model
(``original_model``) and its pruned counterpart (``pruned_model``, e.g. the
output of :func:`onnxsim.apply_structured_pruning_cpp`/
:func:`onnxsim.apply_attention_head_pruning_cpp` or their Wanda-calibrated
counterparts) and, for every MatMul/Gemm node present (by node output name
and op type) in both, re-derives that layer's own surviving output/input
channel indices and re-solves its weight (and bias, if it has one) as a
ridge-regression fit against ``original_model``'s own real activations --
not just re-slicing the original weight (what pruning itself already did),
but choosing the *best* surviving weight values to reconstruct what the
*original*, unpruned layer would have produced, over the channels that
survive.

**Why this is exact, not approximate, for a single pass.** Pruning is a
pure "delete some rows/columns of a weight tensor" operation -- it changes
*which* channels flow through the graph, but never the *value* of a
channel that survives, and every hop in between (a unary activation, a
per-channel bias/scale) is itself channel-order-preserving. So, before any
fine-tuning happens, a pruned layer's own real input activation is
*exactly* equal to ``original_model``'s own activation at that same tensor,
restricted to whichever input channels this layer's own pruning kept --
no need to actually run ``pruned_model`` at all to know what it would
produce there. This is the same one-shot-capture design
:func:`onnxsim.apply_adaround` already uses (it never re-runs
``quantized_model`` between layers either): every layer's own optimization
target is computed once, up front, from ``original_model`` alone.

**Channel correspondence.** Pruning's own ``keep`` index set (see
``onnxsim/pruning.py``'s own ``_apply_chains``) is never exposed by the
pruning functions themselves, so this module reconstructs it by exact
(bit-identical) matching: a pruned weight's own surviving rows (or columns)
are, by construction, an order-preserving subsequence of the original
weight's own rows (or columns) -- a plain ``np.sort``-then-gather, no
reordering, no value change -- so a forward two-pointer scan finds exactly
which original index each surviving one came from
(:func:`_find_keep_indices`). Comparing a *whole* row (or column) this way
needs the *other* axis's own width to already agree between the two
weights, though, so this only recovers the correspondence for a layer
pruned on at most one axis at a time -- its own output channels (this
layer is itself some producer chain's own consumer, or the primary
producer of one), its own input channels (this layer is itself some
chain's own consumer), or neither (still worth fine-tuning against shifted
upstream activations, but only once weight equality confirms nothing else
changed it) -- see :func:`_find_channel_correspondence`. A layer pruned on
*both* axes at once (an interior layer that is itself both some chain's
own consumer and a different chain's own producer), or one whose weight
isn't a clean subsequence of its own original counterpart at all (this
module was pointed at two unrelated models, or something other than
pruning touched the weight in between), is left completely untouched,
never guessed at -- the same conservative boundary every ``pruning.py``
chain finder already holds.

**The fit itself.** For one layer, let ``X`` be ``original_model``'s own
captured input activation restricted to this layer's own surviving input
channels, and ``Y`` be ``original_model``'s own analytically-computed
output (``X_full @ W_orig.T [+ bias_orig]``) restricted to this layer's own
surviving output channels -- both computed directly from one probed
activation and the original weight/bias, mirroring
:func:`onnxsim.apply_adaround`'s own single-probe-point-per-layer design.
The new weight (and bias, if present) minimizes
``||X @ W.T [+ b] - Y||^2 + reg_param * ||[W, b] - [W_pruned, b_pruned]||^2``
-- ordinary ridge regression, ``reg_param`` (relative to the calibration
data's own scale) pulling the fit back toward pruning's own already-sliced
weight when calibration data is too scarce, relative to the layer's own
parameter count, to trust unregularized least squares. Solved in closed
form (a single linear solve per layer) rather than any iterative
optimizer -- unlike AdaRound's own non-convex rounding relaxation, this is
an ordinary (convex) least-squares problem with no relaxation needed.

**Scope.** MatMul/vanilla-Gemm only, matching
:func:`onnxsim.apply_adaround`'s own scope exactly -- no Conv support (a
Conv layer's own im2col expansion, and the padding/stride/dilation
bookkeeping ``pruning.py``'s own Wanda-calibrated Conv path already needs,
is real extra work this module doesn't yet do). A captured activation with
no feature axis at all (rank < 2) is skipped; a higher-rank
``[batch, seq, K]`` one is flattened to ``[batch * seq, K]`` -- exact, as
the fit sums over the same rows either way -- the same boundary
:func:`onnxsim.apply_adaround` already has.
This also does not, itself, touch attention-head-pruned Q/K/V/output-
projection weights that live inside a fused ``com.microsoft::Attention``/
``GroupQueryAttention``/``ai.onnx::Attention`` node (no MatMul/Gemm node
exists for a packed-QKV weight at all) or that feed such a fused op as a
consumer (this module only matches a MatMul/Gemm *consumer*) -- covered
only when attention-head pruning happens to leave ordinary MatMul/Gemm
layers around it (e.g. an unrelated MLP block) unchanged in shape, which
this module still usefully re-fits against the shifted upstream
activations flowing into it. And, per this module's own "Channel
correspondence" note above, a layer pruned on *both* its own input and
output channels at once is left untouched too -- still a real gap for a
model with several matched layers chained back to back, narrower than
what ``apply_structured_pruning``'s own chain finders themselves recognize.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import onnx

from onnxsim.calibration import Tensors
from onnxsim.onnx_simplifier import apply_pruning_finetune_cpp


def _find_keep_indices(
    orig_rows: np.ndarray, pruned_rows: np.ndarray
) -> Optional[np.ndarray]:
    """Returns the ``len(pruned_rows)``-length array of ``orig_rows``'s own
    indices ``pruned_rows`` was sliced from, reconstructed by exact
    (bit-identical) forward two-pointer subsequence matching -- see this
    module's own docstring for why pruning's own slice always makes
    ``pruned_rows`` an order-preserving subsequence of ``orig_rows``.
    Returns ``None`` if it isn't one (declined conservatively).
    """
    if (
        pruned_rows.shape[0] > orig_rows.shape[0]
        or orig_rows.shape[1:] != pruned_rows.shape[1:]
    ):
        return None
    keep = np.empty(pruned_rows.shape[0], dtype=np.int64)
    oi = 0
    n_orig = orig_rows.shape[0]
    for pi in range(pruned_rows.shape[0]):
        while oi < n_orig and not np.array_equal(orig_rows[oi], pruned_rows[pi]):
            oi += 1
        if oi >= n_orig:
            return None
        keep[pi] = oi
        oi += 1
    return keep


def _find_channel_correspondence(
    w_orig_nk: np.ndarray, w_pruned_nk: np.ndarray
) -> "tuple[Optional[np.ndarray], Optional[np.ndarray]]":
    """Finds ``(keep_out, keep_in)`` for one layer's ``[N, K]``-normalized
    weight pair. Row/column matching alone can only recover one axis's own
    ``keep`` set at a time -- comparing a *full* row (or column) requires
    the *other* axis's own width to already agree between the two matrices
    -- so this only handles a layer pruned on at most one axis: its own
    output channels (a producer chain's own pruning), its own input
    channels (a consumer chain's own pruning), or neither (still worth
    fine-tuning against shifted upstream activations, but only if the two
    weights are otherwise identical -- see :func:`_find_keep_indices`'s own
    same-length-subsequence-forces-equality argument). A layer pruned on
    *both* axes at once (an interior layer that is itself both some
    producer chain's own consumer and some other chain's own producer) is
    declined outright -- recovering two simultaneous, independent ``keep``
    sets from the pruned matrix's own content alone is a real ambiguous
    inverse problem this module doesn't attempt to solve, rather than
    risk a wrong (silently corrupting) channel correspondence.
    """
    n_orig, k_orig = w_orig_nk.shape
    n_pruned, k_pruned = w_pruned_nk.shape
    if k_orig == k_pruned:
        keep_out = _find_keep_indices(w_orig_nk, w_pruned_nk)
        keep_in = np.arange(k_orig) if keep_out is not None else None
        return keep_out, keep_in
    if n_orig == n_pruned:
        keep_in = _find_keep_indices(w_orig_nk.T, w_pruned_nk.T)
        keep_out = np.arange(n_orig) if keep_in is not None else None
        return keep_out, keep_in
    return (
        None,
        None,
    )  # both axes changed -- declined, see this function's own docstring


def _ridge_fit(
    x: np.ndarray,
    y: np.ndarray,
    w0: np.ndarray,
    b0: Optional[np.ndarray],
    reg_param: float,
) -> "tuple[np.ndarray, Optional[np.ndarray]]":
    """Ridge-regression fit of ``y ~= x @ w.T [+ b]`` (``x``: ``[num_samples,
    K]``, ``y``: ``[num_samples, N]``, ``w0``/``b0``: the current ``[N, K]``/
    ``[N]`` weight/bias this fit is regularized toward), ``reg_param``
    scaled relative to ``x``'s own Gram matrix so it needs no per-model
    tuning. Returns the new ``(w, b)`` (``b`` is ``None`` iff ``b0`` is).
    """
    if b0 is not None:
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        w0_aug = np.concatenate([w0, b0[:, None]], axis=1)
    else:
        x_aug = x
        w0_aug = w0

    gram = x_aug.T @ x_aug
    lam = reg_param * (np.trace(gram) / gram.shape[0])
    a = gram + lam * np.eye(gram.shape[0])
    rhs = x_aug.T @ y + lam * w0_aug.T
    w_aug_t = np.linalg.solve(a, rhs)  # [K(+1), N]

    if b0 is not None:
        return w_aug_t[:-1, :].T, w_aug_t[-1, :]
    return w_aug_t.T, None


def apply_pruning_finetune(
    original_model: Union[str, onnx.ModelProto],
    pruned_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    reg_param: float = 1e-2,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Fine-tunes every surviving MatMul/vanilla-Gemm layer present (by node
    output name) in both ``original_model`` and ``pruned_model`` to better
    reconstruct ``original_model``'s own real activations over calibration
    data -- see this module's own docstring for the full technique, its
    exactness argument, and its scope boundaries.

    :param original_model: the pre-pruning onnx ModelProto or file path
    :param pruned_model: ``original_model`` after
            :func:`onnxsim.apply_structured_pruning_cpp`/
            :func:`onnxsim.apply_attention_head_pruning_cpp` (or their
            Wanda-calibrated counterparts) -- onnx ModelProto or file path.
            Assumes no MatMul/Gemm node's own output tensor was renamed
            between the two, true of every onnxsim pruning function
    :param calibration_data: representative input batches to fit each
            layer's own reconstruction on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``original_model``'s
            graph inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param reg_param: ridge-regression regularization strength, relative to
            the calibration data's own scale (see this module's own
            docstring), pulling each layer's fit back toward its current
            (already-pruned) weight when calibration data is too scarce,
            relative to the layer's own parameter count, to trust
            unregularized least squares
    :param providers: onnxruntime execution providers to run
            ``original_model`` on when capturing calibration activations
    :returns: ``pruned_model`` with every matched layer's weight (and bias,
            if it has one) replaced by its fine-tuned fit; a layer whose
            weight isn't a clean row/column subsequence of its own
            ``original_model`` counterpart, or whose captured activation
            has no feature axis at all (rank < 2), is left completely
            untouched; a higher-rank ``[batch, seq, K]`` activation is
            flattened to ``[batch * seq, K]``, which is exact

    Delegates to :func:`onnxsim.apply_pruning_finetune_cpp` (the verified
    C++ port), which solves the exact same closed-form ridge-regression
    fit this function's own docstring describes. This pure-Python name is
    kept only for backward compatibility with existing callers.
    """
    if isinstance(original_model, str):
        original_model = onnx.load(original_model, load_external_data=False)
    if isinstance(pruned_model, str):
        pruned_model = onnx.load(pruned_model, load_external_data=False)
    return apply_pruning_finetune_cpp(
        original_model,
        pruned_model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        reg_param=reg_param,
        providers=providers,
    )
