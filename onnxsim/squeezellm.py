"""SqueezeLLM (Kim et al., 2023, "SqueezeLLM: Dense-and-Sparse
Quantization", https://arxiv.org/abs/2306.07629). onnxsim ports the
algorithm, not any framework's code, per the same rationale as
:mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.hqq` (SqueezeLLM's own
reference implementation quantizes live PyTorch ``nn.Linear`` weights, with
no ONNX export path).

Every weight-only INT4 scheme already in onnxsim (``quantize_weight_only_int4``
and everything built on it, plus :mod:`onnxsim.hqq`) quantizes onto a
*uniform* integer grid -- one scale (and, for HQQ, one zero-point) per
group, evenly spaced codes. SqueezeLLM instead lets each group pick its own
small set of arbitrary values (a per-group codebook, not an arithmetic
sequence), fit directly to that group's own weight distribution -- so a
group whose values cluster tightly around a couple of modes gets levels
concentrated there, rather than spread evenly across the group's full
range as a uniform grid would. Two ideas combine to decide *where* those
per-group levels land:

- **Sensitivity-weighted k-means.** A weight element's effect on the
  layer's output scales with its input activation's own second moment (the
  same diagonal-Hessian/Fisher approximation :mod:`onnxsim.gptq` computes
  in full -- here only the diagonal, ``mean(x_k ** 2)`` per input channel
  ``k``, is needed). Each group's codebook is fit by ordinary Lloyd's-
  algorithm k-means, except each element's contribution to a centroid's
  update is weighted by that sensitivity -- so a centroid drifts to sit
  closer to elements the layer's output is more sensitive to, and the
  overall codebook is a weighted-least-squares-optimal fit to the group's
  actual distribution, not a fixed shape guessed in advance (e.g. NF4's
  fixed Gaussian-quantile codebook).
- **Dense-and-sparse decomposition.** A small fraction of weight elements
  (the paper's own default, ``0.45%``, by magnitude across the whole
  tensor) are excluded from the k-means fit entirely (so a handful of huge
  outliers can't drag a group's codebook toward them at the expense of
  every other element) and instead corrected back to their *exact* original
  value by a separate additive term. Unlike :mod:`onnxsim.llm_int8`
  (which excludes activation outlier *channels* from an INT8 matmul and
  computes the excluded part in float), this decomposition is on the
  *weight*, and the correction here is represented as an ordinary dense
  float32 initializer that is zero everywhere except at outlier positions
  -- exact, and expressible with a plain ``Add``, at the cost of not
  getting genuine sparse storage/compute savings (a real sparse-matrix
  deployment format is a separate, downstream concern this module does not
  address, matching :mod:`onnxsim.nf4`'s own choice to trade deployment
  compactness for a graph expressible in ordinary ONNX ops).

Dequantization is expressed with ``GatherND(codebook, codes, batch_dims=1)``
-- a per-group codebook lookup, unlike :mod:`onnxsim.nf4`'s plain ``Gather``
against one *global* fixed codebook -- followed by ``Reshape`` to unblock,
an ``Add`` of the sparse correction, and (when the weight was not already
stored transposed) a ``Transpose`` back to the node's own layout. No custom
op or contrib domain is needed, only ``GatherND``'s ``batch_dims`` support
(opset 12+).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors
from onnxsim.onnx_simplifier import quantize_weight_only_squeezellm_cpp


def quantize_weight_only_squeezellm(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 32,
    bits: int = 4,
    outlier_fraction: float = 0.0045,
    num_kmeans_iterations: int = 20,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight (whose reduction dimension ``K`` is evenly divisible by
    ``block_size``) and a plain 2-D activation input into SqueezeLLM-style
    dense-and-sparse non-uniform quantization -- see this module's own
    docstring for the technique.

    Delegates to the verified C++ port
    (:func:`onnxsim.quantize_weight_only_squeezellm_cpp`,
    ``squeezellm_entry.h``), which reproduces this module's own
    deterministic (no random sampling) weighted k-means fit closely --
    see that port's own "NUMERICAL SCOPE" note.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            input channel's sensitivity (``mean(x_k ** 2)``) on. Each batch
            is a ``{input_name: np.ndarray}`` dict matching ``model``'s
            graph inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative sensitivity estimate than random
            input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param block_size: elements per ``(output channel, block)`` codebook
            group along the reduction dimension
    :param bits: codebook size is ``2 ** bits`` centroids per group (the
            paper's own default, ``4``, i.e. 16 centroids)
    :param outlier_fraction: fraction of weight elements (by magnitude,
            across the whole tensor) excluded from the k-means fit and
            corrected back to their exact original value instead (the
            paper's own default, ``0.0045``, i.e. 0.45%)
    :param num_kmeans_iterations: weighted Lloyd's-algorithm iterations
            refining each group's codebook
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight replaced by
            ``GatherND(codebook, codes, batch_dims=1)`` (per-group
            codebook lookup) followed by ``Reshape``, an ``Add`` of the
            sparse outlier correction, and (when needed) a ``Transpose``,
            feeding the original MatMul/Gemm node; layers with a
            non-constant, non-2-D, non-block-divisible, or activation-less
            weight are left untouched
    """
    return quantize_weight_only_squeezellm_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        block_size=block_size,
        bits=bits,
        outlier_fraction=outlier_fraction,
        num_kmeans_iterations=num_kmeans_iterations,
        providers=providers,
    )
