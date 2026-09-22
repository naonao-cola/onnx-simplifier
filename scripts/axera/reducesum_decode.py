"""First decode of AX650 ``ReduceSum``'s ``npu_params`` input tile table.

Characterization only -- see ``docs/axera-reducesum-decode.md``. For a
single-axis (``axis=0``) reduction whose kept dimensions ("row") stay under
a per-row byte budget, the main offset block of ``npu_params`` reuses
``dma_tile_predict.py``'s already-decoded elementwise DMA tiling rule
exactly: same ``entries``/``stride``/Python-``set``-order derivation, just
applied with ``L`` = the reduced axis's size and ``row`` = the product of
the kept dimensions. This is confirmed for two shapes; a larger-row shape and
the real graph's two-axis (``axis=(0,3)``) reduction both show additional,
undecoded structure -- this module predicts only the confirmed main block,
not the trailing tail group, and raises on anything outside that scope.

This does not predict the compute segment (segment 2): that remains as
opaque here as for every other op this project has decoded, so this module
is not an emitter.
"""

from __future__ import annotations


def predict_main_offsets(
    n: int, row: int, elem_bytes: int = 4, budget: int = 524288
) -> list[int]:
    """The confirmed main offset block for a single-axis reduction over ``n``
    elements, with ``row`` elements kept per reduced-axis step.

    Reuses ``dma_tile_predict.predict_words``'s entries/stride/set-order rule
    (see the module docstring for the confirmed vs. undecoded shapes). Returns
    the ``entries`` offset values only -- not the trailing tail group observed
    after them in the real table, which is not decoded.
    """
    entries = 4
    while True:
        chunk = -(-n // entries)
        if chunk * row * elem_bytes <= budget or entries >= n:
            break
        entries *= 2
    stride = chunk * row * elem_bytes
    return list(set(k * stride for k in range(entries)))
