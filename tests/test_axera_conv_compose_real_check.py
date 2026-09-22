"""Regression coverage for the Conv real-graph composition check (no Docker/device)."""

import os
import sys

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

from conv_compose_real_check import (  # noqa: E402
    _CONV_LEARN_FIXTURES,
    _FIXTURES,
    TERMINAL_OFFSET,
    _load_gz_model,
    check,
    find_weight_code_offset,
    npu_params,
    standalone_reference,
)


def test_terminal_conv_weight_code_is_byte_exact_at_a_fixed_offset():
    standalone_table, origin = standalone_reference()
    composed = npu_params(_load_gz_model(os.path.join(_FIXTURES, "chain1.axmodel.gz")))
    from conv_compose_real_check import _load_gz_npy

    holdout_w = _load_gz_npy(os.path.join(_CONV_LEARN_FIXTURES, "holdout_w.npy.gz"))
    shift, mismatches = find_weight_code_offset(
        composed, standalone_table, origin, holdout_w
    )
    assert shift == TERMINAL_OFFSET
    assert mismatches == 0


def test_interior_conv_weight_code_does_not_match_anywhere():
    standalone_table, origin = standalone_reference()
    composed = npu_params(_load_gz_model(os.path.join(_FIXTURES, "chain1.axmodel.gz")))
    from conv_compose_real_check import _load_gz_npy

    reference_w = _load_gz_npy(os.path.join(_CONV_LEARN_FIXTURES, "reference_w.npy.gz"))
    _, mismatches = find_weight_code_offset(
        composed, standalone_table, origin, reference_w
    )
    # No clean match anywhere: a real match is 0; an unrelated pattern lands near 50%.
    assert mismatches > len(standalone_table) // 4


def test_offset_is_position_dependent_not_weight_dependent():
    """The swap-control chain: same offset, regardless of which weight set is terminal."""
    standalone_table, origin = standalone_reference()
    composed2 = npu_params(_load_gz_model(os.path.join(_FIXTURES, "chain2.axmodel.gz")))
    from conv_compose_real_check import _load_gz_npy

    reference_w = _load_gz_npy(os.path.join(_CONV_LEARN_FIXTURES, "reference_w.npy.gz"))
    shift, mismatches = find_weight_code_offset(
        composed2, standalone_table, origin, reference_w
    )
    assert shift == TERMINAL_OFFSET
    assert mismatches == 0


def test_check_reports_no_mismatches_on_both_fixtures():
    assert check() == 0
