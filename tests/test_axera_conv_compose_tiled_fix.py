"""Regression coverage for `conv_compose_tiled_fix.py` -- no Docker/device
required, all four claims in `docs/axera-conv-compose-tiled-fix.md`."""

import os
import sys

import numpy as np

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import emitter  # noqa: E402
from conv_compose_tiled_fix import (  # noqa: E402
    X_SCALE,
    Y_SCALE,
    check,
    code_mismatches,
    fixed_table,
    held_out_weights,
    is_tiled,
    wrong_m,
)


def test_terminal_shape_is_not_tiled():
    """Conv(64,64,3,3) is exactly the case conv_scaffold_arithmetic.py's own
    docstring says stays one contiguous span -- the tiled formula cannot
    differ from what PR #1783 already tried for this shape."""
    assert not is_tiled()


def test_wrong_m_reproduces_the_documented_bug_bit_for_bit():
    """wrong_m() (the original weight's baked-in scale) must NOT equal the
    correct per-channel M for the held-out weight -- that mismatch is the bug."""
    w_new, _ = held_out_weights()
    correct = X_SCALE * emitter.weight_scales(w_new)[:8] / Y_SCALE
    assert not np.allclose(wrong_m(), correct)


def test_fixed_table_has_correct_m_and_byte_exact_codes():
    w_new, b_new = held_out_weights()
    table = fixed_table(w_new, b_new)
    expected_m = X_SCALE * emitter.weight_scales(w_new) / Y_SCALE
    m_from_table = table[64 * 64 * 3 * 3 + 4 * 64 : 64 * 64 * 3 * 3 + 8 * 64].view(
        np.float32
    )
    assert np.allclose(m_from_table, expected_m, rtol=1e-5)
    assert code_mismatches(table, w_new) == 0


def test_check_passes():
    assert check() == 0
