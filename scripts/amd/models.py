#!/usr/bin/env python3
"""AMD-side alias for the shared synthetic model suite. See scripts/common/."""

from __future__ import annotations

import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from common.synthetic_models import (  # noqa: E402,F401
    all_models,
    build,
    conv_bn_relu,
    foldable_shape_reshape,
    matmul_bias_tanh,
    names,
    redundant_transpose,
    sigmoid_mul_swish,
)

if __name__ == "__main__":
    for n, m in all_models().items():
        print(f"{n:24} {len(m.graph.node)} nodes")
