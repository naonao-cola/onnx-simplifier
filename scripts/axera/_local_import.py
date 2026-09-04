#!/usr/bin/env python3
"""Fixes a real cross-vendor test collision on the bare names "models"/"worker".

``scripts/qualcomm``, ``scripts/intel``, ``scripts/amd``, ``scripts/apple``,
and ``scripts/axera`` each carry their own ``models.py`` (and the first four
also carry their own ``worker.py``), and each vendor's test file
(``tests/test_*_compat.py``) does the same dance: prepend its own
``scripts/<vendor>`` dir to ``sys.path``, then ``import models`` (or ``from
worker import check``) by its bare name. That is fine as long as each
vendor's test file runs in its own process -- but the default test suite
(``pytest tests/``) collects *all* of them into one process, and Python
caches imported modules by bare name in ``sys.modules``: whichever vendor's
``models.py`` gets imported *first* (at collection time, even if that
vendor's tests are then skipped -- the module-level ``import models``
statement still runs) stays cached under the name ``"models"`` for the rest
of the process. Every other vendor's later ``import models`` silently reuses
that first one instead of its own.

This went unnoticed because every pre-existing vendor ``models.py`` is a
thin, functionally-identical alias for the same ``scripts/common/
synthetic_models.py`` suite -- getting "the wrong vendor's models.py" changed
nothing observable. It became a real, visible bug the moment
``scripts/axera/models.py`` added something the others don't have
(``axera_npu_compiled_leaf``): whichever vendor test file collected first
poisoned ``sys.modules["models"]``, and axera's own `models.build(...)` and
`worker.check(...)` calls (both reached in-process, not via a subprocess)
silently ran against a different vendor's module instead.

:func:`fresh` forces a real import from a specific directory regardless of
what is already cached under that bare name.
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType


def fresh(name: str, directory: str) -> ModuleType:
    """Import ``name`` from ``directory``, ignoring a same-named module
    already cached in ``sys.modules`` from a different directory.

    ``directory`` must already be on ``sys.path`` (callers here always
    prepend their own ``HERE`` before calling this).
    """
    cached = sys.modules.get(name)
    if cached is not None:
        cached_file = getattr(cached, "__file__", None)
        if (
            cached_file is None
            or os.path.dirname(os.path.abspath(cached_file)) != directory
        ):
            del sys.modules[name]
    return importlib.import_module(name)
