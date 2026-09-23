"""Load Arm's NSS v1 (neural-graphics-model-gym, Apache-2.0) with only torch/torchvision/safetensors.

The gym package pulls in executorch, pydantic, slangtorch, ... at import time; NSS v1's own model,
its torch pre/post-processing and the data processing need none of them. This module registers
empty parent packages (so the gym's heavy __init__.py files never run) plus small stand-ins for the
four framework modules model_v1.py imports (config, base model, model registry, slang loader), then
builds NSSV1Model with the template config's settings and processing_backend="torch".
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

GYM = Path(os.environ.get("NSS_GYM", Path.home() / ".cache/arm-nss/gym"))
GYM_COMMIT = "fc5fdaf6caf0ff179b1be334c61be0503f0a5f81"  # arm/neural-graphics-model-gym main, 2026-09-18


def _pkg(name: str) -> None:
    if name in sys.modules:
        return
    m = types.ModuleType(name)
    m.__path__ = [str(GYM / "src" / name.replace(".", "/"))]
    sys.modules[name] = m


def _stub(name: str, **attrs) -> None:
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m


def _install() -> None:
    if (
        "ng_model_gym.core.config.config_model" in sys.modules
    ):  # once: model_v1 keeps the first stubs
        return
    import torch

    for p in (
        "ng_model_gym",
        "ng_model_gym.core",
        "ng_model_gym.core.model",
        "ng_model_gym.core.model.layers",
        "ng_model_gym.core.model.shaders",
        "ng_model_gym.core.data",
        "ng_model_gym.core.utils",
        "ng_model_gym.core.config",
        "ng_model_gym.usecases",
        "ng_model_gym.usecases.nss",
        "ng_model_gym.usecases.nss.model",
        "ng_model_gym.usecases.nss.utils",
        "ng_model_gym.usecases.nss.data",
    ):
        _pkg(p)

    class NSSV1ModelSettings(types.SimpleNamespace):
        pass

    class BaseNGModel(torch.nn.Module):
        def __init__(self, params):
            super().__init__()
            self.params = params

    _stub(
        "ng_model_gym.core.config.config_model",
        ConfigModel=object,
        NSSV1ModelSettings=NSSV1ModelSettings,
    )
    _stub("ng_model_gym.core.model.base_ng_model", BaseNGModel=BaseNGModel)
    _stub(
        "ng_model_gym.core.model.model_registry",
        register_model=lambda **kw: (lambda c: c),
    )
    _stub(
        "ng_model_gym.core.model.shaders.slang_utils",
        load_slang_module=None,
        SlangOutput=object,
    )


def build(quality: str = "high", scale: float = 2.0):
    """NSSV1Model(quality) with the template config (reinhard tonemapper, luma derivative, sharp theta)."""
    if not (GYM / "src").exists():
        sys.exit(
            f"clone arm/neural-graphics-model-gym at {GYM_COMMIT} into {GYM} (nss.py fetch)"
        )
    _install()
    mv1 = importlib.import_module("ng_model_gym.usecases.nss.model.model_v1")
    settings = sys.modules["ng_model_gym.core.config.config_model"].NSSV1ModelSettings(
        scale=scale,
        recurrent_samples=16,
        quality=quality,
        processing_backend="torch",
        nss_v1_luma_derivative=True,
        nss_v1_sharp_theta=True,
        gt_history_augmentation=False,
        gt_history_augmentation_chance=0.0,
    )
    params = types.SimpleNamespace(
        model=settings, dataset=types.SimpleNamespace(tonemapper="reinhard", exposure=2)
    )
    return mv1.NSSV1Model(params)


def process_frame(frame: dict):
    """The gym's test-time data processing (tonemapping etc.) of one safetensors window."""
    _install()
    pf = importlib.import_module("ng_model_gym.usecases.nss.data.process_functions")
    return pf.process_nss_data(frame, augment=False, exposure=2, tonemapper="reinhard")
