"""vLLM 0.29.0 plugin: explicitly NOT a running H2O eviction backend.

Disabled by default and inert when vLLM runs for unrelated workloads. An
enabled canary validates the exact target engine before registering a custom
FlashAttention *passthrough* backend. There is no H2O runtime arm yet.
"""
import os
from .compat import assert_engine_version

def register() -> None:
    mode = os.getenv("H2O_VLLM_MODE", "disabled").strip().lower()
    # General plugins load in every engine/worker process. Do not globally
    # break unrelated vLLM workloads when this package happens to be installed.
    if mode == "disabled":
        return
    if mode not in ("passthrough",):
        raise RuntimeError(
            f"Unsupported H2O_VLLM_MODE={mode!r}. Real H2O GPU eviction is "
            "not implemented. Allowed values: disabled, passthrough. "
            "Passthrough must NEVER be reported as H2O."
        )
    # The authors' Declarative Attention plugin also owns the single CUSTOM
    # backend slot. Reject a combined process even if H2O's plugin happens to
    # load first; load order must never silently select a different kernel.
    if any(os.getenv(name) for name in ("DATTN_MODEL", "DATTN_BASE_BACKEND")):
        raise RuntimeError(
            "H2O pass-through and Declarative Attention cannot share the "
            "vLLM CUSTOM backend slot. Run them in isolated engine processes."
        )
    assert_engine_version()
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )
    custom = AttentionBackendEnum.CUSTOM
    if custom.is_overridden():
        raise RuntimeError(
            "CUSTOM vLLM attention backend is already registered. "
            "Refusing to overwrite another plugin (possibly Declarative Attention)."
        )
    register_backend(
        custom,
        "h2o_vllm.vllm_backend.H2OPassthroughBackend",
    )
