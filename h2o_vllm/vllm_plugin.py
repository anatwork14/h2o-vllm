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
    assert_engine_version()
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "h2o_vllm.vllm_backend.H2OPassthroughBackend",
    )
