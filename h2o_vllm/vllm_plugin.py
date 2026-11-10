"""vLLM 0.29.0 entry point: explicitly labeled FlashAttention pass-through.

Only a registration/wiring canary. Real H2O scoring and physical KV eviction
are not integrated. Deliberately fail closed rather than spoof H2O results.
"""
import os
from .compat import assert_engine_version

def register() -> None:
    assert_engine_version()
    mode = os.getenv("H2O_VLLM_MODE", "disabled").lower()
    if mode == "disabled":
        return
    if mode != "passthrough":
        raise RuntimeError(
            "H2O vLLM real KV eviction is not integrated or GPU-qualified. "
            "Only H2O_VLLM_MODE=passthrough is available as a wiring canary; "
            "never label it an H2O experiment."
        )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    register_backend(AttentionBackendEnum.CUSTOM, "h2o_vllm.vllm_backend.H2OPassthroughBackend")
