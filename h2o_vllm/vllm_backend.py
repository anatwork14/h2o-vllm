"""vLLM 0.29.0 FlashAttention registration canary: NO H2O algorithm.

Preserve FlashAttention's upstream capability/name semantics. The benchmark
manifest, not a backend name override, distinguishes this diagnostic mode.
"""
from .compat import assert_engine_version

assert_engine_version()

from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend


class H2OPassthroughBackend(FlashAttentionBackend):
    """Unmodified FlashAttention: no attention scoring or KV eviction."""
    pass
