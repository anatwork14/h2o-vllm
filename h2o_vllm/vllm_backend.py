"""vLLM 0.29.0 CUSTOM backend registration canary (NOT H2O attention)."""
from .compat import assert_engine_version
assert_engine_version()
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

class H2OPassthroughBackend(FlashAttentionBackend):
    """Delegates all operations to FlashAttention unchanged; NOT an H2O arm."""
    @staticmethod
    def get_name() -> str:
        return "H2O_PASSTHROUGH_NOT_H2O"
