"""Strict engine compatibility guard: no silent version fallback."""
from importlib import metadata
PINNED_VLLM_VERSION = "0.29.0"

def assert_engine_version() -> str:
    try:
        installed = metadata.version("vllm")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("vLLM is not installed; install h2o-vllm[vllm]") from exc
    if installed != PINNED_VLLM_VERSION:
        raise RuntimeError(
            f"H2O research adapter requires vLLM=={PINNED_VLLM_VERSION}; "
            f"installed {installed!r}"
        )
    return installed
