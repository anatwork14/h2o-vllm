"""Stable export metadata for anatwork14/vllm-benchmark."""
from dataclasses import asdict
from .compat import PINNED_VLLM_VERSION
from .policy import H2OPolicyConfig

SCHEMA_VERSION = "h2o-vllm/1"

def manifest(config: H2OPolicyConfig, *, mode: str = "policy-only") -> dict:
    if mode not in ("policy-only", "passthrough"):
        raise ValueError("Runtime H2O mode is deliberately unsupported")
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": "vllm",
        "engine_version": PINNED_VLLM_VERSION,
        "method": "h2o-policy-simulation" if mode == "policy-only" else "flash-attn-passthrough",
        "mode": mode,
        "real_h2o_eviction": False,
        "gpu_qualified": False,
        "performance_claims_allowed": False,
        "policy": asdict(config),
        "source": {
            "repository": "https://github.com/FMInference/H2O",
            "commit": "ac75c2a8a9e76832b2a4139b9363373b56336bfb",
        },
    }
