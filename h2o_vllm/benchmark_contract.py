"""Conservative, versioned capability declaration for vllm-benchmark.

This is a static TARGET description, not a report that vLLM actually ran.
"""
from dataclasses import asdict
from .compat import PINNED_VLLM_VERSION
from .policy import H2OPolicyConfig

SCHEMA_VERSION = "h2o-vllm/1"
H2O_ORIGINAL_SHA = "ac75c2a8a9e76832b2a4139b9363373b56336bfb"
VLLM_TARGET_SHA = "98dff2a81d747d1dba01a47f939f48c3526d4206"


def manifest(config: H2OPolicyConfig, *, mode: str = "policy-only") -> dict:
    if mode not in ("policy-only", "passthrough"):
        raise ValueError("Runtime H2O mode is deliberately unsupported")
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": "vllm",
        "engine_version": PINNED_VLLM_VERSION,
        "engine_tag_sha": VLLM_TARGET_SHA,
        "runtime_verified": False,
        "method": "h2o-policy-simulation" if mode == "policy-only" else "flash-attn-passthrough",
        "mode": mode,
        "attention_probabilities_collected": False,
        "physical_kv_eviction": False,
        "real_h2o_eviction": False,
        "gpu_qualified": False,
        "performance_claims_allowed": False,
        "policy": asdict(config),
        "source": {
            "repository": "https://github.com/FMInference/H2O",
            "commit": H2O_ORIGINAL_SHA,
        },
    }
