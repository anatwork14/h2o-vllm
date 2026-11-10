# H2O → vLLM 0.29.0 research port

**Status: CPU policy unit-tested; vLLM `0.29.0` API registration canary only. Real H2O GPU KV eviction is NOT IMPLEMENTED or qualified. Do not report the canary as H2O latency, VRAM, quality, or throughput.**

This repository ports the H2O heavy-hitter plus recent-token cache **selection policy** from [FMInference/H2O](https://github.com/FMInference/H2O), pinned to commit `ac75c2a8a9e76832b2a4139b9363373b56336bfb`, and exports a benchmark contract for [anatwork14/vllm-benchmark](https://github.com/anatwork14/vllm-benchmark).

The complete original upstream repository is linked as a Git submodule at `upstream/H2O`, preserving the original source rather than rewriting its legacy Hugging Face or FlexGen systems as though they were compatible with vLLM V1.

## Reproduce the original upstream

```bash
git clone --recurse-submodules https://github.com/anatwork14/h2o-vllm.git
cd h2o-vllm
git submodule status
# Original H2O dropper:
# upstream/H2O/h2o_hf/utils_real_drop/modify_llama.py
```

Source paper: **H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models** (original upstream README labels its code MIT).

## Implemented components

- `h2o_vllm.policy`: deterministic H2O heavy-hitter / recent / optional sink token selector fed by real externally supplied attention mass.
- `h2o_vllm.headwise`: per-head H2O selector and block-union diagnostic; not a physical eviction mechanism.
- `h2o_vllm.session`: request/layer state isolation and cleanup.
- `h2o_vllm.compat`: strict `vllm==0.29.0` guard.
- `h2o_vllm.vllm_plugin` and `vllm_backend`: **FlashAttention pass-through only**, registered through vLLM `vllm.general_plugins` and `AttentionBackendEnum.CUSTOM`; it does **not** score, prune, or evict KV cache.
- `h2o_vllm.benchmark_contract`: versioned provenance, configuration and explicit qualification status.

## CPU tests / metadata

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m h2o_vllm --heavy-hitter-tokens 128 --recent-tokens 128 > h2o-policy-manifest.json
```

To test the declared vLLM target **in a separate GPU environment**:

```bash
python -m pip install -e '.[vllm]'
python -c 'from h2o_vllm.compat import assert_engine_version; print(assert_engine_version())'
```

To smoke-test only the custom backend registration (NOT H2O):

```bash
H2O_VLLM_MODE=passthrough vllm serve MODEL_ID --attention-backend CUSTOM
```

The GPU startup command has **not** been executed or validated. `H2O_VLLM_MODE=h2o` intentionally fails closed rather than claiming unimplemented eviction.

## Scientific integration gates

To call this a *working H2O×vLLM 0.29.0* integration, complete and verify:

1. **Real attention probabilities** per request, layer, head and decode step. Stock FlashAttention output is not a substitute for H2O scores; prove score parity against the original HF implementation.
2. **Physical KV eviction with scheduler synchronization**, including block allocator, refcounts, prefix sharing, absolute RoPE positions, slot mappings and writes after eviction. Hiding blocks or compacting a block table is insufficient.
3. **Continuous batching and VLM support**, including Qwen3.5 GQA/multimodal positions, request cancellation, cache reuse, CUDA graph behavior and unsupported-mode rejection.
4. **Speculative decoding validation** for MTP, DFlash, EAGLE3; validate draft-only, target-only and both independently.
5. **Evidence**: same-engine fixed-hardware comparisons with real cache byte savings, TTFT/TPOT/throughput, quality and GPU execution traces.

**Until the gates are passed, a benchmark runner must reject any proposed `h2o` arm.** Valid current labels are `h2o-policy-simulation` and `flash-attn-passthrough`. See [benchmark contract](docs/benchmark-integration.md).
