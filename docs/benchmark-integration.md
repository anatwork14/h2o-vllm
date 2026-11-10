# Contract with anatwork14/vllm-benchmark

| Component | Pinned source | Scope |
| --- | --- | --- |
| Original H2O | `FMInference/H2O@ac75c2a8a9e76832b2a4139b9363373b56336bfb` | Original HF/FlexGen algorithms accessible via submodule |
| Engine API | `vllm==0.29.0`, upstream tag SHA `98dff2a81d747d1dba01a47f939f48c3526d4206` | Target only, not GPU-qualified |
| Canary | `h2o_vllm.vllm_backend.H2OPassthroughBackend` | Unmodified FlashAttention, NOT H2O |

`python -m h2o_vllm --heavy-hitter-tokens 128 --recent-tokens 128` emits a JSON object with `schema_version=h2o-vllm/1`, `engine_version=0.29.0`, `method=h2o-policy-simulation`, `real_h2o_eviction=false`, `gpu_qualified=false` and the original upstream SHA.

**Consumer policy:** Reject `method=h2o` unless `real_h2o_eviction=true`, and reject H2O GPU performance claims unless `gpu_qualified=true`. Record package and engine SHA, hardware, attention backend, KV budget/block size, cache-hit settings, speculative config and quality verification.

`H2O_VLLM_MODE=passthrough` is only for packaging/registration smoke tests, labeled `flash-attn-passthrough`, never a H2O benchmark.

The target `anatwork14/vllm-benchmark` currently contains only a README and LICENSE. No changes are required there until an actual runtime-qualified H2O backend is ready. Keep the method packaged here instead of copying experimental vLLM internals into the benchmark repo.

## Pending GPU acceptance tests

- Custom backend loading and full-cache output/logit parity at batch=1 and batch>1.
- Per-head H2O attention-score parity against HF dense reference.
- Real KV allocation/eviction/reuse safety with refcounts and correct absolute positions.
- Qwen3.5 VLM multimodal prefill, continuous batching, cancellation and cache sharing.
- Per-speculative-paradigm draft/target correctness with MTP, DFlash and EAGLE3.
- Measured allocated and peak VRAM reduction and inference speed with no score fabrication.
