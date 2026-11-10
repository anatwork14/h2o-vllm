# Contract with anatwork14/vllm-benchmark

| Component | Pinned source | Scope |
| --- | --- | --- |
| Original H2O | `FMInference/H2O@ac75c2a8a9e76832b2a4139b9363373b56336bfb` | Original HF/FlexGen algorithms accessible via submodule |
| Engine API | `vllm==0.29.0`, upstream tag SHA `98dff2a81d747d1dba01a47f939f48c3526d4206` | Target only, not GPU-qualified |
| Canary | `h2o_vllm.vllm_backend.H2OPassthroughBackend` | Unmodified FlashAttention, NOT H2O |

`python -m h2o_vllm --heavy-hitter-tokens 128 --recent-tokens 128` emits a JSON object with `schema_version=h2o-vllm/1`, `engine_version=0.29.0`, `method=h2o-policy-simulation`, `real_h2o_eviction=false`, `gpu_qualified=false` and the original upstream SHA.

**Consumer policy:** Reject `method=h2o` unless `real_h2o_eviction=true`, and reject H2O GPU performance claims unless `gpu_qualified=true`. Record package and engine SHA, hardware, attention backend, KV budget/block size, cache-hit settings, speculative config and quality verification.

`H2O_VLLM_MODE=passthrough` is only for packaging/registration smoke tests, labeled `flash-attn-passthrough`, never a H2O benchmark.

The `anatwork14/vllm-benchmark` feature branches now contain a real
research_bench benchmark harness and vLLM 0.29.0 source snapshot. This document
must not claim it is a README-only repository. The current H2O library is still
policy-only, so there is NO qualified H2O runtime to connect into either
`vllm-benchmark` or `anatwork14/colab-cli-server`.

The standalone `colab-cli-server` research branch contains
`research_ops/vllm_engine/h2o_bridge.py`, a **no-GPU, no-eviction**
candidate/protection policy seam. It validates this package's signed-by-source
manifest contract, including `real_h2o_eviction=false`. It is not a loadable
combined attention backend and cannot report VRAM savings.

## Conflict with Declarative Attention

Both this H2O canary and the author-original
`itsnamgyu/declarative-attention` plugin register into vLLM 0.29.0's
**single CUSTOM backend slot**. Attempting to enable both packages in one
vLLM process is unsafe. H2O `vllm_plugin.register()` now rejects
`DATTN_MODEL`/`DATTN_BASE_BACKEND` environments and existing CUSTOM
registrations before mutating the registry. For real experiments, isolate
DA-only and H2O-only engines in separate processes until a **single**
composable backend owns the metadata builder and scheduler policy.

**Fundamental semantic conflict:** DA can return to GLOBAL (dense history)
after a FOCUS/LOCAL phase. Physically evicting a block simply because it
was not selected by a previous DA step can make that GLOBAL transition
incorrect. Before any DA × H2O physical eviction, prove either that future
reaccess is impossible or that the exact original KV can be reconstructed
with absolute positions intact. A CPU mask union is not sufficient.

Do not interpret a candidate H2O block-drop plan as permission to free
shared vLLM KV pages; cache manager refcounts, mixed hybrid attention,
prefix cache, speculation and restart/cancellation require real GPU proof.


## Pending GPU acceptance tests

- Custom backend loading and full-cache output/logit parity at batch=1 and batch>1.
- Per-head H2O attention-score parity against HF dense reference.
- Real KV allocation/eviction/reuse safety with refcounts and correct absolute positions.
- Qwen3.5 VLM multimodal prefill, continuous batching, cancellation and cache sharing.
- Per-speculative-paradigm draft/target correctness with MTP, DFlash and EAGLE3.
- Measured allocated and peak VRAM reduction and inference speed with no score fabrication.
