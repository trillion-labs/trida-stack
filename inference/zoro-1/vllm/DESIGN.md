# Serving Trida block-diffusion on vLLM — native `ModelState` port

This documents the design of the native vLLM 0.27.x block-diffusion backend:
how the Qwen3.5 two-stream block-diffusion checkpoint is served on a stock
vLLM install, and why it is built the way it is.

## Why this path (and not the dllm-plugin)

The out-of-tree `vllm-project/dllm-plugin` was not viable here:

- it pins **vLLM 0.20.x** (a downgrade from the **0.27.x** that already serves
  the autoregressive path),
- it requires a source-built coherence **fork** (needs `nvcc`),
- and its `DllmRuntimeWorker` deadlocks against a stock wheel (base-vs-fork IPC
  mismatch),
- all to enable **pure diffusion**, the weakest decode mode.

Instead, **vLLM 0.27.x already ships everything needed, in-tree**:

| piece | where (in vllm 0.27.x) |
|---|---|
| block-diffusion decode machinery + reference | `model_executor/models/diffusion_gemma.py` (`DiffusionGemmaModelState`) |
| the `ModelState` abstraction | `v1/worker/gpu/model_states/interface.py` |
| **Qwen3.5 model (base arch)** | `model_executor/models/qwen3_5.py` (`Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`) |
| **Qwen3-Next GDN backbone** | `model_executor/models/qwen3_next.py` (`Qwen3NextAttention`) |
| **GDN linear-attention kernels** | `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` |

So the port is **marrying two things vLLM already has** — the Qwen3.5-GDN
backbone (which runs the AR path) and the block-diffusion `ModelState` — not
building a diffusion engine or reimplementing GDN.

## The `ModelState` contract (what we implement)

A model advertises its state via `@staticmethod get_model_state_cls()`. The v1 gpu
`model_runner` drives, per step, for diffusion requests:

- **`prepare_inputs(input_batch, req_states) -> {"inputs_embeds": ...}`** — embed the
  canvas into a *persistent* `inputs_embeds` buffer (CUDA-graph stable address), and
  apply **self-conditioning** for denoise requests.
- **`prepare_attn(input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups,
  kv_cache_config, ...) -> attn_metadata`** — build attention metadata with a **per-request
  `causal` tensor**: `causal=True` for commit/clean (encoder) requests, `False` for denoise
  (decoder) requests, so mixed batches work.
- **`custom_sampler(sampler) -> (DiffusionSampler, None)`** — the accept/renoise commit gate.
- **`add_request` / `remove_request`** — per-request diffusion state bookkeeping.
- plus `get_mm_embeddings` (abstract; return `None` for text-only).

DiffusionGemma is a single backbone whose `forward(mode=...)` switches
**encoder (causal, writes KV)** vs **decoder (bidirectional, reads KV)**.

## The mapping to Trida (two-stream GDN)

**Encoder/decoder is exactly the clean/noisy two-stream:**

| DiffusionGemma | Trida |
|---|---|
| encoder mode — causal, KV write | **clean stream** (committed / carried state) |
| decoder mode — bidirectional, KV read | **noisy stream** (denoise the masked block) |
| `is_encoder_phase[req]` → per-request `causal` | per-request commit-vs-denoise flag |
| `DiffusionSampler` (entropy-bound accept) | **confidence-shift / threshold commit gate** |
| self-conditioning (probs @ embed) | carried-seed shift variant |

## The crux — the ¾ GDN layers

Qwen3.5 is hybrid: **3/4 layers are `linear_attention` (gated-delta, recurrent), 1/4 are
`full_attention` (softmax).** The `prepare_attn` per-request `causal` tensor works **as-is
for the full-attention layers** (vLLM's softmax backend already mixes causal/bidirectional
per request). The **GDN layers do not take a `causal` mask** — they are a left-to-right
recurrent state machine. For diffusion we need:

- **clean pass:** run the GDN scan causally, committing the block-end recurrent state.
- **noisy pass:** re-scan the masked block **from the clean block-entry state**, without
  corrupting the committed clean state (snapshot/restore of the recurrent + short-conv state).

This is the **block-end-readout two-stream GDN scan** used by the HybridDiffusion
two-stream SGLang backend (the SGLang reference) and by training. The port runs
that scan **inside vLLM's GDN layer + its mamba/GDN state cache**
(`layers/mamba/gdn/`), driven by the ModelState's clean/noisy mode — reuse, not
new kernels. The kernel itself lives in `block_causal_readout.py` (derived from
FLA; see that file's header).

## Implementation status

The full two-stream path is implemented:

1. Module + registration + `Qwen3_5DiffusionModelState` with all `ModelState` hooks;
   model class wrapping `Qwen3_5ForCausalLM` with a `mode` switch.
2. Full-attention path: the ModelState serves with the ¼ full-attention layers
   doing causal(clean)/bidirectional(noisy) via the per-request `causal` tensor.
3. GDN two-stream: block-ENTRY snapshot/restore of the recurrent+conv state, plus
   the block-END readout kernel for the noisy pass, patched onto the GDN layer.
4. Commit gate: a lean confidence-shift / threshold `DiffusionSampler` mirroring
   the SGLang reference's LowConfidence family.

## Notes

- No vLLM fork, no 0.20.x. Runs on a stock vLLM 0.27.x install. On older GPU
  drivers you may additionally need a CUDA compat shim (see `serve_diffusion.sh`).
- AR serving is unaffected (already works via `Qwen3_5ForCausalLM` on this vLLM).

## Self-spec (AR-Trust) mode

The same plugin also serves FLARE's AR-Trust decoding: the canvas is `[pending, specs, MASKs]`
(2N−1 slots), one forward both drafts N−1 new tokens from the MASK slots and verifies the
previous specs against exact AR logits (greedy accept), and the GDN state is committed from a
per-step intermediate-state ring with no copy-back. Since 2026-09-10 the step is presented to
vLLM as a spec-decode batch (bonus token + 2N−2 drafts) so FULL cuda graphs capture the whole
forward.
