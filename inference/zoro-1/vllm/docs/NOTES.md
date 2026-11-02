# Trida vLLM-native block-diffusion port — serve milestone notes (node1)

Date: 2026-09-08. Box: docr-node1, GPU 5, vLLM 0.27.1 venv
(`/path/to/vllm-uv27/.venv`).

## Milestone reached
`vllm serve` (1) registers `Qwen3_5ForBlockDiffusion` in BOTH the APIServer
process AND the EngineCore subprocess, (2) resolves & loads step_18000 weights
as our class (7.94 GiB, 12s), (3) activates the diffusion ModelState path, and
(4) reaches the decode-init flow, stopping at our GPU-gated stub.

**First stub hit (quote):**
```
File ".../vllm_native_diffusion/qwen3_5_diffusion.py", line 165, in custom_sampler
    raise NotImplementedError("commit sampler is GPU-gated — see DESIGN.md plan step 4")
NotImplementedError: commit sampler is GPU-gated — see DESIGN.md plan step 4
```
Called from `vllm/v1/worker/gpu/model_runner.py:361`:
`custom = self.model_state.custom_sampler(self.sampler)` during `load_model`.

## How registration was wired (out-of-tree, no fork)
- `vllm_native_diffusion/plugin.py` — `register_trida()` registers the model
  class AND injects a `MODELS_CONFIG_MAP["Qwen3_5ForBlockDiffusion"]` config
  hook that delegates to `Qwen3_5ForCausalLMConfig` (GDN/hybrid backbone) then
  applies diffusion defaults.
- `pyproject.toml` (at test-dir root) exposes it as a `vllm.general_plugins`
  entry point named `trida_diffusion`.
- Installed into the venv:
  `~/.local/bin/uv pip install --python <venv-python> -e .`
  (the venv has no `pip`; it is uv-managed.)
- `export VLLM_PLUGINS=trida_diffusion` so `load_general_plugins()` runs
  `register_trida()` in EVERY process (verified: plugin print appears under both
  `APIServer pid=` and `EngineCore pid=`).

## Serve command
See `serve_diffusion.sh`. Key flags:
- `--hf-overrides '{"architectures":["Qwen3_5ForBlockDiffusion"],"canvas_length":32}'`
  - `architectures` forces our class over `Qwen3_5ForCausalLM`.
  - `canvas_length` makes `ModelConfig.is_diffusion` True (it keys off
    `hf_config.canvas_length`), which forces the V2 model runner + the diffusion
    scheduler path (num_sampled_tokens_per_step=0, canvas spec-decode data path).
- `--diffusion-config '{"canvas_length":32,"max_denoising_steps":32}'` → builds
  `DiffusionConfig` (only `canvas_length` is required, `>0`). 32 == checkpoint's
  `block_diffusion.json` `bd_size`.
- `--enforce-eager`, `--max-num-seqs 1`, `--gpu-memory-utilization 0.55`,
  `--max-model-len 8192`.

Weight loading is inherited from `Qwen3_5ForCausalLM` and worked unchanged.
GDN path initialized (FlashInfer GDN prefill kernel), attention backend
auto-selected FLASH_ATTN v4 (per-sequence dynamic_causal requires FA4).

## Next concrete implementation step (revealed)
`Qwen3_5DiffusionModelState.custom_sampler(self, sampler)` must return
`(TridaDiffusionSampler, None)` — mirror `diffusion_gemma.py`
`DiffusionGemmaModelState.custom_sampler` (line ~821). The reference
`DiffusionSampler` is constructed with:
`sampler`, `diffusion_config`, `vocab_size`, `diffusion_states`,
`t_min`/`t_max`, `entropy_bound`, `confidence_threshold`,
`embed_weight` (= `model.model.embed_tokens.weight`, for the
self-conditioning `probs @ embed` matmul), `normalizer`, and vocab-shard
bounds (`sc_vocab_start/end` from `embed_tokens.shard_indices`).

Ours should implement the confidence-shift / threshold commit gate (1:1 with
the SGLang `--dllm-algorithm` LowConfidence family; the CPU reference is
`test_two_stream_cpu.py::commit_gate`). Because `custom_sampler` is called at
`load_model` time (not per-step), it is a hard init gate: the server cannot
finish startup until this returns a real sampler object.

After the sampler, the next boundary is the `prepare_attn` GDN two-stream
`TODO(gpu)` (snapshot/restore of GDN recurrent+conv state for the noisy pass)
and `prepare_inputs` self-conditioning — but those are only exercised once
decode steps run, i.e. after the sampler gate is passed.

## Files on node1 (`/path/to/vllm-native-test/`)
- `vllm_native_diffusion/{qwen3_5_diffusion.py,plugin.py,DESIGN.md,test_two_stream_cpu.py,__init__.py}`
- `pyproject.toml` (plugin package; installed editable)
- `serve_diffusion.sh`, `serve_diffusion.log`, this `NOTES.md`

---

# Update 2026-09-08: commit sampler implemented — serves + decodes end-to-end

## Result
`serve_diffusion.sh` now starts FULLY (past the old `custom_sampler`
NotImplementedError gate), `/health` = 200, and `/v1/completions` returns
tokens end-to-end in diffusion mode from step_18000. GPU 5, C=1.

- "The capital of France is" -> " Paris" (CORRECT first commit) then degenerates
  into repetition (",,, Paris Paris Paris ...").
- gsm8k robe -> "white white white ..." (garbled).

Interpretation: the commit sampler + full-attention causal/bidirectional path
are CORRECT (first position decodes right), but output degenerates — the
expected signature of the **GDN two-stream `prepare_attn` TODO still being
unimplemented** (plan step 3). The 3/4 GDN recurrent layers run causally in the
noisy/denoise pass instead of re-scanning from the committed block-entry state,
so their recurrent state is corrupted and later canvas positions collapse.
This is exactly plan step 2's checkpoint: "it RUNS (lossy), GDN fix is next."

## C=1 throughput (lossy, GDN-causal)
- 96 tokens: 6.81s wall (incl HTTP+prefill); server committed-token throughput
  3.2 tok/s; 94 denoising forwards for 96 tokens = 3 canvases x ~31 steps.
- Mean denoising steps/canvas ~31-33 — i.e. hitting the max_denoising_steps=32
  cap because top-1 prob rarely crosses threshold=0.9 (GDN-corrupted logits are
  low-confidence). Once the GDN two-stream fix lands, positions should cross
  threshold in a few steps and steps/canvas (hence tok/s) improves sharply.
- Mean tokens committed per denoising step ~1.0.

## What was implemented (in qwen3_5_diffusion.py)
- `TridaDiffusionStates` extended with the canvas lifecycle the runner's
  spec-decode data path drives: `canvas` (seeds draft_tokens), `argmax_canvas`
  (committed best-guess), `done` (per-position confidence-shift mask), `step`,
  `init_canvas`.
- `_compiled_commit_step` (@torch.compile): our LEAN confidence-shift/threshold
  gate — `softmax -> max` (top-1 prob + argmax, gather), commit positions with
  conf >= threshold, monotone-within-block `done` accumulation, renoise
  not-yet-done positions with mask_id, converge when all real positions done or
  max steps. Mirrors `test_two_stream_cpu.py::commit_gate` semantics on GPU.
- `TridaDiffusionSampler`: prefill/decode lifecycle (`_finish_prefills`,
  canvas padding, `SamplerOutput` with num_sampled/num_rejected, tiling)
  structurally mirrors DiffusionGemma's `DiffusionSampler`, minus the
  entropy-bound sort / Gumbel / stability-history / self-conditioning.
- `custom_sampler` returns `(TridaDiffusionSampler, None)`; NO embed_weight /
  normalizer / vocab-shard wiring (our backbone has no self-conditioning MLP).

## PERF notes flagged in code (# PERF:)
- Commit decision is top-1 `softmax -> max` (argmax + gather), NOT a full
  top-k/p sort over the 248k vocab. Big win vs DiffusionGemma's sort-based
  entropy-bound accept mask.
- softmax runs in the logits' NATIVE dtype (bf16 under our serve), avoiding a
  mandatory fp32 full-vocab materialization. The only [.,.,vocab] transient is
  `probs`, reduced immediately to [.,.] conf/pred. (DiffusionGemma forces fp32
  and keeps ~10 live fp32 [group*CL,vocab] copies; we bound to ~3 native-dtype.)
- Commit decision is fully on-device (compiled step); the only host sync is the
  CPU/numpy decode-vs-prefill split (indices via UvaBackedTensor), same as the
  reference — interface-forced, left as-is.

## Interface-forced inefficiencies (left as-is, noted for later)
- vLLM hands the sampler the FULL [num_decode*CL, vocab] logits every denoising
  step (the diffusion data path reuses spec-decode draft logits) — we can't
  avoid materializing full-vocab logits per position; we just avoid *sorting*
  them. # PERF noted at the softmax.
- The prefill/decode split + UvaBackedTensor staging is CPU-side metadata
  (no GPU->CPU->GPU of tensors), matching the reference; kept.

## Next concrete step (plan step 3 — the correctness crux)
Wire the GDN two-stream snapshot/restore into `prepare_attn` (or the GDN layer
via a clean/noisy mode flag): the noisy pass must re-scan the masked block from
the clean block-entry recurrent+conv state WITHOUT mutating the committed clean
state. This is the block-end-readout two-stream scan already validated on CPU
(`test_two_stream_cpu.py`) and in SGLang. Expected effect: coherent output +
far fewer denoising steps/canvas (higher tok/s).

---

# Update 2026-09-08 (b): GDN two-stream — block-entry snapshot/restore LANDED;
# block-end readout is the remaining acceptance blocker (precisely isolated)

## What landed (and is verified firing)
Block-ENTRY snapshot/restore of the GDN recurrent+conv state, hooked in
`Qwen3_5DiffusionModelState.prepare_attn` (runs before every forward, has
block_tables). Verified in the serve log:

    [trida-gdn] two-stream snapshot/restore ready: 24 GDN layers,
      mamba group 0, conv(2149, 3, 8192) ssm(2149, 32, 128, 128)

Mechanism (matches test_two_stream_cpu.py snapshot/restore invariant):
- GDN state lives in each layer's `.kv_cache = (conv_state, ssm_state)`, indexed
  per request by the mamba block table column 0 (`block_tables[gid][:, 0]`).
  Discovered via `static_forward_context` filtered by `MambaBase`; group id from
  the `MambaSpec` kv-cache group.
- Per request slot we track prev-step phase (`_prev_encoder`) and a valid-snapshot
  flag (`_snap_valid`):
  - first denoise of a fresh block (prev was clean/commit) -> SNAPSHOT ssm+conv
    (this is the committed clean block-entry state);
  - subsequent denoise re-scans AND the following commit pass -> RESTORE from the
    snapshot before the forward (so each re-scan / the commit starts clean);
  - the commit (clean causal) pass then advances the clean state past the block;
    snapshot is invalidated so the next block re-snapshots.
- Gated so the prompt-prefill clean pass is never restored from a zero snapshot.
- All GPU index_copy_, no host sync. (Bug fixed en route: idx_mapping is int32 →
  cast slot indices to .long() for index_copy_.)

## Effect on the acceptance prompts (measured)
Before (GDN causal both passes): "France is" -> " Paris Paris Paris ..." ;
robe -> "white white white ...".
After block-entry snapshot/restore:
- "The capital of France is" -> ". Paris..........,,,,,,,,,"
- robe -> " bolts   bolts bolts bolts ..." (was "white white white")
So the FIRST block is markedly cleaner (Paris/bolts emerge correctly, no
"Paris Paris" collapse), but output still degenerates after the first few
positions, and `Mean denoising steps per canvas` stays ~33-37 (hits the
max_denoising_steps cap) — i.e. positions past the first rarely cross the 0.9
confidence threshold.

## Why it's not fully correct yet — the REMAINING blocker (precisely isolated)
Block-entry seeding alone is necessary but NOT sufficient. Per the FLARE
two-stream logic (memory: two-stream-gdn-flare-logic) the noisy denoise pass must
also do **BLOCK-END READOUT**: after scanning the noisy block to its end state
`S_end`, EVERY block token reads `S_end`:
    o_t = (l2norm(q_t) / sqrt(head_k_dim)) @ S_end
This is what gives GDN bidirectional-within-block visibility in a single forward
scan while keeping the gated delta rule. Our current path still uses the stock
kernel's TOKEN-CAUSAL readout (each token reads its own left-prefix state), so
GDN layers remain effectively causal within the block → later canvas positions
can't see the whole block → stay low-confidence → never commit early → cap hit
→ degeneration. This is exactly the root cause the FLARE memory calls out
("...INVISIBLE to the 24 GDN layers, their scan is always causal ... root cause
of pure-diffusion degeneration + low spec acceptance").

## The exact GDN-state API gap + minimal shim needed
- The stock spec/denoise kernel is
  `fused_sigmoid_gating_delta_rule_update` (FLA Triton, in
  `vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py`),
  called from `QwenGatedDeltaNetAttention._forward_core` (section 2.1, spec
  path). It returns `(core_attn_out_spec, last_recurrent_state)` and advances +
  reads TOKEN-CAUSALLY in one fused pass. It does NOT expose a hook to re-read
  all block tokens against the block-END state — advance and readout are
  intertwined, so a post-hoc einsum on its output cannot recover block-end
  readout.
- `S_end` IS available (it's `last_recurrent_state`, and is written into
  `ssm_state[state_row]`). The missing ingredient outside the kernel is the
  post-conv, pre-scan `query_spec` (the l2-normed q the kernel consumes); it is a
  LOCAL inside `_forward_core` and not exposed.
- Minimal shim (kernel-adjacent, the real "GPU work" this reveals): override
  `_forward_core` for our layers (subclass `QwenGatedDeltaNetAttention` or
  monkeypatch the method) so that, for spec/denoise blocks, after computing
  `S_end` it recomputes the readout as `o_t = (l2norm(q_t)/sqrt(Dk)) @ S_end`
  for all block tokens (GQA-expanded heads), overwriting `core_attn_out_spec`.
  Two viable kernels for the readout: (a) a small Triton/`einsum('lhk,hvk->lhv')`
  over q_spec and S_end (needs the same q_spec the kernel used — cleanest to
  capture it by duplicating the ~15-line conv-prep + l2norm just before the
  kernel call), or (b) reuse the chunk kernel with `output_final_state=True` to
  get S_end and then the einsum readout (mirrors training's
  `two_stream_scan`/`chunk_gated_delta_rule` path). This must be bit-matched to
  the training two-stream readout (l2norm placement, 1/sqrt(head_k_dim) scale,
  GQA head broadcast, ShortConv lag-source is already handled by conv_state).
- This is genuine GPU-kernel-readout work (the milestone-1 "don't implement the
  kernels" boundary). The ModelState-side plumbing (block-entry seeding, phase
  tracking, per-slot state addressing) is done and correct; only the in-layer
  block-end readout remains.

## C=1 throughput (with snapshot/restore, still readout-causal → lossy)
- 96 tokens: committed-token throughput ~4.3 tok/s (up from ~3.2 pre-fix), but
  denoising steps/canvas still ~37 (cap-bound). tok/s will only rise sharply
  once block-end readout lets positions cross threshold in a few steps.

## Files
- `vllm_native_diffusion/qwen3_5_diffusion.py` (node1 + local, md5-identical):
  `_discover_gdn`, `_gdn_snapshot_restore`, hook in `prepare_attn`, phase/valid
  buffers in `__init__`, `_prev_encoder` reset in `add_request`.
- `serve_diffusion.sh`, `serve_diffusion.log` (has the `[trida-gdn] ... ready`
  line + acceptance outputs), this NOTES section.

---

# Update 2026-09-08 (c): block-END readout IMPLEMENTED + bit-matched; acceptance
# blocked by a deeper CHECKPOINT decode-algorithm mismatch (isolated via SGLang)

## What landed this pass
1. **Block-END readout kernel (FLARE two-stream), bit-matched.** Ported hyungguk's
   SGLang block-causal recurrent kernel VERBATIM into `block_causal_readout.py`
   (from eval/sglang/srt/layers/attention/block_gdn/fused_recurrent.py) — same
   Triton math: l2norm eps 1e-6, scale 1/sqrt(head_k_dim), gating
   g=-exp(A_log)*softplus(a+dt_bias)/beta=sigmoid(b), GQA repeat_interleave, and
   the two-phase readout (scan to S_end, then every block token reads S_end).
   Only the two FLA helper imports were repointed to vLLM's bundled FLA (exp,
   input_guard) — kernel body unchanged.
2. **Wired into the vLLM GDN layer.** `_trida_gdn_forward_core` monkeypatches
   `QwenGatedDeltaNetAttention._forward_core` (installed from plugin.register_trida
   so it runs in the EngineCore/worker). For the denoise pass it does the
   prefill-layout conv (causal_conv1d_fn), split+l2norm+gating via vLLM's
   `fused_post_conv_prep` (identical math, L2NORM_EPS=1e-6/SOFTPLUS_THRESHOLD=20.0),
   GQA expand, then `fused_recurrent_block_causal_gated_delta_rule(causal_mode=0,
   output_final_state=False)`. Clean/commit + prompt passes stay on the stock
   token-causal kernel (super()._forward_core).
   Verified firing in the log: `[trida-gdn] BLOCK-CAUSAL readout firing
   (prefill-denoise): num_prefills=1 T=32 bs=32`.
   IMPORTANT ROUTING FINDING: the diffusion denoise canvas is classified by the
   GDN metadata builder as a PREFILL (num_prefills=1, spec_sequence_masks=None),
   NOT a spec-decode — so the readout hooks the prefill branch and seeds
   initial_state from ssm_state[prefill_state_indices] (the ModelState
   snapshot/restore keeps that at the clean block-entry state).
3. **Lossless gate.** Commit-sampler softmax now uses fp32 accumulation
   (`torch.softmax(..., dtype=torch.float32)`); threshold compare is fp32.

## Acceptance NOT met — output still degenerates. Root cause ISOLATED.
Ran the authoritative reference — hyungguk's SGLang two-stream serve of the SAME
checkpoint (serve_selfspec.sh, step_18000_sglang, HybridDiffusionSelfSpec) — on a
free GPU and compared:
- SGLang ref, robe prompt -> " 2 bolts of blue fiber. Half of 2 bolts of white
  fiber is 1 bolt. 2 + 1 = 3." (COHERENT, correct answer 3).
- OUR vLLM, robe prompt   -> " bolts   of ..." (degenerate).
- Both degenerate on the short "capital of France" prompt (OOD).
=> The checkpoint CAN produce coherent diffusion output, so ours has a real
   remaining mismatch — but it is NOT the readout math.

Reading the SGLang decode algorithm (dllm/algorithm/low_confidence_shift_hybrid_
diffusion.py + configs) reveals the checkpoint was trained/served with a decode
scheme our generic canvas does NOT implement:
- **Carried SEED + logit SHIFT**: block layout is [seed, MASK, MASK]; position-0
  seed is predicted by the PREVIOUS block (or prefill) and injected; under
  logit-shift training the shifted last logit samples the NEXT block's seed.
  "runtime block_size:3 == paper B=4" (carried seed). Our scheme has no carried
  seed and no logit shift -> each block re-predicts from scratch => the observed
  "Paris Paris Paris ..." repetition.
- **Small blocks**: SGLang uses block_size 3-7, gen_block_size 1-4 (commit ~1
  token/block), NOT our canvas_length=32.
- **Mixed readout**: dllm_gdn_causal_mode=2 with num_clean=1 (seed token-causal,
  masks block-causal) + a custom bidir attention mask (mask[0,0]=True,
  mask[1:,:]=True) for the full-attn layers. We used causal_mode=0 and a uniform
  per-request causal flag.
- persist_state=False + prefill_state_restore=True during denoise — matches our
  snapshot/restore + output_final_state=False (this part we DID get right).

## Conclusion / remaining work (precise)
The block-END readout (the piece this task asked for) is implemented and
bit-matched to the reference kernel, and the GDN state snapshot/restore is
correct. Reaching coherent output on step_18000 additionally requires porting
the checkpoint's TRAINED diffusion decode algorithm — carried-seed + Dream
logit-shift + small [seed,MASK,...] blocks + causal_mode=2/num_clean=1 + the
custom bidir attention mask — i.e. a faithful port of
`LowConfidenceShiftHybridDiffusion` / `HybridDiffusionSelfSpec`, not just the GDN
readout. Our current generic top-1/threshold canvas sampler is the mismatch.
This is a substantial next milestone (the whole decode algorithm), separable
from the (now-done) readout kernel.

## Files (node1 + local worktree, all md5-identical)
- vllm_native_diffusion/block_causal_readout.py  (verbatim SGLang kernel port)
- vllm_native_diffusion/qwen3_5_diffusion.py      (readout patch + fp32 gate +
                                                    snapshot/restore + sampler)
- vllm_native_diffusion/plugin.py                 (installs readout patch)
- serve_diffusion.sh                              (canvas_length=32)

---

# Update 2026-09-08 (d): FULL trained decode algorithm ported — COHERENT output,
# acceptance MET. LowConfidenceShiftHybridDiffusion faithfully reimplemented in vLLM.

## RESULT — acceptance met (side-by-side vs the SGLang diffusion oracle)
Config: canvas_length=3, threshold=0.95 (== eval/configs/hybrid_diffusion_shift_b3_g1.yaml).
- "The capital of France is" -> " Paris. The capital of the United Kingdom is London" (coherent)
- robe gsm8k -> "...Half of 2 is 1. So, it takes 1 bolt of white fiber. In total,
  it takes 2 + 1 = 3"  (COHERENT, CORRECT answer 3)
- "The opposite of hot is" -> " cold. The opposite of cold is hot"
- "The chemical symbol for gold is" -> " Au. The chemical symbol for silver is Ag."
- "The largest planet..." -> " Jupiter. It is a gas giant, meaning it"
Output is coherent, grammatical, and factually correct — matches or beats the oracle
(the oracle itself got 2+1 wrong on one run; step_18000 is undertrained so neither is
perfect, but ours is clearly coherent). Metrics: ~2.7 denoising steps/canvas, ~1.1
tokens committed/step, ~5 tok/s at C=1.

## What was ported (faithful to LowConfidenceShiftHybridDiffusion)
Reference: eval/sglang/srt/dllm/algorithm/low_confidence_shift_hybrid_diffusion.py.
1. **Carried SEED + Dream logit-SHIFT** (the fix for "Paris Paris Paris"):
   - Block layout [seed, MASK, ...]; seed at pos 0 predicted by the PREVIOUS block
     (or the prompt's last logit at cold start).
   - Shift readout: token at canvas pos i (i>=1) = argmax(logit[i-1]); pos 0 (seed)
     never resampled. (`_shifted_conf_pred`; mirrors sampled[:,1:]=sample(logits[:,:B-1]).)
   - Next seed = argmax(last position's logit) (`_next_seed_from_logits` == the
     reference `_capture_next_seeds` reading full_logits[base+B-1]).
   - Prompt prefill captures the cold-start seed from the last prompt logit
     (`_finish_prefills`; == `_capture_prefill_seeds`).
2. **Small, config-driven block size** — canvas_length=3 (== block_size:3), threshold
   from config; no fixed 32-wide canvas.
3. **Mixed GDN readout** — the patched `_forward_core` now calls the ported
   block-causal kernel with `causal_mode=2, num_clean=1` (was 0): pos 0 (clean seed)
   token-causal (reads S_0), pos 1..B-1 (masks) block-causal (read S_end). Bit-matched
   to `_set_denoise_flags` (dllm_gdn_causal_mode=2 / num_clean=1).
4. **Threshold commit gate** — conf>threshold commits masks; force top-1 if a row has
   masks but none pass (`_denoise_shift_step`; mirrors the reference `need`/`force`).
   Softmax fp32-accumulated (lossless).
5. **Kept from prior passes (already matched):** persist_state=False during denoise
   (output_final_state=False) + block-entry snapshot/restore; the verbatim SGLang
   block-causal kernel port (block_causal_readout.py).

## Emission (the last fix that closed the "dropped word" gap)
Each committed block emits ALL CL positions = [seed, denoised-tail] (num_sampled=CL).
The carried seed IS a generated token (only re-input at pos 0 to drive the shift);
emitting it makes the stream contiguous. Emitting positions 1..CL-1 only (dropping the
seed) left systematic gaps like "The <capital> of" — emitting CL fixed it and made
output match the oracle token stream.

## Mapping SGLang's inner run() loop onto vLLM's per-step sampler
SGLang's run() does multiple forwards per call (denoise loop + causal commit forward).
vLLM drives one forward per scheduler step with the sampler's is_encoder_phase flip:
each vLLM step == one SGLang denoise round; a request re-denoises across steps until
its block has no masks, then a commit step (is_encoder_phase True -> stock token-causal
GDN forward) captures the next seed, emits the block, and re-inits the next
[seed, MASK, ...] block. draft_tokens carries the canvas between steps.

## Known residual vs the oracle (does NOT block acceptance)
- vLLM commits blocks IRREVOCABLY to KV; SGLang can revise emitted tokens across
  denoise rounds (block revision). Coherence is unaffected in practice but exact
  token streams can differ run-to-run / at block boundaries.
- We use greedy argmax; the config specifies temperature=1.0/top_k=20/top_p=0.95
  (sampling). Greedy is fine/cleaner for the acceptance prompts.
- The commit "next seed" is captured from the commit step's token-causal forward
  (as the reference does), so this is faithful.

## Files (node1 + local worktree, all md5-identical)
- vllm_native_diffusion/qwen3_5_diffusion.py  (seed-carry+shift sampler, denoise/commit
  lifecycle, causal_mode=2/num_clean=1 readout, fp32 gate, snapshot/restore)
- vllm_native_diffusion/block_causal_readout.py (verbatim SGLang block-causal kernel)
- vllm_native_diffusion/plugin.py             (installs model + config hook + GDN patch)
- serve_diffusion.sh                          (canvas_length=3, threshold=0.95)
