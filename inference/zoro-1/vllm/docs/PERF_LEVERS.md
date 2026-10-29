# Trida Diffusion Decode — Per-Forward Cost & Optimization Levers

**Status:** IMPLEMENTED — coherent diffusion serve + de-sync + PIECEWISE cuda-graph shipped (**32 tok/s**, `c008ce1`). FULL cuda-graph attempted & rejected (run-both regression); the single-kernel FULL path is documented future work (see "Results" below).
**Owner:** vLLM native diffusion port (`feat/vllm-native-diffusion`).
**Date:** 2026-09-08.

---

## 0. TL;DR

At **concurrency 1**, our block-diffusion decode only reaches **parity** with AR (it should, in theory, dominate). hyungguk's server-counter decomposition:

> Diffusion commits **4.30 tokens/forward** but each denoise forward costs **~4× an AR decode forward** — full fp32 logits over 31 window positions × 248k vocab, top-k/p sorts, a host sync per round, and a non-graphed commit forward per block. **4.3 ÷ 4 ≈ 1** → parity.

This doc pins down that 4× and lists concrete levers to cut it, each cross-checked against open-source implementations. Cut the per-forward cost from ~4× to ~2× and diffusion becomes ~2× AR at C=1, as theory predicts.

---

## RESULTS — what shipped (2026-09-08)

Implemented on `feat/vllm-native-diffusion` (C=1, canvas=3 / b3, greedy, robe 128-tok, best-of-3):

| stage | tok/s | syncs/fwd | coherence | commit |
|---|---|---|---|---|
| coherent diffusion port | — | ~13.2 | ✓ | `bc14b97` |
| + de-sync decode path | 17.7 (eager) | **2.1** | byte-identical | `4817dc1` |
| + **PIECEWISE cuda-graph** | **32** | 2.1 | byte-identical | `c008ce1` |
| SGLang b3 (reference) | 54.5 | — | ✓ | — |

- **De-sync**: replaced 3 `if mask.any(): nonzero → index_copy_` patterns with fixed-shape masked `torch.where` (`aten::nonzero` 219→0, `index_copy_` 4464→0). Prerequisite for cuda-graph; **~no C=1 speed gain on its own** — the denoise loop is serial (round N+1 needs round N's commits), so removing syncs recovers no overlap at C=1.
- **PIECEWISE cuda-graph** = 1.85× over eager, byte-identical. Shipped.
- Still **~60% of SGLang** (32 vs 54.5): under PIECEWISE the 24 GDN layers run **eager** (they're `splitting_ops`); SGLang graphs its denoise GDN kernel end-to-end.
- **The bigger picture** (from full profiling, AR/bd4/self-spec/bd32): GEMM ms/forward is flat (~4ms, one weight read) across block sizes — no bug. It's a **tokens-per-forward race**: `ms/token = per-forward-cost ÷ tok/fwd`. b3 is structurally slow (1 commit forward per ~2 tokens); **bigger blocks (bd32) are ~2–4× faster** by amortizing — a separate lever from cuda-graph.

## FULL cuda-graph — attempted, REJECTED (run-both); the real path is future work

- **Correction to the earlier premise:** the GDN readout is a vLLM `splitting_op`, so it **already runs eager even under FULL** — it was **never** the graph blocker. FULL *does* capture: GDN backend caps FULL→**FULL_AND_PIECEWISE** (`UNIFORM_SINGLE_TOKEN_DECODE`), and our 3-token block = `1 + 2 spec tokens` qualifies as uniform decode → 1 FULL + 4 PIECEWISE graphs, no crash.
- **run-both approach = REJECTED** (stashed, NOT committed — `git stash`, md5 `37c947f`): running both readouts every step + mask-select does **2× GDN work** → halves PIECEWISE (32→**15.5**), and is **not byte-identical** (the extra bf16 readout perturbs the recurrent state → diverges on step 1). FULL run-both = 61 tok/s but **degenerate/looping**. Root cause: the two GDN kernels (block-causal denoise `causal_mode=2` vs token-causal commit) are **not bf16-identical**, so run-both can't be lossless.
- **The real FULL path (to pursue):** a **single** GDN readout kernel that branches on a device `causal_mode`/`num_clean` tensor (persistent in-place buffer, `diffusion_gemma._causal_buf` pattern) — NOT run-both. **Gate:** first prove on CPU that `block_causal_readout.py` at token-causal mode reproduces the stock commit kernel **bit-for-bit (bf16)**; only then does one FULL graph replay correctly for both phases. Pre-perturbation FULL headroom was ~61–79 tok/s → would beat SGLang's 54.5.
- Refs: [vLLM CUDA Graphs design doc](https://docs.vllm.ai/en/stable/design/cuda_graphs/); GDN cuda-graph is bleeding-edge (batch-invariant rejected for `GDN_ATTN`, `selector.py:154`, [#42960](https://github.com/vllm-project/vllm/issues/42960); [PR #34571](https://github.com/vllm-project/vllm/pull/34571)).

---

## 1. The measured picture (single GPU, low concurrency)

hyungguk's table (100 prompts across 7 benchmarks, thinking-on, max 4096 tok):

| Decoding | C=1 tok/s | C=1 p50 | C=4 tok/s | C=4 per-req | scaling 1→4 | tokens/forward |
|---|---|---|---|---|---|---|
| AR | 222 | 4.7s | **796** | **202** | 3.6× | 1.0 |
| Self-spec b15_g8 | 209 | 4.1s | 642 | 148 | 3.1× | ~2.1 |
| Diffusion bd=32 | **239** | 4.1s | 510 | 108 | 2.1× | 4.30 |
| Diffusion bd=4 | 57 | 12.1s | 204 | 51 | 3.6× | 1.75 |

- **C=1: parity.** Diffusion bd=32 (239) barely edges AR (222); self-spec (209) is just under. Theory says diffusion should *dominate* at low concurrency — it doesn't.
- **C=4: AR wins everything** (796 vs 642 vs 510) — batching amortizes AR's single matmul; diffusion's per-position fixed costs scale with positions × batch and don't amortize.

Independent cross-check (my SGLang measurement, gsm8k C=1, step_18000_sglang): self-spec ~210 ≈ his 209; AR cuda-graph ~214 ≈ his 222. **AR eager (cuda-graph off) = 34.6 tok/s** → AR's competitiveness is itself 6.2× dependent on cuda-graph.

---

## 2. Per-forward cost model (concrete)

Model config: **vocab = 248,078**, **hidden = 2560**, **tied embeddings**, **32 layers**, **bd_size = 32** (→ 31 masked canvas positions scored per forward), hybrid attention (**¾ GDN gated-delta recurrent + ¼ full-attention KV**).

**AR forward — scores 1 position:**
- `hidden(1×2560) @ embedᵀ(2560×248k)` → 1×248k logits → argmax/sample.
- Bottleneck at bs=1 is **memory-bound**: read the tied embed matrix (248k×2560×2B ≈ **1.27 GB**) once. Logit tensor ~1 MB.

**Diffusion forward — scores all 31 masked positions:**
- `hidden(31×2560) @ embedᵀ` → **31×248k = 7.69M logits**.
- Materialized in **fp32**: 31×248k×4B = **30.7 MB** (31× AR's tensor, 2× the bytes of bf16).
- **fp32 softmax** over 7.69M values (confidence signal).
- **top-k/p sort** over 248k × 31 rows (31 full-vocab sorts).
- **host sync per round** (read commit decision to CPU) — matches the profiled **73% GPU-idle**.
- **non-graphed commit forward per block** — diffusion hardcodes `can_run_cuda_graph=False` (`.../dllm/algorithm/low_confidence_shift_hybrid_diffusion.py:282`); AR captures cuda-graph.
- (likely) **self-conditioning**: `probs @ embed` feeds the committed distribution back → a *second* full read of the 1.27 GB tied matrix per forward. *(unconfirmed — flagged for code check.)*

→ ~4× the wall-time of one AR forward. **Correction (from precision research):** the 30 MB fp32 *materialization* is trivial memory — it is NOT a dominant cost. The real per-forward buckets are: the **31× sort** (removable, lever 3), the fp32 **softmax compute** over 7.69M values (keep, needed for a lossless gate), the per-round **host sync** (lever 5), the **non-graphed** commit forward (lever 5), and (if present) **self-conditioning**'s second embed read.

**Why most of it is wasted:** only ~4.3 of 31 positions commit per forward; the rest stay masked and are **re-scored next round**. A block takes ~32/4.3 ≈ **7–8 forwards**, so a given position's full-vocab fp32 logits are recomputed **~7×** before it commits.

---

## 3. Optimization levers

Each lever below has: **the idea**, **full logic**, **OSS references** (filled in by research subagents), **applicability to our model**, **expected gain**, **risk/losslessness**.

### Lever 1 — fp32-**accumulated** softmax (NOT full bf16); drop the redundant fp32 logits *copy*
- **Correction from research:** the "30 MB fp32 logits" materialization is **trivial** (31×248k×4B ≈ 30 MB/forward) — not a real memory cost. So bf16-for-memory is a **small** lever here, and a **risky** one. Revised guidance below.
- **What's actually correct:** keep logits in bf16 storage but force the **softmax reduction in fp32** via `F.softmax(logits, dim=-1, dtype=torch.float32)`, then threshold-compare in fp32. Summing ~248k exps in bf16 loses low-order bits (catastrophic accumulation) → shifts max-prob → flips the gate. fp32 accumulation is value-preserving vs an fp32 reference (bf16 ⊂ fp32) and is the field standard.
- **Losslessness:** a full-**bf16** gate (LLaDA/Dream/SDAR do this) is **NOT provably lossless** — borderline max-prob flips at the cutoff. The fast/production diffusion stacks upcast the gate: **Fast-dLLM → fp64**, **dInfer (Ant, prod) → fp32**, **vLLM/HF → fp32** full-vocab. fp32-accumulated softmax + fp32 threshold = *effectively lossless*.
- **OSS references:** §5.2.
- **Action for our port:** the current `custom_sampler` does softmax in **native bf16** → change to `dtype=torch.float32` accumulation for a lossless gate (one-line, ~free; avoids a redundant full-fp32 logits copy but does NOT store fp32 logits). Optionally run a commit-flip A/B (bf16 vs fp32 gate) for a tech-report-grade losslessness claim.
- **Expected gain:** ~none for speed — this is a **correctness** fix. Real per-forward speed comes from levers 3 + 5.

### Lever 2 — score only still-masked positions
- **Idea:** run the expensive head (`lm_head → softmax → gate`) on only the positions still masked this round, not all 31.
- **Full logic:** maintain a per-position commit mask over the 32-block. Backbone still runs over the full block (committed tokens supply context via attention / GDN state), but before the head, **gather only still-masked hidden states** and score those. As the block fills (31→24→18…), head cost shrinks each round instead of staying at 31. Committed positions are frozen (never re-scored).
- **OSS references:** _(research-masked subagent — pending)_
- **Applicability:** head-only gating is architecture-agnostic; caching committed context is nuanced for us (KV cache on the ¼ full-attn layers; snapshot/restore recurrent state on the ¾ GDN layers).
- **Expected gain:** roughly ½ the total head work over a block (triangular shrink vs 31×rounds), *if* the current path scores all 31.
- **Risk:** low for correctness (frozen tokens are final); need to confirm current code doesn't already gate.

### Lever 3 — commit gate = argmax + threshold (drop the full sort)
- **Idea:** the gate needs "is max-prob ≥ threshold?", which is a single **max-reduction (argmax + max value)**, not a full top-k/p **sort** of 248k values.
- **Full logic:** per position, one `O(n)` max pass gives the top-1 token and its prob; compare to threshold; commit or keep masked; always commit the global top-1. Replaces `O(n log n)` HBM-heavy sort over 248k × 31. **At temperature 0 the committed token *is* the argmax → provably lossless.** For temp>0, use a small partial-top-k instead of a full sort.
- **OSS references:** Fast-dLLM `get_transfer_index` threshold branch = our gate *verbatim* (no sort); FlashInfer `SamplingFromLogitsKernel` = fused argmax; vLLM `Sampler.sample()` short-circuits to `argmax` before any sort at temp 0. Full logic + verified code in **§5.1**.
- **Applicability:** our eval is temp 0 → exact. **Already applied in the port's `custom_sampler`** (top-1 + threshold, no sort).
- **Expected gain:** removes the entire sort bucket of the 4×.
- **Risk:** none at temp 0; define temp>0 path explicitly.

### Lever 4 — fused logit → softmax → threshold kernel
- **Idea:** one kernel computes per-position logits, running-max (online softmax), and threshold, keeping the 248k row in registers/shared memory — the 30 MB fp32 tensor never lands in HBM.
- **Full logic:** streaming/online softmax (running max + running sum) yields max-prob without materializing the full probability vector; only the tiny result (committed token id + prob, or "stay masked") is written. Combines levers 1+3 into one pass.
- **OSS references:** FlashInfer dual-pivot **rejection sampling** (sort-free top-p/top-k) + `GetMaxValue`/online-softmax (max-prob without an HBM prob vector). Full logic in **§5.1**.
- **Applicability:** biggest engineering lift (custom kernel); do after 1+3 prove the gain.
- **Expected gain:** eliminates repeated 30 MB HBM round-trips (materialize → read → softmax → read → gate).
- **Risk:** kernel correctness; validate against the reference gate.

### Lever 5 — cuda-graph the diffusion forward — ⚠️ PORT-PARITY, not a beat-AR lever
- **Correction (measured 2026-09-08):** SGLang **already cuda-graphs** the diffusion denoise forwards. Measured on a GPU node, diffusion bd=32 C=1: **cuda-graph ~116 tok/s vs EAGER ~30** (~4×); AR **214 vs 34.6** (~6×). So cuda-graph is NOT a missing lever in SGLang. Earlier belief that "diffusion runs eager (`can_run_cuda_graph=False`)" was wrong — that line is an initializer overwritten by `out.can_run_graph`; the "73% idle" profile was a **torch-profiling artifact** (profiling forces cuda-graph off).
- **Where it still matters — the vLLM PORT:** our port currently runs `--enforce-eager`, so it'll be ~30 tok/s-class until graphed. cuda-graph is required to **match** SGLang, not to beat AR.
- **Full logic:** the vLLM native `ModelState.prepare_attn` accepts `cudagraph_mode` / `CUDAGraphMode.FULL`; the diffusion_gemma reference shows the path (padded shapes + persistent buffers). Only the **per-block commit/persist forward** is genuinely non-graphed in SGLang (one of the four per-forward costs).
- **Applicability:** requires stable shapes across the denoise loop (the hard part — variable commit counts). The vLLM port is the vehicle.
- **Requirements (verified from `diffusion_gemma` reference):** (1) FULL cuda-graph mode uses **padded** shapes `num_reqs_after_padding`/`num_tokens_after_padding` → fixed per-forward dims (dodges the dynamic-shape blocker SGLang hit); (2) all inputs in **persistent, stable-address buffers** (our scaffold already has `_inputs_embeds_buf` + the `CUDAGraphMode.FULL` branch in `prepare_attn`); (3) override **`get_additional_cg_support()`** to declare support; (4) the denoise loop stays **host-side control flow**, each forward = one graph replay. Our TODO: make the **GDN snapshot/restore + block-end-readout buffers stable-address & host-sync-free** (the agent's on-device `index_copy_` is the right base — no host sync also kills the "host sync per round" cost), pad canvas to `block_size`, and capture the **persist forward** (`persist_state=True`) separately or fold it in.
- **Sequencing:** lands AFTER the readout correctness fix + commit (same file; can't graph a broken/changing forward).
- **Risk:** the persist-forward mode + variable denoise-round count; mitigate by fixing per-forward shape and keeping the round-loop on the host.

---

## 4. Prioritization & experiment plan

1. **Verify first (no code change):** read hyungguk's diffusion sampler to confirm which levers are *already* applied (does it gate masked positions? sort or argmax? fp32 forced where?). Avoid chasing an applied lever.
2. **Lever 3 (argmax, temp 0)** — lossless, smallest change, kills the sort bucket. Land in the port's `custom_sampler`.
3. **Lever 1 (bf16)** — measure commit parity vs fp32; halve logit/softmax traffic.
4. **Lever 2 (masked-only head)** — ~½ head work per block.
5. **Lever 4 (fused kernel)** and **Lever 5 (cuda-graph)** — larger lifts, after 1–3 quantify the wins.

**Each lever's acceptance test:** output-token parity + acceptance-rate parity vs the fp32/sort baseline on a fixed prompt set, plus C=1 tok/s delta.

---

## 5. Open-source research appendix

### 5.1 Commit gate & fused sampling (levers 3 + 4) — byte-verified

**Our gate exists in the wild, verbatim — Fast-dLLM's threshold branch.**
`NVlabs/Fast-dLLM` · `v1/llada/generate.py` · `get_transfer_index` (byte-verified via GitHub API):
```python
x0 = torch.argmax(logits_with_noise, dim=-1)                       # committed token = argmax
p  = F.softmax(logits.to(torch.float64), dim=-1)
x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)           # max-prob of argmax
confidence = torch.where(mask_index, x0_p, neg_inf)
if threshold is not None:                                          # ← NO sort, NO topk
    transfer_index = mask_index & (confidence >= threshold)
    max_conf = torch.argmax(confidence, dim=1, keepdim=True)       # "always unmask max c^i"
    transfer_index = (transfer_index | scatter(max_conf)) & mask_index
    return x0, transfer_index
# else: LLaDA top-k-count path → torch.sort  (the removable work)
```
- Confidence = **max softmax prob of the argmax token** (not entropy). "Commit ≥1" = unconditional OR with the single `argmax(confidence)` position. `add_gumbel_noise` is a **no-op at temperature 0**, so `x0 = argmax(logits)` exactly.
- **Diagnosis for us:** a full `torch.sort`/`topk` only lives in the `threshold is None` (top-k count) branch, or in a top-k/top-p filter applied *before* the gate. If our SGLang path sorts, that's a filter we can delete — the threshold gate itself needs no sort.
- Fast-dLLM's SGLang production gate (`third_party/sglang/.../dllm/algorithm/low_confidence.py`, default threshold 0.95) is the same shape, with `topk(k=1)` used *only* as an empty-set fallback. *(read via WebFetch, not byte-verified.)*
- LLaDA (`ML-GSAI/LLaDA/generate.py`) instead unmasks a **scheduled count** k per step via `torch.topk` (a partial sort) — evidence that the *count-based* schedule is what forces a sort; the *threshold* schedule (ours) does not.

**FlashInfer sort-free sampling** (`flashinfer-ai/flashinfer` · `include/flashinfer/sampling.cuh`, byte-verified):
- **Greedy/argmax fast path — `SamplingFromLogitsKernel`:** one `BlockReduce` over the vocab returns the argmax index; no pivot loop, no sort. At temp 0 this *is* our committed token.
- **Dual-pivot rejection sampling** (top-p/top-k without sorting): maintain a cutoff interval `[low,high]`; each round pick two pivots, count tokens above each in **one streaming reduction pass** (no HBM write of a filtered vector); interval halves → `O(log(1/ε))` rounds, all fused in one launch. Needed only for temp>0. "Stochastically equivalent," not bit-identical to sort-based top-p.
- **Fused max-prob (lever 4):** `GetMaxValue` = running-max via one `BlockReduce`; online-softmax blocks track `(m,l)` (running max + denominator) flash-attention style. **max-prob = `1/l` after subtracting `m`** — a single streaming pass, the full 248k prob vector never touches HBM. `*_from_logits` variants fuse this softmax into the sampler.

**vLLM / SGLang dispatch:**
- vLLM `Sampler.sample()` computes `greedy_sampled = argmax(logits)` and returns immediately when `all_greedy` — **never enters `TopKTopPSampler`, so no sort at temp 0**. `TopKTopPSampler.forward_cuda` uses FlashInfer rejection sampling; `forward_native` is the slow `logits.sort(...)` fallback. *(docs-verified, not source-byte-verified.)*

**Verdict (temp 0): lossless.** Committed token = `argmax(logits)`; gate needs only `max_prob = 1/l` from the streaming `(m,l)` pass. Drop any pre-gate top-k/top-p sort; adopt Fast-dLLM's threshold branch (we already do in `custom_sampler`); optionally fuse the max-prob via FlashInfer `GetMaxValue`/`SamplingFromLogitsKernel`. **temp>0** needs FlashInfer rejection sampling (`*_from_logits`) — sort-free but only stochastically equivalent.
**Cost:** replaces an `O(V log V)` sort over V=248k × 31 positions/forward with an `O(V)` streaming reduction (temp 0). Algorithmic claim; not benchmarked on our shape.

**Sources:** Fast-dLLM [repo](https://github.com/NVlabs/Fast-dLLM) · [get_transfer_index](https://github.com/NVlabs/Fast-dLLM/blob/main/v1/llada/generate.py) · [SGLang low_confidence](https://github.com/NVlabs/Fast-dLLM/blob/main/third_party/sglang/python/sglang/srt/dllm/algorithm/low_confidence.py) · [paper](https://arxiv.org/abs/2505.22618); LLaDA [generate.py](https://github.com/ML-GSAI/LLaDA/blob/main/generate.py); FlashInfer [sampling.cuh](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/sampling.cuh) · [sorting-free blog](https://flashinfer.ai/2025/03/10/sampling.html); vLLM [Sampler](https://docs.vllm.ai/en/stable/api/vllm/v1/sample/sampler/) · [TopKTopPSampler](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/ops/topk_topp_sampler/).
_Verification: §1a/1b/2 (argmax, get_transfer_index, pivot loop, GetMaxValue, SamplingFromLogitsKernel) byte-verified via gh API; §1c/§3 read via WebFetch/docs. FlashInfer line numbers track `main`; pinned 0.27.x may differ in signatures, not algorithm._

### 5.2 Precision — bf16 vs fp32 logits/softmax (lever 1) — verified

**Precision map across stacks (verified code):**

| Stack | softmax dtype | gate/decision | full-vocab fp32 copy? |
|---|---|---|---|
| vLLM v1 | fp32 (whole tensor upcast) | argmax/log_softmax fp32 | yes |
| HF `transformers` generate | fp32 (upcast at slice) | softmax/argmax fp32 | yes |
| FlashInfer | native storage, **fp32 online-softmax** accumulation | rejection sampling fp32 | **no** (streaming) |
| LLaDA / Dream / SDAR | **native bf16** | bf16 max-prob gate | no |
| **Fast-dLLM v1** | **fp64** for the confidence gate | `conf ≥ thr` in fp64 | yes → fp64 |
| **dInfer** (Ant, production) | **fp32** for the gate | `conf > thr` in fp32 | yes → fp32 |
| BD3-LM | fp32 (`autocast(fp32)`) + log_softmax | Gumbel-argmax fp32 | yes |

**The load-bearing fact:** the model matmul→logits stays bf16 everywhere; the split is on the **softmax/exp reduction**. Summing ~248k exps in bf16 loses low-order bits (catastrophic accumulation) → shifts max-prob → **flips the commit gate**. argmax (which token) is bf16-robust; **max-prob ≥ threshold is the fragile part** near the cutoff.

**Recommendation:** `F.softmax(logits, dim=-1, dtype=torch.float32)` — keeps `logits` bf16 in memory but forces the exp/normalize in fp32 (identical effect to vLLM's `dtype=torch.float32` and dInfer's `.to(fp32)`, without a redundant full-vocab fp32 copy). Threshold-compare in fp32. This is *effectively lossless* vs an fp32 reference. A full-**bf16** gate (LLaDA/Dream/SDAR) is **not provably lossless**; Fast-dLLM (fp64) and dInfer (fp32) both upcast for exactly this reason. For temp>0 Gumbel, upcast to fp64 (arXiv:2409.02908 shows 32-bit Gumbel-max is biased). **No repo publishes a bf16-vs-fp32 commit-flip A/B** — run our own for a tech-report-grade losslessness claim.

**Sources:** [vLLM sampler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py) · [HF `_sample`](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py) · [FlashInfer sampling.cuh](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/sampling.cuh) · [LLaDA](https://github.com/ML-GSAI/LLaDA/blob/main/generate.py) · [Fast-dLLM v1](https://github.com/NVlabs/Fast-dLLM/blob/main/v1/llada/generate.py) · [dInfer parallel_strategy](https://github.com/inclusionAI/dInfer/blob/master/python/dinfer/decoding/parallel_strategy.py) · [BD3-LM](https://github.com/kuleshov-group/bd3lms/blob/main/diffusion.py) · [Gumbel precision arXiv:2409.02908](https://arxiv.org/abs/2409.02908).

### 5.3 Masked-only scoring & caching (lever 2) — verified

**Two architecturally-separate optimizations:**

**(a) Restrict the head to masked positions.** *Verified finding: no surveyed repo gathers before the `lm_head`* — they slice after `.logits`. Dream is closest: `mask_logits = logits[mask_index]` (`generation_utils.py::_sample`), but the 248k matmul already ran on all positions. **Our win:** move the gather one op earlier — `h_masked = h[mask_index]; logits = lm_head(h_masked)`. Lossless (committed positions can't re-commit), ~10 lines, **applies to both GDN and full-attn layers** (it's downstream of attention). Over ~7 rounds on a 32-block the masked count decays 31→0 (avg ~15–16) → **~2× fewer head FLOPs** + skip softmax/gate on committed rows. Caveat: with token-shift (predict `i` from hidden `i−1`), gather at `i−1`.

**(b) Cache committed context across rounds** — splits by layer type:
- **Full-attn (¼ of layers) → KV cache.** Safest/lossless: **BD3-LM / SDAR block-boundary commit** — store a block's K/V only once the whole block commits, reuse as read-only prefix. More aggressive: **Fast-dLLM dual-cache** (prefix+suffix warmed once/block) or **dKV-Cache** (delayed per-token KV, one-step commit delay + periodic full reload, and it also shrinks the *trunk* input to still-masked rows `x[~prv_transfer_idx]` → 2–10×). **dLLM-Cache** = training-free feature caching (K/V/AttnOut/FFNOut, cosine-sim drift refresh, up to 9.1×).
- **GDN / gated-delta (¾ of layers) → NOT KV; snapshot/restore recurrent state.** ⚠️ **No open-source precedent** — every surveyed diffusion-LM cache is pure-attention. GDN carries a recurrent state matrix `S`, not per-position K/V, so you can't "slice the prefix." **What ports (from BD3-LM's block-boundary idea):** when a block commits, roll the recurrence forward over the committed block **once** to advance `S`, **snapshot `S`**; each denoise round of the next block **restores** the snapshot and re-scans only the current 32-block. **What does NOT port:** intra-block per-position KV reuse (linear-attn state is order-dependent, not per-position patchable) — within a block, GDN layers must re-scan. **Correctness:** snapshot at the *true* commit point — **after** the block-end readout (matches our FLARE two-stream, [[two-stream-gdn-flare-logic]]); **must bit-exactness-test vs a no-cache run.** This is exactly the port's current `prepare_attn` GDN two-stream fix.

**For our hybrid, in order:** (a) head-gather first (lossless, both layer types) → KV-cache the ¼ full-attn (BD3-LM block-boundary) → snapshot/restore the ¾ GDN (the fix in flight). dLLM-Cache feature-caching is the architecture-agnostic fallback if GDN snapshot/restore proves fiddly.

**Sources:** [Fast-dLLM v1](https://github.com/NVlabs/Fast-dLLM/blob/main/v1/llada/generate.py) · [Fast-dLLM v2 paper](https://arxiv.org/abs/2509.26328) · [Dream](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/blob/main/generation_utils.py) · [dKV-Cache](https://github.com/horseee/dkv-cache) · [dLLM-Cache](https://github.com/maomaocun/dLLM-cache) · [BD3-LM](https://github.com/kuleshov-group/bd3lms/blob/main/diffusion.py) · [SDAR](https://arxiv.org/html/2510.06303v1). _Flag: SDAR/LLaDA-2 from paper text not code; GDN snapshot path has no published precedent — prototype + bit-exactness test._
