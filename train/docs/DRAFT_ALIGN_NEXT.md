# Draft-align: what to run next (written 2026-09-11, before the 1000-step evals landed)

## The decisive fact from run 3756

The diffusion loss **plateaued by step 100** and then oscillated for 900 more steps:

| step | 50 | 100 | 250 | 500 | 750 | 1000 |
|---|---|---|---|---|---|---|
| diff | 1.335 | 1.212 | 1.366 | 1.329 | 1.202 | 1.219 |
| AR | 0.098 | 0.083 | 0.088 | 0.085 | 0.074 | 0.075 |

~0.4B supervised tokens (12.9k resp tokens × 16 ranks × 2 accum × 1000 steps) bought no measurable
descent after the first ~40M. So "more steps of the same" is dead: the limit is the objective/LR/capacity,
not the token budget. This is consistent with the pilot (flat acceptance) and with the offline per-slot
diagnostic (slot-1 agreement ~76%, the ceiling is the shared weights, not vLLM canvas fidelity).

## Decision table (fill in when evals 3757-3768 + slotdiag 3769 land)

| observation at step 1000 | reading | action |
|---|---|---|
| warm all-accepted ≥55% (N=4) or ≥15% (N=8), AR guard within 1 problem | it does move, just slowly | **Plan A** |
| acceptance +1-3 pts only (pilot-like), AR guard holds | LR-limited or objective-limited | **Plan B** |
| acceptance flat AND AR guard down (≤23/30) | we are trading quality for nothing | **Plan C** |

## Plan A — scale the run that works (≈14 h, 2 nodes)
Same recipe, 3000 steps, LR 5e-6 → cosine to 5e-7, save every 500, fresh optimizer from step_18000.
Only worth it if acceptance actually moved; the loss curve says it will not.

## Plan B — attack the plateau, cheap A/B first (≈2 h, 2 nodes, 4 × 300 steps)
Four 300-step probes, evaluated on tok/fwd + slot-1 agreement, run sequentially on both nodes:
1. **LR 2e-5** (FLARE used 1e-5 constant; ours is 5e-6 — likely too timid for a real distribution shift).
2. **Slot-weighted loss**: weight the diffusion CE by draft distance, heaviest on slot 1 (the slot that
   sets tok/fwd; today every slot in the suffix contributes equally and the far slots are unlearnable).
3. **Mask distribution matched to the DECODER's real state**: today P(m=bd-1)=0.5 cold + uniform; the traces
   say the true mix is 34% cold / 66% warm at N=4 and 48/52 at N=8 → sample m from the measured distribution.
4. **bd 4 instead of 8** (the shipped config is N=4; training at bd 8 spends half the capacity on slots the
   N=4 decoder never uses).
Pick the winner, then scale it with Plan A's budget.

## Plan C — separate drafter (the DFlash shape), ≈2 days
Freeze the 4B, train a small (4-6 layer) block-diffusion drafter on its greedy outputs. Evidence it works:
`z-lab/Qwen3.5-4B-DFlash` reaches 6.5 accepted of 16 on stock Qwen3.5-4B (vLLM) / 6.25 on SGLang, versus our
1.3 of 3. Costs: a new model to train, serve and ship; but it is the only route with a demonstrated 2-3x.
Prerequisite work: drafter architecture choice (reuse DFlash's), data is the same self-distilled corpus.

## Regardless of branch — do these first (cheap, they de-risk everything after)
- **Fix `max_packed_rows=1`**: the collator drops every conversation that does not fit row 1. With
  `PACK_EXAMPLES=20` that is most of the pull, i.e. we are training on a biased subsample (short convs).
  Carry leftovers to the next step. This alone may explain part of the plateau.
- **Add the acceptance metric to the training loop** (slot-1..k argmax agreement with the clean head, from
  the same forward, free): today we only learn the answer 40 min after a run ends.

## 2026-09-11: literature check + the AUF probe

Search turned up a paper on exactly our failure mode:

- **Spec-AUF, "Accept-Until-Fail Training under Train-Inference Misalignment for Masked Block Drafters"**
  (arXiv 2607.01893). Masked block drafters are trained with uniform CE over masked positions, but inference
  accepts left-to-right and stops at the first miss. They reweight each position by whether the preceding
  positions were accepted. This is precisely what our data showed: +4-7 pts per slot bought 0 acceptance,
  because acceptance is a conjunction and our slots are strongly correlated (histogram is bimodal:
  43% accept nothing, 34% accept everything).
- **DFlare** (arXiv 2606.02091): scale draft capacity, condition the drafter on the TARGET model's hidden
  features; explicitly beats shared-weight drafting. **DFlash** (arXiv 2602.06036): >6x lossless, 2.5x over EAGLE-3.
- **Cost-aware diffusion draft trees** (arXiv 2606.01813): verify several candidate continuations per forward —
  structurally sidesteps the conjunction, and directly targets our 43%-accept-nothing bucket.
- **Block verification for speculative diffusions** (arXiv 2606.13426): training-free, but +6.3% and aimed at
  sampling, not our greedy path. Skipped.

### Implemented: AUF weighting (`--auf_floor`, `HFBlockDiffusionHybrid(auf_floor=...)`)
Per masked run, weight slot k by (detached) "every predecessor in this run was predicted correctly", floored at
`auf_floor` (0.1) so later slots keep some gradient. Folded into the existing `rate` path (`_diff_ce` uses
w = 1/rate, normalised), so no change to the loss helper. Verified on CPU incl. multi-run base propagation.
Logs `auf` = fraction of masked slots whose predecessors all survived (a live proxy for draft survival).

### Also measured today (before the probes)
- The training metric reads slot-1 at **0.93** on our agentic corpus but the offline diagnostic reads **0.79**
  on GSM8K for the same model and mask shape → **our distillation corpus is much easier than our eval**.
- vLLM's fully-causal canvas costs 30-46% of JOINT acceptance vs bidirectional (0.179 vs 0.232 at N=4 tuned);
  same weights give 2.74 tok/fwd on SGLang vs 2.31 on vLLM. Engine fix = free ~30%, no training.

### Running: minimal paired probe (100 steps each, 1 node each, identical except AUF)
`probe-a-base` (3814) vs `probe-d-auf` (3813, `AUF_FLOOR=0.1`); both BD=7 PCOLD=0.34 MWARM=3 (deploy-matched
canvas for N=4), LR 5e-6, ACCUM=4. Compare with `slot_diag.sbatch` (fixed protocol, both checkpoints + baseline)
on the CAUSAL-cold column and its slot product = the joint the decoder actually needs.
