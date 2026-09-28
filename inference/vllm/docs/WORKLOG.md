# Trida diffusion — "faster than AR" work log

Append-only log. Config fixed: **block size 4** (vLLM `canvas_length=4` == SGLang `block_size 3`), **threshold 0.9**, greedy.
Reference implementation: SGLang `LowConfidenceShiftHybridDiffusion`. Harness: GSM8K, chat template, nothink, "answer after ####", 1024 tok, first-####/last-number extraction.

## Goal
Diffusion decoding faster than AR at bd4/thr0.9 with quality intact. "Slower than AR" is treated as a defect to root-cause, not an inherent property.

## Baseline (2026-09-09, node1, n=200 GSM8K, validated harness: causal 87-89%)
| config | acc | TPS |
|---|---:|---:|
| SGLang causal | 89.0% | 220 |
| SGLang bd4 | 87.0% | 57 |
| vLLM causal | 87.5% | 212 |
| vLLM bd4 | **49.5%** | 34 |
AR forward ~4.8 ms/token. Diffusion step ~21 ms (SGLang) / ~26 ms (vLLM) at bd4 = 4-5x an AR forward although a 4-row forward should cost ~1.1-1.3x (memory-bound GEMMs).

## Defect map
- **A. vLLM boundary correctness bug** — damage scales with #block boundaries (CL=2 ~2%, CL=3 ~25%, CL=4 49.5%, CL=32 66.5%; SGLang bd32 ~70% ≈ vLLM bd32 => near-parity at big blocks). Ruled out: attention setup (forcing causal -> 36%, worse), MASK-id commits (0 in trace), gate (byte-identical to SGLang). First text divergence vs SGLang at CL=2 is at a *seed slot* (block 2) => the commit forward's last-logit at a boundary is suspect. Not the speed bug (SGLang is slow too), but required for vLLM to be usable and for its speed numbers to be meaningful.
- **B. Per-step overhead** — measured on vLLM at CL=32: step 24.1 ms = 7.3 GPU compute + ~11 ms host dispatch in the eager 24-layer GDN override (kernel itself 0.84 ms) + 3.9 ms eager 24-layer snapshot/restore + 1.4 sampler. Exists on SGLang too (21 ms step vs 4.8 AR).
- **C. Extra commit forward per block** — commits zero new tokens; ~25-30% of forwards at bd4. Fusable with the next block's first denoise.
- **D. Acceptance at thr0.9** — vLLM ~1.3 tok/fwd; SGLang bd4 true tok/fwd NOT yet measured (old "2.98" was a log-line artifact). Partly A-depressed (19% forced commits), partly calibration.

## Plan (validated with user 2026-09-09)
1. Measure SGLang bd4 TRUE tok/fwd (dllm_stats deltas) + vLLM bd4 tok/fwd, same prompts. Sets the acceptance target.
2. Fix A: AR-oracle localization — for each block boundary k, feed the diffusion's emitted prefix to the causal model (same weights); its next token MUST equal the diffusion's block-(k+1) seed if the commit is correct. First mismatch = the bug's boundary. Then dump positions/state there, hypothesize, fix, validate at bd4 (acc >= 85%, tok/fwd, TPS >= 34).
3. Fix B: fuse per-layer GDN override ops; route denoise conv to scratch and drop per-step snapshot/restore.
4. Fix C: fuse commit(k) into denoise(k+1).
5. Validate vs goal: bd4/thr0.9 tok/s vs AR on the same workload. Then same B/C on SGLang.

## Log
- 2026-09-09 ~13:10 — Plan validated. Launching step 1 (GPU0 SGLang bd4 tok/fwd, GPU1 vLLM bd4 tok/fwd) and step 2 oracle (GPU2 vLLM bd4 traced, GPU3 vLLM causal) in parallel. Results dir: /path/to/diag/s12_<ts>/.
- 2026-09-09 13:05 — **Step 1 result (run s12_20260909_125634): true tok/fwd at bd4, 5 GSM8K items, greedy 256 tok.**
  | item | SGLang bd4 (fwds, tok/fwd) | vLLM bd4 (fwds, tok/fwd) |
  | 0 | 223, 1.13 | 229, 1.12 |
  | 1 | 130, 1.18 | 169, 1.21 |
  | 2 | 210, 1.22 | 201, 1.27 |
  | 3 | 227, 1.13 | 216, 1.19 |
  | 4 | 227, 1.13 | 210, 1.22 |
  **Acceptance is IDENTICAL across backends: ~1.15 tok/fwd at bd4/thr0.9.** => acceptance is a model/calibration property, NOT depressed by the vLLM bug (D is not A-coupled). Consequence for the goal: with 1.15 tok/fwd, beating AR (4.8 ms/token) needs a diffusion step < 5.5 ms; a 4-row forward's compute alone is ~5.3-6.2 ms => engine work (B+C) reaches ~AR PARITY at bd4, not a win. Beating AR at fixed thr0.9 requires higher acceptance (calibration/training) — lowering thr was measured to cost accuracy.
- 2026-09-09 13:05 — **Step 2 result: AR-oracle seed check, vLLM bd4, item 0, 31 boundaries.** 29/31 seeds equal the causal model's next token for the same prefix. Mismatches: boundary 1->2 (exact tie ' at' 0.483 vs ' every' 0.483 — numerics, not a bug); **29->30: seed ' The' vs AR ' Subtract' (0.59; ' The' not in top-3) — REAL; 30->31: seed '.' vs AR ' at' (0.70) — REAL.** => the commit forward is right ~94% of the time; the bug is rare-but-fatal wrong seeds (unconditionally committed, off-distribution -> derails reasoning). Both real mismatches sit at/after block 29, whose first position 208 = 13x16 is a KV-page boundary (block 1, the tie, also starts a page at 96). Hypothesis to test: paged-KV slot mapping for canvas blocks starting a new 16-token page. Note CL=2's collapse (~2%) is larger than a page-only effect predicts — keep open.
- Harness note: vLLM 49.5% vs SGLang 87% at bd4 were measured with the IDENTICAL harness (g8k_v3: same prompt/extraction/greedy/max_tokens), which reproduces the known causal baseline on BOTH backends (87.5% vLLM, 89% SGLang). Live repro 4/10 + oracle mismatches confirm a real decode defect independent of harness. User asked to use the repo's official eval scripts — proposed: run lm_eval against both endpoints to remove any doubt.
- 2026-09-09 13:25 — **Decision (user): fix vLLM bd4 correctness (A) FIRST, then return to the tok/fwd knobs (#3-#8).** Knob analysis + wave plan recorded in chat; deferred.
- 2026-09-09 13:30 — **A1 launched**: AR-oracle seed check on items 0-3 at max_tokens 256 (+ item 0 at 128 and with +1/+2-token pads for a controlled page-alignment shift). Real mismatch = seed's AR prob < 0.5x AR-top prob. Record per mismatch: item, block, abs start pos, pos mod 16 (KV page=16), is-last-2-blocks, AR top-3. Reads: cluster at pos%16==0 -> H1 paged-KV slot mapping; cluster in last blocks -> H3 seed/end-of-gen; uniform -> H2 GDN state carry. GPUs 0 (traced diffusion) + 1 (causal oracle). Dir: diag/a1_<ts>/.
- 2026-09-09 13:35 — A2 prep (code read while A1 runs): the sampler reshapes `logits[:num_decode*CL].reshape(num_decode, CL, -1)` (qwen3_5_diffusion.py ~L521) with NO guard that each decode request actually has CL logits (`per_req_nlogits_np`). If vLLM ever schedules a partial canvas (< CL draft tokens, e.g. capped at max_tokens), the seed is read from the wrong "last position" and block emission misaligns -> a concrete H3 mechanism that would hit the LAST blocks — matching the two real mismatches at blocks 30/31 of 32 (max_tokens=128). A1's 256-token runs test this: if mismatches again sit in the final blocks, H3.
- 2026-09-09 13:50 — **A1 RESULT (a1_ run, 396 boundaries, items 0-3, 128/256 tok): 38 real seed mismatches = 9.6% per boundary.** H3 (end-of-generation) FALSIFIED: only 3/38 in the last 2 blocks. H1 (KV-page start) FALSIFIED: start-pos mod 16 histogram {4:9, 8:7, 12:8, 0:3, 6:4, 5:2, 14:2, ...} — no concentration at 0. (Pads didn't shift P — chat template strips trailing newlines — but items' natural P%16 = 12/5/4/14 cover alignments.) **Key pattern: every item is CLEAN for its first 11-30 blocks (44-120 tokens), then mismatches start and recur every ~2-8 blocks** (first mismatch block: item0=30, item1=24, item2=25, item3=11). Mismatches are not near-ties (e.g. ' The' vs AR ' Subtract' 0.59). => **accumulating STATE DRIFT across commits** (H2 variant): the block-by-block commit chain diverges from the from-scratch causal state, compounding until argmaxes flip. Candidates: persisted GDN recurrent state (chunk-per-block vs one-shot; bf16 round-trip per block), or attention KV written by commits. A2 = per-block state comparison: after each commit dump the GDN state row (checksum/L2) and compare with a fresh causal prefill of the same emitted prefix in the same server process; first block where they diverge = the fault.
- 2026-09-09 13:55 — Building the HTML decode visualizer (user request): block grid x slots colored by commit type/confidence, per-block round-by-round replay vs the 0.9 gate, AR-oracle badges, page-start markers, forced-rate/mismatch timeline. Data: item 0 bd4 (oracle-checked) + item 18 (stutter case).
- 2026-09-09 14:05 — **Decode visualizer published:** https://claude.ai/code/artifact/396271de-9b3d-4f8e-8cd1-223f84074757 (source: scratchpad/decode_viz.html; data: diag/viz_data.json from s12 item-0 trace + oracle, blk item-18 trace). Views: block×slot grid (type/confidence, seed oracle badges, KV-page markers), per-block round-by-round replay vs the 0.9 gate, forced-rate by slot, seed≠oracle timeline. Proposed A2 (awaiting go): per-block GDN-state comparison against a fresh causal prefill of the same emitted prefix, in the same server process, to find the first diverging block.
- 2026-09-09 14:15 — **A2 launched** (user go). Instrumentation: `TRIDA_DUMP_STATE=1` -> `[STATE]` fingerprint (per GDN layer: 4-dim random projection of ssm_state row + 2-dim of conv_state row) at the start of every step, pre-mutation (code md5 a863887). Run: item 0, bd4, 128 tok on GPU0; then fresh prefill of each emitted prefix (P+4k, k=1..31) on the same server. Compare the diffusion's block-entry fingerprint (= post-commit state at P+4k) with the reference's first-step fingerprint (= from-scratch causal state at P+4k), per layer, relative to the early-block noise floor. First divergent block = fault location; ssm vs conv tells which state; if fingerprints match but seeds differ -> attention KV side. Dir: diag/a2_<ts>/.
- 2026-09-09 14:40 — **A2 RESULT (a2_20260909_132741, item 0, bd4, 31 blocks).** Parser bugs fixed offline (regex `$` w/o MULTILINE; reference pairing must use the FIRST DECODE step dump — the prefill step also dumps, pre-forward/empty). Diffusion post-commit GDN state vs fresh causal prefill of the identical emitted prefix: **ssm rel-diff 13% at block 1, monotonically -> 51% at block 31** (bit noise would be ~1e-3). conv 25-75%, no trend. **Per-layer: GDN L0,L1,L2 = 0.00|0.00 (exact), divergence starts at GDN L3 = the first GDN layer after model layer 3 = the FIRST FULL-ATTENTION layer.** => GDN chain (initial-state carry, FlashInfer chunk prefill, persist, snapshot/restore) is CORRECT; the fault is injected by the full-attention layers in the COMMIT forward and compounds with committed blocks. Triton GDN-prefill A/B therefore predicted null (GDN exonerated) — skipped unless requested.
  **Leading hypothesis: the block's K/V written during noisy denoise rounds is not cleanly replaced by the commit (or the commit attends to stale noisy K/V of its own positions), so each block's persisted KV is contaminated; later blocks attend to it -> growing drift.** Alternatives: canvas position/RoPE or slot-mapping mismatch between denoise and commit.
  **A2b (proposed):** fingerprint the attention KV cache for the just-committed block's positions at block-(k+1) entry and compare with the reference prefill's KV for the same positions; also snapshot block 0's KV before/after block 1's denoise rounds to see if a later block's denoise overwrites earlier positions. First differing (layer, position, K vs V) pins the write/read path.
- 2026-09-09 14:55 — **A2b launched** (continuation of approved A2 branch "attention KV side"). Added `TRIDA_DUMP_KV=1`: at every step start, per attention layer, 4-dim projections of K and of V for (a) the just-committed block's positions [seqlen-2CL, seqlen-CL) and (b) fixed positions 92-95 (`TRIDA_KV_POS0`) to detect later overwrites. Code md5 23010116. Same protocol: item 0 bd4 128 tok, then fresh prefills of each emitted prefix; compare diffusion block-(k+1)-entry KV of block k vs reference first-decode KV of the same positions, per layer, K vs V; and track KV0 across the diffusion run.
- 2026-09-09 15:05 — **A2b result** (run `diag/a2_20260909_133908`, analyzer `diag/analyze_kv.py`). Attention KV cache facts, bd4/thr0.9, item 0, 32 blocks:
  - (B) No clobbering: block 0's KV (positions 92-95) is written once and then byte-constant for the rest of the run. Earlier positions are NOT overwritten by later blocks' denoise.
  - The **commit forward does write** the block's KV (KV at 92-95 changes between the last denoise dump and the block-2-entry dump, rel-diff 0.48).
  - (A) But the **persisted post-commit KV is wrong**: vs a fresh causal prefill of the identical prefix, rel-diff per attention layer is 0.1–1.5 for **both K and V** (block 1: L0 K 1.52 / V 0.24; mean K 0.44, V 0.35). Same magnitude for every block, no growth → a per-block systematic error, not drift.
  - Re-checked per-GDN-layer state divergence from the same log: GDN L0,L1,L2 ssm+conv rel-diff **exactly 0.00** at every block entry; GDN L3+ (after attention L0 = model layer 3) 0.1–1.3.
  - Contradiction to resolve: attention L0's V depends only on model layers 0-2, whose GDN states are bit-exact, so the commit forward's V for those rows should equal the prefill's. It differs by 24%. ⇒ either (i) the hidden state entering layer 3 in the commit step differs anyway (e.g. GDN readout / conv path for the commit chunk), or (ii) the written K/V rows are not all from the commit (per-position/slot problem).
  - **A2c launched** (eager server, `TRIDA_DUMP_HID=1` + per-position `[KVP]`, code md5 ce0f8477, script `diag/a2c_hid.sh`): per-decoder-layer residual-stream fingerprints of the 4 canvas rows in the commit step vs the same rows in the reference prefill, RoPE positions of those rows, and per-position K/V diffs. First differing (layer, position) decides (i) vs (ii).
- 2026-09-09 15:35 — **A2c result** (run `diag/a2c_*` eager, analyzer `diag/analyze_hid.py`). Per-decoder-layer residual-stream fingerprints of the 4 canvas rows, commit step vs reference prefill of the identical prefix, blocks 1–3:
  - RoPE positions identical (92-95 / 96-99 / 100-103). Embeddings (layer -1) identical. Layer 0 output identical to 0.1–0.9%.
  - **Layer 1 (a GDN layer, before any attention) output already differs 8–39% at all 4 positions**, growing through the stack (e.g. 2–18× at layers 16–20). Persisted K/V per position: every position wrong (no single bad slot) → hypothesis (ii) "slot/position" is dead; it is (i): the hidden state entering the attention layers is wrong.
  - Round-by-round at the SEED position of block 1: first denoise round (block entry, fresh clean state) matches reference to ≤1.1% through layer 6; **every round after the first RESTORE (den1, den2) and the commit are off 10–15% from layer 1**. Layer 0 stays exact in all rounds.
  - ⇒ **The GDN state a layer reads after a snapshot-restore is not the block-entry state** (conv and/or ssm, at layers ≥1). Kernel-numerics hypothesis (chunk-with-initial-state vs from-scratch) is dead: den0 uses the same kernels/initial state and matches.
  - Next (A2d, running, md5 5947d3af): `[GDNIN]` dump inside the GDN forward: the exact ssm/conv rows each layer reads (post-restore) + metadata row indices (`prefill_state_indices`, `non_spec_state_indices_tensor`, `spec_state_indices`) and `has_initial_state`. Compare to the block-entry `[STATE]` dump per layer → which of conv/ssm, which layers, and whether the row index the kernel uses ≠ the row snapshot/restore uses.
- 2026-09-09 16:05 — **ROOT CAUSE FOUND (A2d, `[GDNIN]` dump).** The per-layer GDN forward reads its state from `prefill_state_indices` = **row 1 for layers 0,3,6,…, row 2 for layers 1,4,7,…, row 3 for layers 2,5,8,…** — vLLM's hybrid KV-cache allocator puts the 24 GDN layers into **three mamba kv-cache groups** (equal layer count per group: 8 attn / 8+8+8 GDN), each with its own block table and state row. `_discover_gdn` picked the FIRST MambaSpec group and `_gdn_snapshot_restore` used `block_tables[that_group][:, 0]` for **all 24 layers** → snapshot/restore was correct for the 8 layers in group 0 and a no-op (wrong row) for the other 16. Denoise rounds dirty the conv state (our override's `causal_conv1d_fn` writes `conv_states`); ssm is untouched (`output_final_state=False`). So from the 2nd denoise round on, and in the commit, 16/24 GDN layers run with the previous round's **noisy conv state** → wrong hidden states from model layer 1 (first non-group-0 GDN layer) → wrong commit K/V persisted + wrong next-block state → compounding error, 49.5% GSM8K. Explains every earlier observation: layer 0 exact in all rounds (group 0), den0 exact (no restore needed), "L0–L2 state exact" in A2 was an artifact (the dump also read group-0's row for every layer), CL-dependence (fewer restores per token at large CL), forced-causal not helping, gate identical.
  - **Fix (md5 16ab2ed5, deployed):** `_discover_gdn` maps every GDN layer name (`static_forward_context` key) to its MambaSpec group via `kv_cache_groups[i].layer_names`; `_gdn_snapshot_restore` uses each layer's own group row (`layer_rows[li]`) for snapshot, restore and the `[STATE]` diagnostic. No kernel/gate/attention changes.
  - Validation launched in parallel (production PIECEWISE config unless noted): (1) `run_bd4_fix.sh` GSM8K n=200 bd4/thr0.9/ms8, 4 shards GPUs 2–5 (`g8k_v3.py`); (2) eager `a2c_hid.sh` rerun on GPU 0 → expect seed-row error after restore ≈ den0 (≤1%) and post-commit K/V ≈ reference; (3) `a1_oracle_fix.sh` AR-oracle on GPUs 6,7 → expect 0 real seed mismatches (ties excepted).
- 2026-09-09 16:40 — **Fix A VALIDATED** (code md5 16ab2ed5, bd4 = CL4 / thr 0.90 / ms8, production PIECEWISE unless noted):
  | check | before fix | after fix | bar |
  |---|---|---|---|
  | GSM8K n=200 accuracy (`run_bd4_fix.sh`, `g8k_v3.py`, 4 shards) | 49.5% ±6.9 | **83.0% ±5.2** | ≥85% (SGLang bd4 87.0 ±4.7, causal 89.0) — within CI, point est. 4 pts under |
  | single-stream TPS | 34 | **40.3** | ≥34 |
  | avg output tokens (loops) | 392 | **304** | SGLang 335 |
  | AR-oracle real seed mismatches (`a1_oracle_fix.sh`, 7 traces) | 9.6% | **0 / 369 boundaries** | 0 |
  | tok/fwd (denoise+commit) | 1.13–1.27 | **1.27** | unchanged |
  | commit-step hidden state vs AR prefill (eager `a2c_hid.sh`) | 8–39% from layer 1 | **≤1–2% all layers** | bf16 noise |
  | persisted block K/V vs AR prefill | 0.1–1.5 | **≤0.02 typical, ≤0.10 max** | bf16 noise |
  - Remaining 83 vs 87 gap (not significant at n=200): the one known engine difference left is the seed row's attention in denoise (vLLM: seed attends bidirectionally incl. MASK slots; SGLang: seed attends only itself) — visible as 6–36% seed-row deviation at layers ≥3 in denoise rounds 2+ (commit rows are clean). Optional follow-up if a paired n=1319 run shows a real gap.
  - Committing fix + diagnostics; then back to the throughput plan (Fix B/C, knobs #5/#6/#3/#4/#10).
- 2026-09-09 18:40 — **Comprehensive run results** (`runs/20260909_143259`, report page https://claude.ai/code/artifact/f86b53b5-c728-4cc4-8ff6-5f65f587e512). Full GSM8K 1319, fork client (boxed prompt), greedy, no-think:
  | engine | acc | ±95% | tok/fwd |
  |---|---|---|---|
  | vLLM bd4 thr0.9 | 77.0% | 2.3 | 1.376 (trace, 200 req) |
  | SGLang bd4 thr0.9 | 78.5% | 2.2 | 1.396 (dllm_stats) |
  | vLLM AR | 81.3% | 2.1 | 1.0 |
  | SGLang AR | 81.8% | 2.1 | 1.0 |
  - **Engine gap closed**: vLLM bd4 ≈ SGLang bd4 within CI, tok/fwd within 1.5%. Remaining ~4-pt bd4-vs-AR gap is the model's, identical on both engines.
  - Trace (16,839 blocks): rounds/block 1:40% 2:29% 3:31%, mean 1.91 → 4/(1.91+1)=1.376. First-round acceptance slot1 91%, slot2 65%, slot3 47%. Seed chain 0/16,639 mismatches. Rejected preds at conf 0.80–0.90 match the final token 92% (calibration monotone) → thr 0.80 candidate; must be measured (early commits change the context).
  - Timers (12,694 steps): median step 26.0 ms = snapshot/restore 4.15 + forward 20.4 (GDN override host 11.6) + sampler 1.44; AR 4.8 ms → speed 0.25× AR; break-even needs 5.4 tok/fwd (> block-4 ceiling of 4). Lever ladder (estimates): Fix C 4/1.91=2.10 tok/fwd; Fix B step ≈ 11 ms; B+C ≈ 0.8× AR; B+C+1-round blocks ≈ 1.7×.
  - Caveat: clean multi-replica jobs were CPU-starved (16 cores/job; 5 replicas → 27 tok/s vs 38 single) → speed from the single-replica sweep; grid jobs get 8 cores/replica.
  - Checkpoint `block_diffusion.json` says `bd_size: 32` → model trained at block 32 yet accurate only at small blocks; training lever = accuracy far from the seed, not window size.
- 2026-09-09 18:45 — **Grid run launched** (user: thr 0.80, block 4 and 32; plan approved; b32/thr0.90 control included): `runs/grid_20260909_153355`, Slurm 3543–3548: {vllm,sglang} × {b4-t080, b32-t080, b32-t090}, full 1319, 3 replicas × 8 cores each, vLLM in trace mode. New yamls `diag/trida_b4_t08.yaml`, `diag/trida_b32_t08.yaml`; `run_job.sh` now takes CL/THRESH/MAXSTEPS/SGL_YAML; `submit_all.sh grid`.
- 2026-09-09 19:05 — **Fix B in autopilot** (user: "go, start in autopilot. until the end"). Dev copies on the box so the running grid jobs keep the frozen code: `vllm-native-dev` (B1+B2, md5 2568f1b6), `vllm-native-dev2` (B1+B2+B3a, md5 f76012f8); `CODE_DIR` env now selects the code dir in `serve_diff_cl.sh` and the job scripts.
  - **B1**: ssm snapshot/restore removed — the denoise override never writes the ssm cache (copies initial state, `output_final_state=False`), so it was a no-op costing ~2 ms/step.
  - **B2**: conv restore removed — denoise conv now reads/writes a per-slot WORK buffer `_conv_work_all[L,R,dim,W-1]` (refreshed from the block-entry snapshot `_conv_snap_all` in ONE copy per step), `cache_indices = slot ids`; the kv-cache conv row is never dirtied, so the commit needs no restore. Block-entry snapshot happens on entry steps only (1 host sync/step, ~1 in 3 steps at bd4).
  - **B3a**: denoise path switched to `fused_recurrent_block_causal_gated_delta_rule_packed` (already in `block_causal_readout.py`: the SGLang reference's own denoise kernel — split + l2norm + gating + recurrence + mixed readout in one launch, ssm state read in place by row). Replaces prep kernel + views + gather + recurrent kernel. Numerics = SGLang's (softplus/l2norm formulas differ from the stock prep by fp32 rounding) → bar is accuracy parity, not byte-identity. `TRIDA_GDN_PACKED=0` restores the legacy path for A/B.
  - Validation jobs (Slurm, 1 GPU each): `runs/fixb_20260909_155943` (B1+B2: 30-item trace for byte-identity vs baseline run + 20-item timers), `runs/fixb3_20260909_160251` (B1+B2+B3a: same). Compare with `runs/lib/compare_identity.py`.
- 2026-09-09 19:40 — **Fix B bug found and fixed (null block).** First B1+B2 builds produced garbage from block 1 (0/30). Probe chain: NaN/absmax per layer clean → production-faithful round-0 A/B/C inside the server (`TRIDA_DEBUG_NAN=1`): the work buffer held exactly the cache row (|cache_row−work_row|=0) yet the conv kernel left the output untouched and wrote no state when `cache_indices=[0]`. Cause: **vLLM's `causal_conv1d_fn` treats state index 0 as the NULL block (reserved block id) and skips the sequence** (`conv_states_input_coord == null_block_id`). Production cache rows are 1/2/3 so it never shows; my slot ids start at 0. Fix: work/snapshot buffers get R+1 rows and are indexed by slot+1. Note for the record: the earlier "legacy vs new" A/B was invalid because `.clone()` of the non-dense `[T, dim]` view silently changed the memory layout the kernel keys on (channel-last check) — always clone the `[T, dim]` view first, then transpose.
  - Timings from the (wrong-output) builds are still valid for the code path: B1+B2 snapshot/restore 4.15 → 0.23 ms (step 26.0 → 23.4); B3a GDN override 11.6 → 5.3 ms (step → 16.5 ms).
  - Validation resubmitted (`runs/fixb_v2_20260909_163351`): dbg-abc (confirm |P−W|≈0), b12-trace/timers (`TRIDA_GDN_PACKED=0`), b3a-trace/timers (`=1`), 30/20 items, vs baseline run traces.
- 2026-09-09 20:05 — **Fix B validation (post null-block fix, `runs/fixb_v2_20260909_163351`)**:
  | build | byte-identity vs baseline (30 items) | acc (30) | tok/s (1 replica, trace on) | step ms (loaded node, indicative) |
  |---|---|---|---|---|
  | baseline (c6d71c8) | — | 60.0% | 37.7 | 26.0 (snap 4.15, gdn 11.6) |
  | B1+B2 (`TRIDA_GDN_PACKED=0`) | **30/30 identical** | 60.0% | **42.1** (+12%) | 27.4 (snap 0.23, gdn 14.9 — CPU-contended node) |
  | B1+B2+B3a (packed kernel) | 13/30 identical (expected: SGLang-kernel numerics) | 60.0% | **50.1** (+33%) | 18.7 (gdn 6.9) |
  - B1+B2 accepted (byte-identical). B3a needs accuracy parity at n=200 → job `b3a-acc200` (2 replicas, same 200 items as baseline's 77.5%). Timer numbers to be re-taken on an idle node once the grid drains.
  - Grid partials (full 1319): **SGLang b4/thr0.80 = 78.0%** (vs 78.5% at 0.90 → threshold 0.80 costs nothing measurable on SGLang), b32/thr0.80 = 56.9% vLLM / 56.7% SGLang (engines agree; block 32 loses ~21 pts), speeds 64 / 115 tok/s per GPU for b32.
- 2026-09-09 20:20 — **Concurrency sweep (run 1, fixed 512-tok outputs, 64 prompts, 1 replica each, baseline code):**
  | engine | C=1 | C=4 | C=8 | C=16 |
  |---|---|---|---|---|
  | vLLM AR | 219 | 722 | 1633 | 2970 |
  | SGLang AR | 222 | 80 * | 153 * | 500 |
  | vLLM bd4 | 38 | 137 | 249 | **435** |
  | SGLang bd4 | 38 | 101 | 191 | 375 |
  - vLLM bd4 == SGLang bd4 at C=1 (38 tok/s, 0.17× AR); vLLM bd4 batches fine (`--max-num-seqs 16` works, 435 tok/s at C=16, ahead of SGLang bd4's 375). * SGLang AR at C=4/8 is anomalous (slower than C=1) — a server-side batching/CUDA-graph-size issue in that replica, flagged not trusted.
  - Sweep scripts had a `wait` that also waited on the server processes (job hung after finishing) → fixed to wait on the bench runners only; sweep-grid inherits the fix before it starts.
- 2026-09-09 20:40 — **Grid results (full 1319, `runs/grid_20260909_153355`, 4 of 6 jobs done):**
  | config | vLLM acc | SGLang acc | vLLM tok/fwd | SGLang tok/fwd | vLLM rounds/block |
  |---|---|---|---|---|---|
  | b4 / 0.90 (run 1) | 77.0 ±2.3 | 78.5 ±2.2 | 1.376 | 1.396 | 1.91 |
  | **b4 / 0.80** | **77.0 ±2.3** | **78.0 ±2.2** | **1.462** | 1.418 | 1.74 |
  | b32 / 0.80 | 56.9 ±2.7 | 56.7 ±2.7 | 2.047 | 2.112 | 14.6 |
  | b32 / 0.90 (control, partial) | running | running | 1.761 (partial) | — | 17.2 |
  - **Threshold 0.80 is free at block 4**: identical accuracy on vLLM, −0.5 pt (noise) on SGLang, +6% tok/fwd. → sweet spot so far = **b4 / thr 0.80**.
  - **Block 32 is not viable on this checkpoint**: −21 pts on both engines for only +45% tok/fwd. First-round acceptance decays with distance from the seed: 91 → 69 → 51 → 35 → 24 → 18 → 13 → 9% at slots 1–8 (thr 0.80). This is the curve for any training decision; block 8/16 would land between.
  - Report page: added a "verdict" section (AR beats bd4 at every concurrency on this checkpoint: 0.17× at C=1 … 0.15× at C=16) and the grid section with the acceptance-vs-slot chart.
- 2026-09-09 21:10 — **FLARE re-read + self-spec (AR-Trust) experiment launched.** FLARE's headline throughput (FLARE-4B 1,293 tok/s GSM8K @C=8, A100) is its AR-Trust mode (diffusion drafts, AR verification); Diffusion-Trust is the slower, lower-quality mode. Our vLLM AR already does 1,633 @C=8 (H100). User asked to measure our spec mode: `runs/spec_20260909_170829`, SGLang `self-spec` (`HybridDiffusionSelfSpec`, `bd_bidir_shift`, `use_spec_verify`) with draft lengths g4/b7, g8/b15, g16/b31 (`diag/trida_selfspec_g{4,8,16}.yaml`, block_size = 2·gen_block − 1), full GSM8K ×3 replicas each (jobs 3566–3568) + concurrency sweep vs causal (3569). `run_job.sh` gained `sglang/selfspec`; `sweep_spec_job.sh` added.
  - Fix B: B3a as-is = 75.0% vs 77.5% on 200 items (+25% tok/s) → full-1319 accuracy job running (3564); numerics-matched packed kernel (stock softplus form, stock l2norm, bf16 rounding of q/k) deployed as dev3 → 30-item byte-identity test (3565).
- 2026-09-09 21:25 — **B3a numerics**: matching the packed kernel's l2norm form (`x·(1/√(Σ+1e-6))`, bf16 round-trip as the stock prep stores q/k), stable softplus and threshold raised byte-identity from 13/30 to **21/30** items (others diverge >200 chars in; residual = fp32 reduction-order rounding, not a formula gap). Recurrence arithmetic verified identical line-by-line. Byte-identity is not attainable across kernels → acceptance criterion = full-1319 accuracy within ~1 pt of 77.0% (job 3564 running, SGLang-numerics variant). Matched variant lives in `vllm-native-dev3` (block_causal_readout.py, also in repo working tree).
- 2026-09-09 21:50 — **Self-spec (AR-Trust) ported to vLLM — first run works.** Plan in `SELFSPEC_VLLM_PLAN.md`. Implementation (`TRIDA_SELFSPEC_N=N`, canvas 2N−1): `_selfspec_step` sampler (GPU fixed-shape greedy verify: cumprod of argmax==spec; emit canvas[0:1+n_acc]; next canvas [argmax(logit_{n_acc}), new specs from MASK slots if all accepted]), `_trida_gdn_selfspec_core` (conv on the cache row + packed kernel `causal_mode=2, num_clean=N` with per-slot intermediate states; sampler persists ssm state after slot n_acc and rebuilds the conv window from saved pre-conv rows), attention fully causal (verify exact; drafts weaker than the reference's mask-rows-see-all). Key runner fact: our ModelState has no bonus token (`num_new_sampled_tokens_per_step=0`) → canvas is the whole query and emitted tokens must be the canvas prefix; the clean/corrected token becomes next step's slot 0.
  - Smoke (5 items, N=4, `runs/vspec_20260909_172532`): 5/5 correct, **4/5 token-identical to vLLM AR greedy** (1 near-tie divergence, expected bf16 effect of a 7-token forward vs 1-token decode), tok/fwd **2.17–2.49**, accepted-specs histogram ≈ {0:45%, 1:12%, 2:9%, 3:34%}, 98 tok/s single-stream on the trace server.
  - Submitted: N=4 and N=8 × 200 items (identity + accuracy vs `vllm-causal-clean`), `sweep_vspec_job.sh` (vLLM spec N=4/N=8 vs vLLM AR, C=1..16). v1 limitation: mixed prefill+canvas batches fall back to the stock kernel with a warning (state persistence would be wrong) → relevant at C>1; the sweep will show.
- 2026-09-09 22:30 — **vLLM self-spec: 200-item results + state fingerprint.** N=4: **82.0%** vs AR 83.0% on the same 200 items, 109/200 generations token-identical to AR greedy, tok/fwd **2.37**; N=8: 81.5%, 107/200 identical, tok/fwd 2.46; 98 / 95 tok/s single-stream on trace-instrumented servers (clean sweep pending). No mixed-batch fallbacks triggered. Fingerprint (`a3_selfspec_state.sh`, item 0, 65 steps): persisted ssm/conv state vs fresh prefill of the same prefix differs 0.3–2% (ssm) / ~1% (conv, single-layer spikes up to 15% on a 2-dim projection) and is FLAT over steps (no drift) — same order as the post-fix diffusion commit path → numerics (bf16 state, canvas forward vs chunked prefill), not a state-commit bug. Divergences from AR greedy are near-tie flips, consistent with 55% identity and equal accuracy.
  - **Process correction (user, repeated 3×):** every job must take the whole cluster (2 nodes, 16 GPUs, DP) and jobs queue in Slurm. Added `run_job_2n.sh` (srun 1 task/node, each node 8 replicas over its half of the items via the client's new `--offset`; analyzer merges `<job>-n0/-n1`), memory note `slurm-whole-cluster-per-job`. Queued: full-1319 self-spec N=4 as a 2-node job, then a timers job for the self-spec step breakdown.
- 2026-09-09 22:50 — **Self-spec results, both engines (full 1319 on SGLang; 200 on vLLM):**
  | engine / config | acc | tok/s per GPU (single-stream) | tok/fwd |
  |---|---|---|---|
  | SGLang self-spec g4 | 80.8% | 151 | (dllm_stats, see summary) |
  | SGLang self-spec g8 | **82.0%** | **163** | |
  | SGLang self-spec g16 | 81.7% | 160 | |
  | SGLang AR (run 1) | 81.8% | 121 | 1.0 |
  | vLLM self-spec N=4 (200 items) | 82.0% (AR 83.0 same items) | 107 (clean sweep C=1) | 2.37 |
  | vLLM self-spec N=8 (200 items) | 81.5% | 112 | 2.46 |
  | vLLM AR (run 1) | 81.3% | 219 | 1.0 |
  - **On SGLang, AR-Trust beats its own AR by 1.35× single-stream (matches FLARE's claim).** On vLLM our day-one port is lossless (82.0 vs 83.0 on the same items, 55% token-identical) and 2.9× the diffusion path (107 vs 38 at C=1; 1107 vs 435 at C=16) but **0.5× vLLM AR** — the step cost: a 2N−1-token forward through the eager GDN path + my naive 24-layer state-commit loop. vLLM AR itself (219 / 2970 @C=16) is far ahead of SGLang AR (121 / 500), so the bar on vLLM is higher.
  - Next lever (Fix S): zero-copy state commit (read next step's initial ssm state directly from the intermediate buffer at the accepted index; one fused kernel for the conv window across all layers), then FULL cuda-graph capture of the now fixed-shape step. Timers job queued to quantify the current breakdown.
- 2026-09-09 23:30 — **Fix S (self-spec step cost) implemented → dev5.** Zero-copy GDN commit: intermediate states live in a per-layer ring `[L, 2R, N, HV, V, K]`; step t reads its initial ssm state directly at `((par·R+slot)·N + prev_acc)` and writes its N intermediates into half `1−par`; the sampler only records `prev_acc` and flips the slot's parity (per-slot parity so requests that skip a step stay consistent at C>1). Conv: per-slot clean window `[L, R+1, W−1, dim]` (row slot+1; row 0 = null block) copied into a kernel scratch in ONE stacked copy per step; next window = one `cat`+`gather` over all layers instead of 24×8 ops. Cache rows are imported into the ring once per request (first self-spec step). ssm/conv caches are never written by self-spec afterwards. Per-step launches: ~3/layer (preconv save, conv, packed) + ~6 → ~80 vs ~290 before. Queued: 30-item identity smoke + timers (1 GPU each, behind the whole-cluster jobs).
  - g32 (block 63 = training block) queued on both engines as 2-node jobs; committed `4b9da18` (self-spec + Fix B1/B2, packed kernel opt-in).
- 2026-09-09 23:50 — **SGLang self-spec concurrency sweep** (fixed 512-tok, 1 replica): g4 **316** / 248 / 488 / **908** tok/s at C=1/4/8/16 vs SGLang AR 223 / 141 / 280 / 557 → AR-Trust = 1.4–1.6× its own AR on SGLang, and **SGLang self-spec at C=1 (316) already beats vLLM AR (219)**. Implied step cost: SGLang 316/2.74 ≈ 115 fwd/s ≈ 8.7 ms; ours 107/2.37 ≈ 45 fwd/s ≈ 22 ms; target ≤ 4.8 ms (AR step) → ≥ 500 tok/s @C=1.
  - Plan to get there (in order): Fix S (in validation) → direct Triton launches with cached args (bypass the `causal_conv1d_fn` / packed wrappers' Python) → fuse conv into the packed kernel (1 launch per GDN layer, window rows written by the kernel itself, no preconv copy) → FULL cuda-graph capture of the fixed-shape step (needs the GDN op out of `splitting_ops` + capturable metadata; the shelved FULL design applies).
  - **Goal reset by user (autopilot):** self-spec on vLLM with throughput lower-bounded by vLLM AR; diffusion-mode speed work deprioritized. Memory `goal-selfspec-on-vllm`.
- 2026-09-10 00:10 — **Fix S validated**: 30/30 generations identical to self-spec v1 (pure data-movement change), 117 tok/s vs 98 on the trace server (+20%). 16/30 identical to AR greedy (near-tie flips, as before). Next: fuse the conv into the packed kernel (one Triton launch per GDN layer, no wrapper Python), then cuda-graph.
- 2026-09-10 00:40 — **Self-spec step timing after Fix S**: 16.9 ms median (forward 15.6, sampler 0.9, prep 0.4) at N=4 → tok/fwd 2.37 ⇒ ~140 tok/s ceiling @C=1 vs AR's 4.8 ms step. The forward is still 24 eager GDN ops (conv kernel + packed kernel + wrapper Python each) under PIECEWISE. **Fix S2**: `fused_selfspec_gdn_layer_kernel` — conv (width-4 SiLU from x + the per-slot clean window, stock numerics: bf16 products into an fp32 bias-seeded accumulator) + split/l2norm/gating + recurrence + mixed readout in ONE Triton launch per layer, reading the ring state by index and caching intermediates; wrapper caches per-layer constants. `TRIDA_SS_FUSED=0` keeps the two-kernel path for A/B. Deployed to dev6; identity (vs Fix S output) + timers queued.
- 2026-09-10 01:10 — **Results batch.** (1) **vLLM self-spec N=4, full 1319 items (2-node job): 81.3% = vLLM AR 81.3%** → lossless at scale. (2) Self-spec v1 timers: 20.8 ms/step (fwd 17.6, sampler 3.1) → Fix S 16.9 ms (fwd 15.6, sampler 0.9). (3) **B3a full set: 76.6% vs 77.0% on the same 1319 items** (−0.4 pt, CI ±2.3) with +33% tok/s → packed denoise kernel now default ON. (4) S2 fused kernel failed to compile: `conv1d.bias` is None for this model (stock kernel seeds the accumulator with zeros) → wrapper now substitutes a cached zero bias; resubmitted. (5) grid sweep finished → analyzer rerun.
- 2026-09-10 01:25 — **vLLM self-spec N=32 (block 63 = training block), full 1319, 2-node job: 81.6%** (AR 81.3%) → lossless too; tok/fwd from traces pending analyzer. SGLang g32 servers OOM'd at start ("Not enough memory", block-63 self-spec buffers × 32 running requests) → resubmitted with `MAX_RUNNING_REQUESTS=4` (run_job now passes `SGL_ENV` through the `env -i` launch).
- 2026-09-10 01:50 — **Fix S2 validated (fused single-launch GDN layer):** step **11.6 ms** (fwd 10.3, sampler 0.9) vs 16.9 (Fix S) / 20.8 (v1); 158 tok/s on the trace server (AR 219). 21/30 generations identical to Fix S, the rest near-tie flips from the in-kernel conv's rounding; accuracy check moved to the full set (2-node job) + clean sweep. Remaining gap to the 4.8 ms AR step: 24 Python-dispatched Triton launches + 8 eager attention ops + graphed pieces → next is CUDA-graph capture of the self-spec step (GDN op out of `splitting_ops`, persistent index buffers).
- 2026-09-10 02:20 — **CUDA-graph step for self-spec (experiment).** Made the self-spec step graph-safe (dev7): per-step index tensors (slots, ssm read idx, ring write idx, window rows) now live in persistent buffers with `[:n]` views (fixed pointers), mode = `selfspec` also during capture, fused core no longer touches attention metadata (T = n·blk), mixed-batch fallback kept. Experiment: `--compilation-config` with `vllm::qwen_gdn_attention_core` REMOVED from `splitting_ops` (so the 24 GDN layers are captured inside the piecewise graphs) and `cudagraph_capture_sizes=[7,14,28,56,112]` (multiples of the canvas). Queued: dev7 sanity (default config, must equal S2), graph-smoke (identity), graph-timers. Commit `8702e07` = Fix S/S2 + B3a default on.
- 2026-09-10 02:40 — **g32 (block 63) self-spec**: SGLang **74.7%** (986/1319, 120–126 tok/s per GPU, `MAX_RUNNING_REQUESTS=4`) vs **vLLM N=32 81.6%**. The reference loses ~7 pts at the training-block draft length while our port stays lossless — worth flagging to the SGLang owner (possible long-block issue in `HybridDiffusionSelfSpec`, e.g. fast-verify/approximation paths or KV handling at block 63); not a vLLM concern. Best SGLang config remains g8 (82.0%, 163 tok/s).
- 2026-09-10 03:10 — **Fix S2 at scale**: full 1319 = **81.7%** (AR 81.3%) → lossless. Clean sweep (fixed 512-tok, 1 replica, dev6): N=4 **161 / 526 / 875 / 2278** tok/s at C=1/4/8/16; N=8 168 / 607 / 798 / 2353. vs vLLM AR 219 / 722 / 1633 / 2970 → **0.74× / 0.73× / 0.54× / 0.77×** (was 0.49× / 0.51× / 0.32× / 0.37× before Fix S/S2). Remaining gap = the 24 eager GDN launches + 8 eager attention ops (graph experiment running).
- 2026-09-10 03:40 — **CUDA-graph experiment (GDN op inside piecewise graphs, capture sizes = multiples of 7)**: correctness OK (graph-smoke 30/30 identical to S2, 170 tok/s on the trace server) but timing barely moved: **11.19 ms vs 11.56 ms** (fwd 9.87 vs 10.26). Either the 7-token canvas batches are not replaying the captured graphs (capture classification / size), or the remaining time is GPU work + the 8 eager attention ops. Sanity job failed on a JSON-quoting glitch in the sbatch export (harmless; the graph job's identity result covers it). Next: torch-profiler capture of the S2 step to split CPU dispatch vs GPU kernel time and see whether cudaGraphLaunch appears.
- 2026-09-10 04:20 — **Self-spec step profile (torch profiler, 128-tok request, 69 forwards, S2 build):** GPU busy **5.06 ms/forward** (GEMM 3.72, fused GDN 0.58, norm/elementwise 0.36, sampler 0.22, attention 0.02, memcpy 0.06); GPU idle **24 ms/forward under the profiler** (83% idle; unprofiled step 11.6 ms ⇒ ~6.5 ms host/launch overhead). Host API per forward: **156 eager kernel launches + 32.5 graph launches**, 2.7 syncs (negligible). The GDN-in-graph compilation config produced an IDENTICAL profile (same launches) → the override was not taking effect / no replay for the canvas batch. **Conclusion: the GPU floor (5.1 ms) ≈ the AR step (4.8 ms); the entire remaining gap is host-side dispatch.** With 2.37 tok/fwd a host-free step would give ≈ 2.2× vLLM AR. Path: reclassify the canvas step as a uniform multi-token decode (spec-decode shape) so vLLM's FULL/FULL_AND_PIECEWISE capture covers the whole forward (the shelved FULL design), plus fold the 24 pre-conv copies into the fused kernel and shrink the sampler's ~59 launches.
- 2026-09-10 04:50 — **S3 (launch reduction)**: the fused kernel now writes the readout straight into `core_attn_out` and the raw pre-conv rows into the per-slot preconv buffer (each program stores its head's q/k/v channels), removing 48 eager launches per step (24 preconv copies + 24 output copies). Why not vLLM FULL graphs yet: the dispatcher's uniform-decode query length is `1 + num_speculative_tokens` (bonus token + drafts) and the GDN metadata builder only takes its spec-decode path when `speculative_config` is set; our ModelState presents a `CL`-token query with no bonus token and no speculative_config → canvas batches are prefill-classified → never FULL-captured (the shelved FULL design). Reshaping the ModelState to the spec-decode layout (num_new_sampled_tokens_per_step=1, drafts = specs+masks, speculative_config with num_speculative_tokens = 2N−2) is the route to graph-capturing the whole step; queued behind S3.
- 2026-09-10 05:20 — **S4 plan (FULL cuda-graph for the self-spec step).** Findings: (1) GPU floor 5.1 ms/step ≈ AR step; all else is host dispatch. (2) vLLM v2 FULL capture needs the batch to be a *uniform spec decode*: query = 1 bonus + drafts (`uniform_decode_query_len = 1 + num_speculative_tokens`), and the GDN/FA builders classify spec-as-decode only via `speculative_config` (`_init_reorder_batch_threshold`, GDN `use_spec_decode`, `decode_cudagraph_max_bs = max_num_seqs·(num_spec+1)`, `build_for_cudagraph_capture` asserts tokens ≤ that). (3) Upstream `model_states/mamba_hybrid.py` shows the v2 plumbing: `ModelSpecificAttnMetadata.get_extra_common_attn_kwargs` supplies `num_decode_draft_tokens_cpu` / `num_accepted_tokens`. Plan: (i) ModelState bonus-token layout: `num_new_sampled_tokens_per_step=1`, config canvas_length = 2N−2 (drafts = specs+MASKs), emit `[accepted specs…, next_tok]`, prefill emits the seed; (ii) supply `num_decode_draft_tokens_cpu`/`num_accepted_tokens` like mamba_hybrid; (iii) plugin-side: inject a temporary `speculative_config(num_speculative_tokens=2N−2)` into `vllm_config` during GDN/FA metadata-builder `__init__` so thresholds/spec tensors/capture sizes are right without a real speculator; (iv) serve with `FULL_AND_PIECEWISE`; (v) validate identity vs S3, timers, profile (expect kernel launches → ~0, graph launches ≈ 1–2/step).
- 2026-09-10 06:30 — **S4 implemented (dev9, md5 eb89e717)**: self-spec now presented in vLLM's spec-decode shape — `num_new_sampled_tokens_per_step=1` (canvas slot 0 = vLLM's last sampled token), diffusion `canvas_length = 2N−2` drafts (`CL=6` for N=4; states/canvas keep 2N−1 slots), emission = `[accepted specs…, next_tok]` (prefill emits the seed), `postprocess_state` tracks last num_sampled → `num_accepted_tokens`, `_TridaSpecAttnMetadata` supplies `is_prefilling` / `num_accepted_tokens` / `num_decode_draft_tokens_cpu` (mirrors upstream mamba_hybrid), and `_install_spec_builder_shim` presents a minimal `speculative_config(num_speculative_tokens=2N−2)` to the GDN/FA metadata builders during their `__init__` only (thresholds, spec tensors, `decode_cudagraph_max_bs`) — no real speculator. Padded requests under FULL replay map to dummy ring/window/preconv rows. Jobs: identity vs S3 under `FULL_AND_PIECEWISE`, timers, and a PIECEWISE control (same code) to separate layout effects from graph effects.
- 2026-09-10 07:00 — **Tooling bug found:** `--compilation-config` overrides never reached vLLM. `sbatch --export` splits values at commas (the graph-config JSON was truncated back to plain PIECEWISE) and the serve script's `${COMPILATION_CONFIG:-{…}}` default appended a stray `}` to any set value (→ "Invalid JSON: trailing characters"). Consequence: the earlier "GDN-in-graph" experiment and profile were PIECEWISE runs (hence identical). Fix: `COMPILATION_CONFIG_FILE` (JSON file) in `serve_diff_cl.sh`; S4 jobs resubmitted (FULL_AND_PIECEWISE identity + timers, PIECEWISE control).
- 2026-09-10 07:40 — S4 first FULL_AND_PIECEWISE attempt failed exactly where predicted without the builder shim: `GDN only supports decode-only full CUDAGraph capture … tokens (7) <= cudagraph capture sizes (1)` — the shim was wired into `qwen3_5_diffusion.register()` but the live entry point is `plugin.register_trida()`; fixed (shim installed from the plugin), resubmitted. The S4 PIECEWISE control (same spec-shape code) started healthy → the layout change itself runs; its identity vs S3 is pending.
- 2026-09-10 08:20 — **S4 second round**: FULL cuda graph now captured (`Capturing CUDA graphs (FULL): 1/1`) — the builder shim works. Remaining failure was the scheduler: for diffusion models it hard-codes `num_sampled_tokens_per_step = 0` (no bonus token), so it scheduled 6 tokens/step (drafts only) while the runner/kernels expect 7 → our fused core hit the mixed-batch fallback (T=6) → garbage, and the async scheduler asserted `num_output_placeholders >= 0` when the prefill emitted the seed. Added a scheduler shim (`Scheduler.__init__` post-hook → `num_sampled_tokens_per_step = 1` in spec-shape). Resubmitted PIECEWISE control + FULL identity/timers.
- 2026-09-10 08:55 — **S4 root cause of the garbage output**: FULL graphs captured and the step ran (7.5 ms, tok/fwd 2.36 = S3), but every item started "Tod\nTo determine…" under both PIECEWISE and FULL → layout bug, not graphs. `_handle_prefill` returned `num_rejected=num_sampled` (=1 in spec-shape), and vLLM's `post_update` advances `num_computed_tokens` by `query_len − num_rejected`, so the runner sat one token behind the scheduler forever and the first step overwrote the last prompt token's KV. Fix: `num_rejected=0` at prefill. Resubmitted PIECEWISE control (identity vs s3-n4-30 must be 30/30) + FULL identity + timers.
- 2026-09-10 09:20 — **S4 VALIDATED.** With `num_rejected=0` at prefill: PIECEWISE control and FULL_AND_PIECEWISE both 30/30 token-identical to `s3-n4-30` (acc 23/30, 15/30 identical to AR greedy = same bf16 near-tie profile as before), tok/fwd 2.32. Step 10.0 → **7.5 ms** (fwd 6.16 + sampler 0.90 + snap 0.41); C=1 throughput 188 (PIECEWISE) → **248 tok/s FULL** vs vLLM AR 219 → first time self-spec on vLLM beats vLLM AR at C=1. Queued (whole cluster, sequential): `s4-n4-full` (1319 items, 16 replicas) and `sweep_s4` (spec-n4/spec-n8/causal, C=1/4/8/16; CKPT fixed so the causal baseline actually runs this time; CL 6/14 for the spec-shape layout).
- 2026-09-10 09:50 — **S4 full set (`s4-n4-full`, 1319 items, 16 replicas, FULL cuda graph):** 1077/1319 = **81.7%** (S2: 1077, vLLM AR 81.3%), 772/1319 token-identical to AR greedy (S2: 771). Wall 254 s → **134 s** for the same work. Sweep `sweep-s4` running (spec-n4 / spec-n8 / causal, C=1/4/8/16, whole cluster).
- 2026-09-10 10:25 — Sweep job died at startup for the spec servers under `--max-num-seqs 16`: vLLM's warmup runs the sampler on synthetic batches (e.g. nlogits=[7,1]) → `reshape(num_decode, CL)` failed. Guard added: decode rows whose logit count ≠ CL are treated as a dummy run (emit nothing). Smoke with max-num-seqs 16: 30/30 identical, 253 tok/s. AR baseline from the same sweep job (FULL, max-num-seqs 16): 220 / 733 / 1573 / 2900 tok/s at C=1/4/8/16. Sweep resubmitted (3621, whole cluster).
- 2026-09-10 11:05 — **S4 sweep (`sweep_s4`, FULL cuda graph, max-num-seqs 16, 64 prompts × 512 tok, one replica each, same job as the AR baseline):**

  | engine | C=1 | C=4 | C=8 | C=16 |
  |---|---:|---:|---:|---:|
  | vLLM AR | 220 | 731 | 1572 | 2889 |
  | vLLM self-spec N=4 (S4) | **351 (1.60×)** | **873 (1.19×)** | **1841 (1.17×)** | **2886 (1.00×)** |
  | vLLM self-spec N=8 (S4) | 333 (1.51×) | 906 (1.24×) | crash | crash |

  **Goal met for N=4: ≥ vLLM AR at every concurrency (parity at C=16, 1.6× at C=1).** History at C=1: v1 107 → S2 ~165 → S3 ~190 (PIECEWISE) → S4 351 tok/s. N=8 dies at C=8/16 with an illegal memory access right after a `mixed batch (T=16 n=2 blk=15): stock fallback` — the known v1 limitation (a prompt chunk scheduled alongside canvases takes the stock GDN path, which is not spec-shape-safe). N=4 hit one mixed batch in the whole sweep and survived; fixing mixed batches (run the canvases through the fused path and only the prompt chunk through stock) is the next correctness item, then S5 (single-kernel sampler) and mask fidelity for more tok/fwd.
- 2026-09-10 13:30 — **Full-eval prep (user: GSM8K + FunctionChat + Ko-AgentBench × {AR, self-spec N=4/8/32}, greedy, thinking on; one-node DP-8 jobs for agentic, two-node for GSM8K).**
  - Smokes (30 items, FULL graphs): N=8 **30/30 identical to N=4** (2.42 tok/fwd, 242 tok/s). N=16 27/30, N=32 15/30 with synonym-level late flips (same accuracy) — numerics drift with the 63-row forward, acceptable. But N=32 FULL produced garbage on item 3 (prompt_len 63 = canvas length): vLLM's FULL-graph dispatch is shape-only (`get_uniform_token_count`), so a 63-token PROMPT for one request replays the self-spec decode graph over prompt tokens. PIECEWISE N=32 was clean. Fix: shim `get_uniform_token_count` to return None unless every scheduled request carries drafts (set from `execute_model`'s scheduler_output). Rerun queued.
  - Plugin: `SamplingParams._validate_diffusion` no-op under self-spec (harnesses send temperature 0.1/0.7; verification is greedy anyway). `vllm serve --data-parallel-size 8 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --max-model-len 32768` with the plugin comes up healthy; FunctionChat 2-record smoke passed end to end (tool_calls parsed, gpt-4.1 judge scored). Ko-AgentBench smoke: fixed `.venv-serve` python (PY) and `uv` PATH; rerun queued.
  - Tooling: `tools/agentic_eval/agentic_job.sh` (server + repo harness in external-endpoint mode, subsets concurrent), `run_job.sh` THINK=1 / MAXLEN knobs, FC_TEMPERATURE / KOAB_TEMPERATURE greedy knobs (env-gated, defaults unchanged).
- 2026-09-10 14:10 — **Full eval launched** (`runs/fulleval_20260910`, commit 2f0be50): {AR, self-spec N=4, 8, 32} × {GSM8K 1319 (2-node, 16 replicas, thinking on, max_tokens 8192), FunctionChat 1306 (1-node DP-8, 3 subsets concurrent, gpt-4.1 judge), Ko-AgentBench L1–L7 (1-node DP-8, concurrency 8, gpt-4.1-mini judge)}. Greedy everywhere (FC_TEMPERATURE/KOAB_TEMPERATURE=0). Chained mode by mode; agentic pairs share the cluster one node each. Pre-flight: N=32 FULL fixed (prompt_len == canvas dispatch bug; remaining FULL-vs-PIECEWISE diffs at N=32 are 3/30 formatting near-ties), Ko-AgentBench L1 smoke 11/11 tasks (harness cache misses for PlaceSearch_kakao are pre-existing, cache-mode read).
- 2026-09-10 (run in progress) — AR row: GSM8K 1032/1319 = 78.2% (167 hit the 8192 cap; 22 min on 16 replicas); FunctionChat singlecall 0.922 / dialog 0.905 / calldecision 0.947 (32 min). Self-spec N=4: GSM8K 1026/1319 = 77.8% (158 capped), wall 914 s vs 1349 s = 1.48× AR; Ko-AgentBench 17 min. **AR Ko-AgentBench invalid (rerun queued as 3644):** Ko-AgentBench's LiteLLM adapter had a 60 s per-call timeout; AR thinking calls (2k–9k tokens at ~140 tok/s) timed out 64 times (Avg_TPS 43 vs 437–1478 for self-spec), truncating trajectories. Added `KOAB_LLM_TIMEOUT` (env-gated, default unchanged) = 1200 in agentic_job.sh; pending N=8/N=32 jobs pick it up (self-spec had 0 timeouts either way). FunctionChat unaffected (300 s client timeout, 0 timeouts).
- 2026-09-10 (run in progress) — N=8: GSM8K 1016/1319 = 77.0% (~1660 tok/s per node); FunctionChat 0.918 / 0.895 / 0.946 (clean 200-row dialog); Ko-AgentBench same task-completion profile as N=4, 16 min, 0 timeouts, 0 mixed-batch fallbacks. N=4 FunctionChat dialog was contaminated by the smoke's cached rows (192/200 evaluated) → purged, dialog-only rerun queued (3645). Self-spec FunctionChat jobs take 77–85 min vs AR 32 min: vLLM runs the diffusion plugin with `enable_prefix_caching=False`, so the shared multi-kilotoken tool prompts are re-prefilled every request, while stock AR gets prefix-cache hits; Ko-AgentBench (generation-dominated) is 4× faster on self-spec. Follow-up: prefix caching for the self-spec server.
- 2026-09-10 (final) — **Full eval done** (`runs/fulleval_20260910`, greedy, thinking on, FULL cuda graphs, one sequence per replica):

  | mode | GSM8K (1319) | GSM8K tok/s, 16 GPU | FC singlecall | FC dialog | FC calldecision | Ko-AgentBench SR (task-weighted, 91 tasks) |
  |---|---:|---:|---:|---:|---:|---:|
  | vLLM AR | 78.2% | 2162 | 0.922 | 0.905 | 0.947 | 0.582 |
  | self-spec N=4 | 77.8% | 3149 (1.46×) | 0.920 | 0.905 | 0.941 | 0.615 |
  | self-spec N=8 | 77.0% | 3288 (1.52×) | 0.918 | 0.895 | 0.946 | 0.615 |
  | self-spec N=32 | 77.6% | 2327 (1.08×) | 0.908 | 0.885 | 0.942 | 0.549 |

  Self-spec matches AR within noise on all three benchmarks (GSM8K ±1 pt, FunctionChat ±2 pts, Ko-AgentBench ±0.05 on 91 tasks); N=32 trails by 1–3 pts on FunctionChat/KoAB (drift from the 63-row forward). Speed: N=4/N=8 ≈1.5× AR on GSM8K; N=32 ≈ AR. Reruns: AR Ko-AgentBench (60 s harness timeout → 1200 s; 0 timeouts after) and N=4 FunctionChat dialog (smoke cache contamination; clean 200 rows = 0.905, identical to AR's 181/200). Report page section 9 updated (same artifact URL).
- 2026-09-10 — **Prefix caching for self-spec: measured, premise corrected.** Replayed 40 recorded FunctionChat singlecall requests against AR, AR+`--enable-prefix-caching`, self-spec N=4, and self-spec N=4 with the flag (one node, `fc_bench_job.sh`):

  | server | wall (40 req) | mean latency | prompt tok | completion tok | prefix-cache hit |
  |---|---:|---:|---:|---:|---:|
  | AR | 16.9 s | 0.42 s | 746 | 76 | 0% |
  | AR + prefix caching | 17.8 s | 0.45 s | 746 | 78 | 42.5% |
  | self-spec N=4 | 19.9 s | 0.50 s (p50 0.31) | 746 | 77 | 0% |
  | self-spec N=4 + flag (naive) | 133 s | 3.33 s | 746 | **1409 (runaway)** | 42.1% |

  Findings: (1) vLLM defaults prefix caching OFF for hybrid models, so the AR eval server had it off too — my earlier explanation of the FunctionChat wall-time gap was wrong; (2) FunctionChat is judge-bound (model time ≈ 0.5 s × 1306 ≈ 11 min; the 32–85 min job times are gpt-4.1 n=3 judge latency); (3) prefix caching does not help this workload even on AR (prompts ~750 tokens); (4) naive prefix caching on the self-spec plugin is WRONG — the plugin reads GDN state rows from column 0 of the raw block table and never writes running state back to cache rows, so align-mode blocks are stale → garbage (identity 10/40). Doing it properly = port MambaHybridModelState's align pre/post-copy + per-step ring→cache-row writeback. Not done: no benefit for FunctionChat, real risk. Also: AR vs AR+pc identity 37/40 — prefix caching itself perturbs bf16 near-ties on stock AR.
- 2026-09-11 — **Step 2: DFlash baseline on vLLM 0.27.1 (built-in `method: dflash`), same protocol as our sweeps (64 GSM8K prompts × 512 fixed tokens, greedy, one H100, `runs/2026-09-11_dflash/sweep-vllm`).** Target: stock Qwen3.5-4B + `z-lab/Qwen3.5-4B-DFlash` (6-layer drafter); self-spec rows are trida (AR speed identical, 217 tok/s @C=1).

  | config | C=1 | C=4 | C=8 | C=16 | accept len |
  |---|---:|---:|---:|---:|---:|
  | AR | 217 | 726 | 1556 | 2859 | — |
  | DFlash block 4 | 308 (1.42×) | 1024 (1.41×) | 1904 (1.22×) | 3392 (1.19×) | 3.83 |
  | DFlash block 8 | 445 (2.05×) | 1416 (1.95×) | 2517 (1.62×) | 4363 (1.53×) | 5.18 |
  | DFlash block 16 | **527 (2.43×)** | **1691 (2.33×)** | **2708 (1.74×)** | **4465 (1.56×)** | 6.5 |
  | self-spec N=4 (ours) | 346 (1.59×) | 861 (1.19×) | 1776 (1.14×) | 2953 (1.03×) | ~2.3 tok/fwd |
  | self-spec N=8 (ours) | 334 (1.53×) | 757 (1.04×) | crash (mixed batch) | crash | ~2.4 tok/fwd |

  Reading: DFlash's separate 6-layer drafter accepts 6.5 of 16 drafted tokens per step; our zero-parameter self-spec accepts ~1.3 of 3. At C=1 DFlash b16 is 1.5× our N=4; at C=16 both converge toward AR (1.56× vs 1.03×). N=16/32 self-spec OOMed at max-num-seqs 16 (state ring 24.75 / 49.5 GB) → rerun queued with 4 seqs (3659). Smoke sanity: all outputs coherent.
- 2026-09-11 — Self-spec wide blocks on vLLM (`sweep-vllm-bigN`, 4 seqs per server): N=16 273 (1.25×) / 635 (0.87×) tok/s at C=1/4; N=32 202 (0.93×) / 573 (0.78×). AR 218 / 730. Wider N is slower: the 2N−1-row forward costs more than the extra accepted tokens buy (acceptance saturates ~2.4–2.5 tok/fwd), and the state ring (24 layers × 2R × N × 1 MB) becomes a memory problem at N≥16. On vLLM the useful self-spec range is N=4–8.
- 2026-09-11 — **SGLang for DFlash: environment notes** (`env/sglang-dflash`). PyPI sglang 0.5.9 has no DFLASH (choices EAGLE/EAGLE3/NEXTN/STANDALONE/NGRAM); DFlash lives in git main (`0.5.6.post3.dev10299+g2f7393f0d`), install with `SGLANG_BUILD_RUST_EXTS=none uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python"` (no cargo on the box). It pulls torch 2.13+cu130 → needs `env/cudacompat/.../cuda-13.0/compat` on `LD_LIBRARY_PATH` (driver 570). Also: `SGLANG_DISABLE_CUDNN_CHECK=1`, `FLASHINFER_DISABLE_VERSION_CHECK=1` (flashinfer 0.6.18 vs cubin 0.6.3, no matching cubin wheel), `--cuda-graph-max-bs-decode` (the short flag is now ambiguous), and the model card's `--mamba-scheduler-strategy` does not exist in this build. SGLang smoke (10 prompts, C=1): AR 242, DFlash b8 779, **b16 892 tok/s (3.7×)** — SGLang's DFlash path is much faster than vLLM's (527 at b16); accept len per decode batch 5–9.
- 2026-09-11 — **Step 2 done: DFlash vs self-spec vs AR on both engines** (`runs/2026-09-11_dflash`, 64 GSM8K prompts × 512 fixed tokens, greedy, one H100 per server, stock Qwen3.5-4B target for DFlash):

  | engine / config | C=1 | C=4 | C=8 | C=16 | accept len |
  |---|---:|---:|---:|---:|---:|
  | vLLM AR | 217 | 726 | 1556 | 2859 | — |
  | vLLM DFlash b4 / b8 / b16 | 308 / 445 / **527** (1.4–2.4×) | 1024 / 1416 / 1691 | 1904 / 2517 / 2708 | 3392 / 4363 / 4465 (1.2–1.6×) | 3.8 / 5.2 / 6.5 |
  | vLLM self-spec N=4 / 8 / 16 / 32 | 346 / 334 / 273 / 202 (1.6× / 1.5× / 1.25× / 0.93×) | 861 / 757 / 635 / 573 | 1776 / crash / – / – | 2953 / crash / – / – | 2.3–2.5 tok/fwd |
  | SGLang (main) AR | 243 | 827 | 1697 | 3116 | — |
  | SGLang DFlash b4 / b8 / b16 | 540 / 760 / **854** (2.2–3.5×) | 1384 / 1774 / 2370 | 3032 / 3626 / 3416 | 5014 / 4948 / 4393 (1.4–1.6×) | 3.3 / 5.0 / 6.25 |

  Takeaways: (1) DFlash's 6-layer drafter accepts 6–6.5 of 16 drafts per step; our zero-parameter self-spec accepts ~1.3 of 3 — that acceptance gap is the whole C=1 story (DFlash b16 2.4× on vLLM, 3.5× on SGLang; self-spec 1.6×). (2) At C=16 everything converges to 1.0–1.6× AR: the GPU is full and speculation only trades compute. (3) SGLang's DFlash implementation is ~1.6× faster than vLLM's at the same acceptance (854 vs 527 at C=1) — the vLLM DFlash path leaves engine overhead on the table, the same lesson as our own S1→S4 work. (4) Wider self-spec blocks are slower on vLLM (N=32 below AR). Report page section 10.
- 2026-09-11 — **FLARE-4B (public checkpoint) on the SGLang reference, our protocol (greedy, no-think GSM8K, 400 items, 8 replicas):** AR-Trust g4 **3.38 tok/fwd**, 82.0%; g8 3.14 tok/fwd, 80.8%. trida same protocol: g4/g8 2.74 tok/fwd. FLARE's recipe (block 4, AR weight 1.0, random masks) buys ~23% acceptance over ours; DFlash's separate drafter gets 6.5. Bar for the draft-aligned fine-tune: beat 3.4. (`runs/2026-09-11_flare4b`)
- 2026-09-11 — FLARE-4B AR baseline (same protocol): 82.2% GSM8K, 97.5 tok/s per GPU on the FLARE SGLang fork; its AR-Trust g4 is 1.42× its own AR (138 tok/s) at 3.38 tok/fwd. All three FLARE-4B rows are on the report page (section 8). Self-distillation generation (job 3674) started 2026-09-11.
- 2026-09-11 — **Draft-aligned fine-tune prep** (branch `feat/draft-align`): trainer `--mask_pattern canvas` (suffix masks per turn-aligned block, single view, V-agnostic forward; synthetic check 0 violations), self-distillation generator (v6 prompts, step_18000 AR greedy regenerates every assistant turn, teacher-forced; 16k cap drops the conversation; round-trip exact up to `<|im_end|>`), smoke: 60 conversations → 46 kept (12 capped), ~750 generated tokens/turn. Full generation (40k conversations, 2 nodes) queued as job 3674. Training sbatch `train_2node.sbatch` (step_18000 via --model_id, bd 8, AR weight 1.0, LR 5e-6 const, 1500 steps, weights-only saves under `models/scratch/`), eval hook `eval_ckpt.sh` (vLLM ss4/ss8 traces + AR identity guard).
- 2026-09-11 — **Pivot to a feasibility pilot** (user: "is there a way we can do partially and see if this direction is right or wrong?"). Generation paused at ~6.2k conversations (kept 4,162 = 11,553 regenerated turns; node offsets 3000/3200 of 20000 for resume). Cancelled the queued chain (3676–3689). Pilot: `trida-4b-draftalign-pilot`, same recipe (canvas masks, bd 8, AR weight 1.0, LR 5e-6 const, warmup 20), **100 steps**, checkpoints 50/100, evals (vLLM self-spec N=4/N=8 traces + AR guard) chained (jobs 3690–3696). Signal expected in ~2 h; bars: trida 2.3/2.4 tok/fwd (vLLM N=4/N=8), FLARE-4B 3.38 (SGLang g4). Truncation log: 17% of capped turns are loops; the rest are long deliberations that outgrow 16k. Note: the pilot data dir also holds the two `.truncated.jsonl` side-logs (rows without `messages`, skipped by the loader).

## 2026-09-10 — draft-align pilot: cross-node NCCL failure, TCP fallback

- Pilot train job 3697 (and resubmit 3704) died at the FIRST cross-node collective (`dist.barrier` after model
  sharding): `ncclRemoteError`, `NET/IB ... IBV_WC_RETRY_EXC_ERR(12)`. No rank crashed on its own; all exits
  were the watchdog SIGABRT / SIGTERM cascade.
- `train/tools/draftalign/nccl_probe.{sh,py}` (2-node srun, 16 ranks, 256 MB all-reduce):
  IB HCAs mlx5_2/3/4 -> RETRY_EXC_ERR both directions; RoCE mlx5_0/1 -> `ibv_reg_mr` ENOMEM (memlock);
  `NCCL_IB_DISABLE=1` (TCP over bond.2570, 10.121/16) -> OK, 36 ms per 256 MB all-reduce = 13.5 GB/s bus.
  Interpretation: the IB fabric between two GPU nodes is not passing RDMA right now although
  `ibstat` shows every port Active/LinkUp. hyungguk's v6-2n runs used the same nodes, so this is new.
- Fix: `train_2node.sbatch` now exports `NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=bond.2570` (overridable).
  Expected cost for FSDP 4B: ~32 GB/step over 13.5 GB/s ≈ +2–3 s on a ~36 s step.
- Pilot relaunched as job 3712 (canvas masks, bd 8, AR w 1.0, LR 5e-6, 100 steps, save 50/100 ->
  `luke/models/scratch/trida-4b-draftalign-pilot`); evals 3713–3718 (`pilot-s50`/`pilot-s100` × ss4/ss8/ar)
  chained with `scontrol update Dependency=afterok:3712` — `SBATCH_DEPENDENCY` was silently ignored, so
  `eval_ckpt.sh` gained a `DEP=` knob.
- Job 3712 passed NCCL init but died in step 1: node 1 (a GPU node) root disk 100% full ->
  Triton/inductor `OSError: No space left on device`. Culprits on the shared home's LOCAL disk: wandb
  artifact cache 200 GB (kept, user decision), root-owned HF download `Qwen3.8-Flash-Next` 107 GB (user OK'd
  removal but it is owned by root -> needs infra), uv cache 37 GB (kept). Removed: stale `/tmp/triton_cache_*`,
  `/tmp/torchinductor_*`, ~14k `/tmp/tmp*` dirs, `~/.cache/vllm/torch_compile_cache` -> 40 GB free (92%).
- Pilot relaunched as job 3719; evals 3720–3725 chained via `DEP=afterok:3719 eval_ckpt.sh`.
- Job 3719 ran but did <5 steps in 25 min (GPU mem 40 GB, bond traffic 1.8 GB/s): `--activation_offload`
  was on. hyungguk's final v6-2n runs are the "nooff" ones (`--fused_ce`, no offload; 30 s/step, 64–68 GB).
  sbatch: offload now opt-in (`OFFLOAD=1`). Pilot relaunched as job 3726; evals 3727–3732 chained.
- **Correction (user pushback "it worked before, no infra change" was right):** the IB fabric is fine.
  `ib_write_bw` with `--pkey_index=1` (partition 0x800c, full membership) does 43.5 GB/s between the nodes;
  pkey index 0 is 0x7fff = limited membership, and NCCL's default `NCCL_IB_PKEY=0` uses it -> RETRY_EXC_ERR.
  RoCE (mlx5_0/1) RDMA works at 11.3 GB/s/card — that is what hyungguk's runs used (IB rx counters are 0 since
  boot). Our-side cause of the RoCE failure: Tailscale-SSH logins have soft memlock 8 MB (PAM skipped) and
  sbatch propagates it to the ranks -> `ibv_reg_mr ENOMEM`. Fix: `ulimit -l unlimited` in the sbatch (done);
  transport choice pending `nccl_probe.sbatch` (job 3734, after 3726): roce / ib_pkey1 / default / tcp.
- Task created: `train/docs/TASK_OPTIMIZE_TRAINING.md` (user: "optimizing training code would be priority after
  this pilot run"). Trainer measured at ~8% MFU; facts + ordered plan recorded there.

## 2026-09-10 — draft-align PILOT verdict: flat (100 steps, 4,162 self-distilled agentic conversations)

Job 3726 (canvas suffix masks, bd 8, AR w 1.0, LR 5e-6 const, 20 warmup, 100 steps, ~20 s/step). Loss: diff
1.42 → 1.25, AR 0.19 → 0.077 (self-targets; floors differ from v6's random-mask 3.4 / 0.7). Evals: 30 GSM8K,
vLLM self-spec, traces split into cold steps (no specs to verify) and warm steps:

| ckpt | N | tok/fwd | cold % | warm slot-1 acc | warm all-accepted | GSM8K |
|---|---|---|---|---|---|---|
| step_18000 | 4 | 2.314 | 33.8 | 83.4 | 49.5 | 25/30 |
| pilot s50 | 4 | 2.352 | 33.3 | 85.1 | 50.7 | 25/30 |
| pilot s100 | 4 | 2.381 | 32.6 | 85.1 | 51.9 | 26/30 |
| step_18000 | 8 | 2.418 | 47.6 | 83.9 | 10.2 | 25/30 |
| pilot s50 | 8 | 2.399 | 47.9 | 83.7 | 9.4 | 25/30 |
| pilot s100 | 8 | 2.385 | 48.3 | 84.2 | 7.9 | 26/30 |
| AR greedy s50 / s100 | – | – | – | – | – | 25/30 / 24/30 |

Reading: N=4 drifts up +3% (monotone but within noise on ~4.5k steps); N=8 does not move / slightly down.
AR guard holds (±1 problem). Structural facts learned from the traces: a third (N=4) to half (N=8) of all
steps are COLD (previous step had a rejection → canvas [t0, MASK…] → exactly 1 token); tok/fwd is therefore
governed by the warm all-accepted rate (49.5% at N=4, 10% at N=8), which the fine-tune did not change.
Domain gap noted: training data is agentic multi-turn, eval is GSM8K.
Decision pending with the user. Options: (a) offline per-slot noisy-stream accuracy for step_18000 vs step_100
on held-out data inside the trainer's forward — separates "inference fidelity (canvas causal attention, block
alignment)" from "training/capacity"; (b) in-domain (agentic) tok/fwd eval; (c) resume generation + 300 steps.

**NCCL transport probe (job 3734, memlock unlimited, 256 MB all-reduce, 16 ranks):** IB pkey-index 1 +
RoCE 332 GB/s bus · IB pkey1 alone 176 · RoCE alone 30 · TCP 13.5. `train_2node.sbatch` now defaults to
`NCCL_IB_PKEY=1 NCCL_IB_GID_INDEX=3` (IB enabled). Optimize-training task is next (`train/docs/TASK_OPTIMIZE_TRAINING.md`).
- Autopilot after the pilot (user logged off): report artifact updated (section 11 pilot + 11a FLARE-4B rows);
  `train/train.py` gained `--profile_steps/--profile_dir` (per-phase wall clock data/fwd/bwd/optim + key_averages +
  chrome trace, rank 0); `train/tools/optimize_training/profile_1node.sbatch` runs canvas_bd8 and random_bd32 variants
  on one node (job 3735 → `luke/runs/trainprof_*`); `train/tools/draftalign/slot_diag.{py,sbatch}` = offline per-slot
  agreement of the noisy head with the clean head on the exact cold/warm canvases, bidirectional vs token-causal
  attention, step_18000 vs pilot step_100 (job 3736 → `luke/runs/slotdiag_*`). Both single-node, side by side.
  NCCL probe final: IB pkey1+RoCE 332 GB/s, IB pkey1 176, RoCE 30–35, TCP 14; NCCL default (pkey 0) fails.

## 2026-09-10 evening — offline per-slot diagnostic (job 3743) + training profile (3735/3738)

**Per-slot agreement of the noisy head with the clean (AR) head**, teacher-forced on step_18000's own GSM8K
greedy outputs (30 items; `train/tools/draftalign/slot_diag.py`; cold canvas = [1 clean, 2N−2 MASK], warm =
[N clean, N−1 MASK]; block = 2N−1):

| ckpt | attention in block | N=4 cold, slots 1..6 | N=4 warm, slots 1..3 |
|---|---|---|---|
| step_18000 | bidirectional (training) | 75.9 58.2 41.3 27.2 19.0 14.4 | 74.5 54.9 38.8 |
| step_18000 | token-causal (vLLM-like) | 74.3 52.4 32.0 19.7 15.4 11.6 | 74.0 51.7 33.2 |
| pilot s100 | bidirectional | 77.3 59.9 42.8 31.0 21.1 15.3 | 75.5 57.3 40.7 |
| pilot s100 | token-causal | 76.3 53.7 33.5 21.6 14.4 12.7 | 74.8 52.1 34.0 |

N=8 cold slot-1: 67.7 (base) → 70.0 (s100); later slots decay to ~5–10%.
Reading: (1) **not an inference-fidelity ceiling** — offline slot-1 agreement (74–77%) is no higher than the
online 84%, and causal-vs-bidirectional costs 0–2 points at slot 1 (5–9 at slot 3). (2) **It is capacity/
training**: the shared-weight noisy stream reproduces its own AR argmax only ~76% of the time one token ahead
with identical context; 100 fine-tune steps moved every slot by +1–4 points, consistent with the online
+2–3 points. Extrapolation says hundreds to thousands of steps for a large gain, if it keeps moving.
(3) The offline metric tracks online acceptance → usable as a cheap training-time metric.
Pitfall found: `forward_ar()` on the hybrid class is not usable while the two-stream wrappers are installed
(splits a single stream at T//2; odd L crashes); `forward_flare(..., return_clean_logits=True)` added instead.

**Profile (one node, 8 GPUs, canvas bd8, 2 steps):** 31.3 s/step = fwd 7.8 + bwd 22.2. Self-CUDA shares:
two-stream GDN bwd 26% (`fla_style_full_bwd_kernel`), NCCL all-gather 22% (waiting, not bandwidth), matmuls
18%, elementwise 18%, GDN fwd 9%, flex-attn bwd 5%. CPU: 89% `cudaStreamSynchronize`, half of it under
BlockTrainConvFunctionBackward via `bool(bad.item())` in `_validate_block_train_packed_boundaries` (per GDN
layer, again on checkpoint recompute) → opt-out `TRIDA_BLOCK_CONV_CHECK=0`. random_bd32 recipe: 53.8 s/step.
Variants: `--fsdp_keep_params` 31.1 (no gain), `--compile_mlp` 34.3 (worse in a 2-step window), both 31.0.
No-sync run queued (3739/3744/3745 died silently: `set -e` + empty-VEXTRA substitution; fixed).
- Profile variants (one node, canvas bd8 unless noted; baseline 31.25 s/step): no host sync 31.2 (0%);
  `--fsdp_keep_params` 31.1; `--compile_mlp` 34.3 (worse, 2-step window); both 31.0; all rows padded to 32k
  30.8 s with NCCL all-gather share 18.9% → 13.0% (rank imbalance confirmed as part of the "NCCL" time, but the
  extra padding eats the gain — the real fix is fuller/balanced packing, `PACK_EXAMPLES=20` test running).
  **random_bd32 without the host sync: 53.8 → 34.2 s/step** (the bd-32 `chunk_refine` backward had the CPU
  blocked in `.item()` 55% of the time); A/B re-check queued on one node (check=1 then check=0).
  GDN checkpoint-stride 2/4 variants queued (3750/3751).
- Round-1 optimization results consolidated in `train/docs/TASK_OPTIMIZE_TRAINING.md`: pilot recipe 31.25 → 29.4 s/step
  at +20% tokens/step (≈1.25× real throughput: pack 20 + 32k rows + compile_mlp); production bd-32 recipe 55.4 → 36.5
  s/step (≈1.5×) from removing the conv `.item()` host sync (same-node A/B). Stride knobs, keep_params, compile_glue:
  no gain. `train_2node.sbatch` defaults updated (BUCKETS=32768, PACK_EXAMPLES=20, TRIDA_BLOCK_CONV_CHECK=0, --compile_mlp).
  MFU still ~10%; round 2 = two-stream GDN backward kernel (23% of GPU time), selective checkpointing, balanced packing.

## 2026-09-10 20:20 — user decision: continue the drafter fine-tune with more steps, on autopilot

- Generation resumed (job 3755, both nodes): `gen_job.sh` got `RESUME_OFFSETS="3000 3200"` + `SHARD_SUFFIX=.part2`
  (existing shards are never rewritten); `gen_selfdistill.py` clips `max_tokens` to `ctx_len − prompt` (`--ctx_len 32768`),
  removing the ~3% context-overflow errors. Remaining stride-17 source: ~17k rows per node → ~3.5 h.
- `autopilot_long.sh 3755`: dataset `data-draftalign-v2/` (symlinks: v1 shards + part2) → train 3756 (`afterok:3755`,
  canvas bd 8, AR w 1.0, LR 5e-6 const, warmup 50, **1000 steps**, save every 250, round-1 speed defaults + IB pkey1) →
  evals 3757–3768 (`long-s{250,500,750,1000}` × ss4/ss8/ar) → `slot_diag` 3769 (step_18000 vs step_500 vs step_1000).
  Ids in `luke/models/scratch/autopilot_long.json`. Expected: gen ~3.5 h + train ~5–6 h + evals ~1.2 h → results
  ~2026-09-11 07:00 KST.
- Data note: with pack 20 / max_packed_rows 1 the long run will still cycle its ~40k conversations many times
  (~15 pulls per conversation over 1000 steps); acceptable for self-distillation but collator carry-over stays a TODO.

## 2026-09-11 07:00 — 1000-step run finished; live acceptance metric added; Plan A launched

- **Run 3756 (1000 steps) finished 06:39**, 4 checkpoints saved, 30.7k real tok/s on 16 GPUs (vs ~21k before the
  round-1 speed work: 1.46x confirmed end-to-end). **The diffusion loss plateaued by step 100** and then
  oscillated (1.335 / 1.212 / 1.366 / 1.329 / 1.202 / 1.219 at steps 50/100/250/500/750/1000): ~0.4B supervised
  tokens bought no descent after the first ~40M. Early evals: `long-s250` ss4 22/30, ss8 23/30 vs baseline 25/30.
- **Live draft-acceptance metric** added (`--acc_metric_n`, default 1024 in the sbatch): samples masked positions
  from the same forward and compares the NOISY head's argmax with the CLEAN head's = the AR-Trust accept test.
  Logged as `acc` / `acc_slot1` per step line and to wandb. Smoke (job 3770, step_18000, untrained):
  **acc_slot1 0.90-0.94, acc_all 0.61-0.67** — versus 0.76 offline (slot_diag) and 0.84 online (vLLM traces).
  The training objective is therefore measuring an EASIER task than the decoder runs: `_make_canvas_views` gives
  P(m=bd-1)=0.5 plus uniform m, so the average block leaves ~4 clean in-block tokens before slot 1, while the
  decoder's cold canvas (34% of steps at N=4, 48% at N=8) leaves exactly ONE. Mask-distribution mismatch is now
  measured, not hypothesised → strongest item for the next recipe change.
- **Plan A launched** (user's call, concern logged in `train/docs/DRAFT_ALIGN_NEXT.md`): job 3791, 3000 steps,
  canvas bd 8, AR w 1.0, LR 5e-6 cosine → 5e-7, warmup 100, save every 500 → evals 3792-3809 (`planA-s*`) →
  slot_diag 3810. Ids in `models/scratch/autopilot_trida-4b-draftalign-planA.json`. ETA ~18-24 h + ~3 h evals.
- Tooling fixes: `autopilot_long.sh` gained `TAG` (eval-tag prefix — the first Plan A submission reused
  `long-s500/1000` and would have overwritten run 3756's eval dirs; caught before it ran), plus `LR/LR_MIN_RATIO/
  WARMUP/BD/AR_W/DEPJ` knobs and a per-run `autopilot_<NAME>.json`.

## 2026-09-11 11:00 — FIRST POSITIVE RESULT: deploy-matched canvas masks (+ AUF)

Paired 100-step probes, 1 node each, identical except the loss weighting. Both use the **deploy-matched**
canvas: `BD=7 PCOLD=0.34 MWARM=3` — i.e. train on exactly what the N=4 decoder runs (7-slot canvas, 34% cold
[t0, MASK x6], 66% warm [t0, spec x3, MASK x3]), measured from the vLLM traces.

| ckpt | N=4 tok/fwd | N=8 tok/fwd | warm all-accepted (N=4) | GSM8K | AR guard |
|---|---|---|---|---|---|
| step_18000 baseline | 2.314 | 2.418 | 49.5% | 25/30 | 25/30 |
| old masks, 1000 steps (3756) | 2.368 | 2.284 | — | 23/30 | 23/30 |
| deploy-matched, 100 steps (probe-a-base) | 2.403 | 2.446 | 53.0% | 24/30 | 23/30 |
| deploy-matched + AUF 0.1, 100 steps (probe-d-auf) | **2.435** | **2.457** | **54.1%** | 24/30 | 24/30 |

- **100 steps of deploy-matched masks beat 1000 steps of the old distribution on every axis**, and fix the N=8
  regression the old recipe caused. Mask distribution >> step count.
- **AUF is a small positive**, consistent in sign across N=4, N=8 and warm-all-accepted (~1.2 sigma on 30 items,
  so suggestive not proven). I wrongly called it dead from the offline diagnostic, where it is IDENTICAL to the
  control — which is exactly right: AUF targets the JOINT, the per-slot diagnostic measures MARGINALS.
- **Metric lesson (3rd confirmation): offline slot agreement does not predict tok/fwd.** Run 3756 had a BETTER
  offline joint (0.179) than these probes (0.164) and a WORSE tok/fwd. The offline product assumes independent
  slots; the real all-accepted rate (34%) is ~3x what independence predicts, i.e. slots are strongly correlated
  and acceptance is context-driven. **Decide on decode tok/fwd only** (35 min, `eval_ckpt.sh` + `eval_ckpt_report.py`).
- Corpus note: v6 is NOT mainly agentic — sampled: 36% v11-fleet SFT, **26% KodCode (code)**, 9% agent-loop,
  then physics/nemotron/bfcl. Code is very low-entropy, which is why the training metric reads 0.93 slot-1 while
  GSM8K reads 0.79. Most of the drafting gradient is spent on tokens that are already correct → hard-example
  (clean-head-entropy) weighting is the natural next loss experiment.

## 2026-09-11 17:00 — VERDICT on the draft-align fine-tune: +5%, saturates at ~100 steps

Scaled run 3822 (deploy-matched canvas BD=7/PCOLD=0.34/MWARM=3 + AUF 0.1, LR 5e-6 constant, 2 nodes)
stopped at step 630 once the live metric had been flat for 380 steps. Checkpoints 250/500 evaluated.

| ckpt | N=4 tok/fwd | N=8 tok/fwd | GSM8K (30) | AR guard |
|---|---|---|---|---|
| step_18000 baseline | 2.314 | 2.418 | 25 | 25 |
| old masks, 1000 steps (3756) | 2.368 | 2.284 | 23 | 23 |
| deploy-matched, 100 steps (probe-a-base) | 2.403 | 2.446 | 24 | 23 |
| **deploy-matched + AUF, 100 steps (probe-d-auf)** | **2.435** | **2.457** | **24** | **24** |
| deploy-matched + AUF, 250 steps | 2.423 | 2.404 | 23 | 22 |
| deploy-matched + AUF, 500 steps | 2.441 | 2.421 | 22 | 24 |

**Conclusion:** the recipe is worth **+5.2% tok/fwd at N=4** (2.314 -> 2.435) and **saturates by step 100**.
Steps 100 -> 500 move tok/fwd by <1% (inside noise) while GSM8K decays monotonically 24 -> 23 -> 22. More
steps only erode base quality. **Best checkpoint = `probe-d-auf/step_100`.** Live metrics (acc/slot1/auf)
flattened at step 250 and correctly predicted the decode plateau this time.

Gap context: FLARE-4B 3.38 tok/fwd on SGLang vs trida 2.74 on SGLang / 2.435 here on vLLM. Fine-tuning closed
roughly a fifth of the shared-weight gap; the rest is main-training recipe (block size, loss weighting, AR
weight, data mixture, batch diversity — see the FLARE code comparison in this log) or an engine fix
(vLLM's causal canvas costs ~30% of joint acceptance) or a separate drafter.

**Remaining levers, ranked:** (1) data mixture — FLARE 40% long-reasoning / 40% math / 20% IF vs our single
agentic+code mix (26% code, very low entropy; training metric reads 0.93 slot-1 vs 0.79 on GSM8K);
(2) batch diversity — FLARE 256 rows x 4k vs our 32 rows x 32k per step, same tokens, 8x fewer independent
mask draws; (3) draft trees (arXiv 2606.01813) — the only idea that attacks the 43% of steps that accept
NOTHING; (4) bidirectional canvas in vLLM — ~30% of joint acceptance, zero training.

## 2026-09-14 — nano plan for inference/vllm/

`inference/sglang-backend/` renamed to `inference/sglang/` (two doc refs updated). User: the sglang side is
already nanoGPT-shaped; vllm is not; "we need a thorough plan". Plan written: `docs/NANO_PLAN.md` (moves out
with the rest of docs/ in its own step 2). Inputs: a consistency/voice review of inference/ (page:
claude.ai/code/artifact/56a57138…) and a line-by-line map of the plugin. Map headlines: model file 1,947 lines =
25% debug, 19% parked diffusion mode, 18% shipped self-spec, 3% dead; kernel file 1,172 lines of which the
shipped path calls ~145 (the "verbatim SGLang port" 630 lines are never reached); of 18 TRIDA_* env knobs
exactly one (`TRIDA_SELFSPEC_N=4`) is ever set in production. Target: ~8 files, ~650 lines Python + one Triton
kernel, `serve.py --backend vllm` on a backend-neutral YAML, byte-identical token stream vs a `vllm-plugin-pre-nano`
tag as the acceptance bar. Also found: `serve_diffusion.sh` launches the parked diffusion-only mode, not self-spec.

## 2026-09-14 — long-horizon probe: out-of-order commit is dead; two cluster facts

**Long-horizon determinacy probe** (job 3843, ran 09-13 11:55, `train/tools/draftalign/longhorizon.py`; the
sbatch's output filter dropped the indented result lines — read `$OUT/bd*.log` directly). Question: are far-future
positions ever both confident and correct, so a diffusion decoder could commit them out of order? **No.**
step_18000, cold canvas after a clean seed, GSM8K self-text:

| offset k | bd8 acc | bd32 acc |
|---|---|---|
| 1 / 2 / 3 | .730 / .570 / .415 | .658 / .506 / .341 |
| 4 / 5 / 6 / 7 | .257 / .175 / .145 / .119 | .157 / .124 / .116 / .127 |
| 9–31 (bd32) | — | .033–.099, mean ≈ .057 |

Confidence buckets: mid (k 4–8) 90–95% of positions < 0.5 conf; the 0.3% at ≥ 0.9 are 68% right. **Far (k 9–31):
99.4% of positions < 0.5 conf, acc .057; the 0.6% that clear 0.5 are 0–8% correct.** There is no population to
harvest. The first-principles idea (break the contiguous-draft assumption) has no material on this model; the
prefix is the only thing the drafter knows. Idea closed. This also caps "bigger N": marginal accuracy is unigram-
level past k≈8, consistent with the hazard decay the stats red-team measured.

**Cluster fact 1 — dangling plugin install.** `env/vllm-uv27` has an EDITABLE install of `vllm_native_diffusion`
0.0.1 whose finder points at `luke/mv3/inference/vllm/vllm_native_diffusion` — a directory that no longer exists.
`python -c "import vllm_native_diffusion"` fails in the venv. Serving works only because `scripts/serve_diff_cl.sh`
prepends `code/vllm-native-dev9` to PYTHONPATH. Folded into NANO_PLAN step 7 (uninstall, install from canonical,
drop the PYTHONPATH line). Launcher also pins `--max-num-seqs 1`, gpu-mem 0.55, CUDA-compat LD path — README omits.

**Cluster fact 2 — hyungguk's live run is not what the user believed.** Job 3844 `q35-flare-v6-posdecay`
(sbatch `trida-stack-run/slurm/train_qwen35_flare_v6_2node_posdecay.sbatch`): `--bd_size 32 --ar_loss_weight 0.1
--loss_weighting uniform --pos_decay_gamma 8 --lr_min_ratio 0.1 --activation_offload`, weights-only init from
step_18000, "DFlash intra-block positional loss decay experiment". So: uniform weighting (one of our three findings)
YES; **AR weight is still 0.1, not 1.0; block size still 32; and `--activation_offload` is on (the flag we measured
at ~6x slower)**. Previous run 3842 `-uniform` (28 h) was replaced by this one ~09-14 05:00. Long-horizon job 3845
`eval-commit-ckpt` in his queue is his, not ours.

## 2026-09-15 — hyungguk's runs, read from the cluster

- **3844 posdecay (bd 32, AR 0.1, uniform, pos_decay_gamma 8, offload ON) finished** at its 2000-step cap.
  step_1000 → step_2000 took 11 h 26 m = **41 s/step** vs his nooff runs' ~30 s → offload cost him **~1.4x**, not
  the ~6x I measured on MY bd-8 canvas run (3719). The 6x was recipe-specific; correction logged. diff loss
  1.62 → 1.31 over 2000 steps (uniform-weighted; not comparable to v6's 1/γ-weighted plateau).
- **3856 eval of posdecay step_2000, self-spec, agentic benches:** per-sample tok/fwd 1.96 (median 1.78);
  bfcl 2.06 / functionchat 1.96 / koab 1.68 / tau2 2.09. **No baseline (step_18000) run on the same benches
  exists in his slurm dir**, so whether positional decay helped is undetermined from his data. Our reference:
  step_18000 = 2.74 tok/fwd on SGLang GSM8K; agentic text is different material.
- **3859 fullmask-bd8 RUNNING** (started ~09:20, both nodes): `--full_mask` (100% of each 8-token block masked,
  γ=1, single noisy view), bd 8, uniform, AR 0.1, LR 5e-5, 4000 steps, save every 2000, offload ON,
  weights-only from step_18000. ~37 s/step → ~41 h → done ~09-17 early. This is the "cold canvas at bd 8"
  shape our traces said is 34% of decode steps; ours (bd 7, P(cold)=0.34 + warm m=3, LR 5e-6) gave +5% and
  saturated by step 100 — worth telling him to evaluate at step_2000 rather than wait for 4000.

## 2026-09-16 — posdecay quality collapsed; fullmask-bd8 shares the risky settings

- **posdecay step_2000 (3844) quality, from his own evals (3855 diffusion mode, 3856 self-spec):**
  FunctionChat dialog avg(micro) **0.445** (self-spec) / 0.45 (diffusion) vs step_18000 **0.905** (our full eval);
  tau2 pass@1 **0.35 / 0.50 / 0.30** (airline/retail/telecom, N=20 each) vs step_18000 self-spec **0.90** (his 3530, N=20);
  mmlu_pro diffusion bd32 **41.85%** with **43.9% truncated** (avg_gen 14,993 tokens — runaway generations).
  The 1.96 tok/fwd is moot at that quality. LR 5e-5 constant-then-cosine from step_18000 with AR weight 0.1 (a weak
  anchor) — 10x our LR — is the likely cause; our 5e-6 runs cost 1–3 GSM8K problems, his cost half the benchmark.
- **3859 fullmask-bd8** (started 09-15 19:07, both nodes): step 1410/4000 at 09:56, 37.9 s/step; diff loss
  3.01 → 2.38 by step 400 then flat 2.38–2.50 through 1410 (same early plateau as every fine-tune we ran).
  Uses the SAME LR 5e-5 and AR 0.1 as posdecay → same quality risk. step_2000 saves ~16:10 today; end ~09-17 13:00.
  Recommend to hyungguk: eval step_2000 for QUALITY first (FunctionChat dialog vs 0.905), and add a step_18000
  baseline through the same tok/fwd harness — none exists (his 09-08 tau2 evals recorded reward only).

## 2026-09-16 — first-principles discussion + literature check (page: claude.ai/code/artifact/645592fd…)

Theory: same-step token independence is the core failure, worst for adjacent tokens (supported: ParallelBench,
ReFusion, Gu 2018). Currencies = iteration / verification / TRAINING (dParallel). **Our claim that diffusion excels
at schema-bound tool calls is REFUTED** (Bitter Lesson of dLLMs on BFCL; DINGO +68) — schema fixes token type,
not value. Plan-then-fill validated (Planned Diffusion 1.3–1.8x; ADLM anchors). Two-sided context → ~16 tokens
but only with known gap length. Five layers of sequentiality; block≈4 ≈ one head + dependents. Implication for
trida: verified path for causal work, constrained decoding on the parallel path, pure parallel for non-causal
subtasks only. Memory: diffusion-language-theory.md.

## 2026-09-16 — off-path:on-path ratio measured; we are at the ALGORITHM's ceiling, not diffusion's

From existing self-spec traces (no GPU, `/tmp/offpath.py`): accepted draft = off-path (context-determined),
rejected draft = a choice. GSM8K/30: base N4 **r = 3.93** (79.7% off-path), base N8 3.01, tuned N4 4.58, tuned N8 3.08;
per-request p10/med/p90 at base N4 = 2.89/3.65/6.21. Floor, not estimate (weak drafter miscounts consequences as
choices; deeper probing finds more choices).

**Ceiling arithmetic.** r ≈ 3.9 → 1 choice + ~4 consequences per group.
one forward per choice + one for consequences = (1+r)/2 = **2.47 tok/fwd**; we measure **2.44**.
consequences riding along, no forward burned on a choice = (1+r) = **4.93**.
We are AT the first bound. That retro-explains the month: draft-align +5% then saturation, adaptive N 0.3%,
AUF within noise — none of them changed the regime. Remaining ~2x = the cold step (34%/48% of forwards, 1 token each).
Prices carry-forward and draft trees at up to 2x. Page section 8: claude.ai/code/artifact/645592fd…

## 2026-09-16 — hyungguk's full-mask bd8 step_2000: +17% tok/fwd, and it moves the ceiling

3859 was cancelled at step 2000 of 4000 (only step_2000 exists). Evaluated on our harness, N=4, 30 GSM8K:

| ckpt | tok/fwd | r | %off | cold | warm all-acc | (1+r)/2 | (1+r) |
|---|---|---|---|---|---|---|---|
| step_18000 | 2.314 | 3.93 | 79.7 | 33.8% | 49.5% | 2.47 | 4.93 |
| our draft-align tune | 2.435 | 4.58 | 82.1 | 31.7% | 54.1% | 2.79 | 5.58 |
| **fullmask-bd8 s2000** | **2.717** | **6.83** | 87.2 | 25.4% | 66.3% | 3.91 | 7.83 |

GSM8K 24/30 (base 25/30) → quality intact. accept-hist 0:35 1:8 2:8 3:49 (base 0:43 1:12 2:11 3:34).
**Confirms the paper's prediction that r is drafter-dependent, not a property of the text alone.** Also:
2.717 on vLLM ≈ step_18000's 2.74 on SGLang, i.e. his training bought roughly what the engine's
bidirectional canvas is worth — and the two should compose.
**Consequence for us:** this checkpoint sits at 69% of its same-regime bound where the base sat at 99%,
so the cold step is worth MORE now, not less. Experiments 3870 (E1/E2/E3/E4) switched to run on it.
Also vindicates the deploy-matched-mask finding at 2000 steps: full mask at bd 8 = the cold-canvas shape.
