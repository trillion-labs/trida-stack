# vLLM native block-diffusion backend

A self-contained **vLLM 0.27.x** backend that serves the Qwen3.5 two-stream
**block-diffusion** checkpoint **natively**, via vLLM's in-tree `ModelState`
diffusion path — no vLLM fork, no source build.

It is an alternative to the SGLang serving path in [`../`](../): same checkpoint,
same two-stream diffusion decode, but running inside a stock vLLM install as an
out-of-tree plugin (arch `Qwen3_5ForBlockDiffusion`). See [`DESIGN.md`](DESIGN.md)
for how it maps the two-stream GDN clean/noisy passes onto vLLM's block-diffusion
`ModelState`.

## Install

```bash
pip install vllm==0.27.*          # a stock vLLM 0.27.x wheel
pip install -e inference/vllm     # registers the `trida_diffusion` plugin
```

The editable install wires an entry point in the `vllm.general_plugins` group.
vLLM runs that plugin in **every** process (launcher, EngineCore, workers), which
is what registers the `Qwen3_5ForBlockDiffusion` architecture and installs the
GDN two-stream readout patch where inference actually happens.

## Serve

```bash
bash inference/vllm/serve_diffusion.sh                        # trillionlabs/Trida2.0-4B, self-spec, one stream
MAX_NUM_SEQS=8 bash inference/vllm/serve_diffusion.sh         # self-spec, batched (PIECEWISE cuda graphs)
MODE=causal bash inference/vllm/serve_diffusion.sh            # plain autoregressive (stock vLLM, no plugin)
CKPT=<hf-repo-id-or-local-path> bash inference/vllm/serve_diffusion.sh   # any other checkpoint
```

`CKPT` (an HF repo id or a local checkpoint path) falls back to `$TRIDA_MODEL`,
then to `trillionlabs/Trida2.0-4B` — private on Hugging Face during preview, so (with repo access) run
`huggingface-cli login` or export `HF_TOKEN` before the first download. Everything
else is env-driven with sane defaults:

| env | default | meaning |
|---|---|---|
| `CKPT` | `$TRIDA_MODEL`, else `trillionlabs/Trida2.0-4B` | HF repo id or local path to the checkpoint |
| `MODE` | `self-spec` | `self-spec` \| `causal` \| `diffusion` (pure denoising; not supported by the Trida2.0-4B recipe) |
| `PORT` | `8000` | server port |
| `SERVED_NAME` | `trida` | model id exposed on `/v1/models` (`chat.py`/`eval.py` pick it up automatically) |
| `MAX_NUM_SEQS` | `1` | concurrent sequences per replica; `1` = latency mode |
| `CUDAGRAPH` | *(by mode)* | self-spec: `FULL_AND_PIECEWISE` at every concurrency (lossless at C>1 since the int32 causal-buffer fix); diffusion: `PIECEWISE`. `NONE` = eager |
| `SELFSPEC_N` | `4` | self-spec draft block: canvas `2N−1` = the model's **block of 8** at N=4 |
| `TRIDA_SS_SAMPLE` | `1` | `1` = rejection sampling with the request's sampling params (SGLang-equivalent); `0` = exact-greedy verify |
| `TRIDA_SS_DRAFT` | `sampled` | draft proposal: `sampled` from the truncated top-k/top-p distribution, or `argmax` (one-hot). Measured equal acceptance (2.2–2.3 tok/fwd) under rejection verify; kept for experiments |
| `TRIDA_SS_DRAFT_FUSED` | `1` | fused Triton draft-proposal sampling (temperature → top-k/p → Gumbel-max → q write) + Triton index-prep kernel; `0` = the previous torch path |
| `TRIDA_SS_DRAFT_TEMP` / `TRIDA_SS_GATE` | off | experiment knobs: proposal temperature (q only) and a draft confidence gate. Measured: temperature is neutral (≤ +0.1 tok/fwd), a 0.9 gate *lowers* tok/fwd by 0.2–0.5. Acceptance in speculative sampling is 1 − TV(p, q): the drafter, not the draft policy, sets it (~2.25 tok/fwd here) |
| `MASK_ID` | `248077` | mask token id of the checkpoint |
| `LOG` | *(unset)* | also write the server output to this file — feed it to `eval.py --server-log` for tokens per forward |
| `MAX_MODEL_LEN` / `GPU_MEM_UTIL` | `16384` / `0.8` | passed through; prompt + `max_tokens` must fit in `MAX_MODEL_LEN` or vLLM returns 400 |
| `CANVAS` / `THRESHOLD` / `MAX_STEPS` | `3` / `0.95` / `8` | `MODE=diffusion` only: block B=4 canvas, commit gate, max denoising steps |

The script sets `VLLM_PLUGINS=trida_diffusion` and serves an OpenAI-compatible
endpoint at `http://<host>:<PORT>/v1/chat/completions`. It runs with a plain
`vllm` on PATH. On older GPU drivers you may additionally need a CUDA compat shim
on `LD_LIBRARY_PATH` — export it yourself before running.

### Verify it's running

Once the server logs `Application startup complete`, confirm it generates:

```bash
curl -s http://localhost:${PORT:-8000}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"trida-bd","messages":[{"role":"user","content":"What is 17 * 24?"}],"max_tokens":64}' \
  | python -m json.tool
```

A JSON reply with a `choices[0].message.content` field means the backend is
serving. `model` must be `trida-bd` (the served name the script sets).

## Decode modes

| mode | how | status (step_18000, H100) |
|---|---|---|
| `causal` | stock vLLM, no plugin needed | reference: 220 tok/s @C=1, 2889 @C=16 |
| `diffusion` (block 4) | canvas as draft tokens, block-END GDN readout, confidence commit gate | lossless vs the SGLang reference; ~1.4 tokens/forward on this checkpoint, slower than AR |
| **`self-spec`** (AR-Trust) | diffusion draft + exact greedy AR verification; the step is presented to vLLM as a spec-decode batch (bonus token + 2N−2 drafts) so **FULL cuda graphs** capture the whole forward | **≥ vLLM AR at every concurrency**: 351 / 873 / 1841 / 2886 tok/s at C=1/4/8/16 (1.60× / 1.19× / 1.17× / 1.00×), GSM8K 81.7% vs AR 81.3%, 7.5 ms per step at N=4 |

### Self-spec — results (Trida2.0-4B, block 8, canonical sampling, GSM8K 200, one H100, FULL graphs)

| C | causal AR agg / per-req tok/s | self-spec agg / per-req tok/s | p50 latency | accuracy |
|---:|---:|---:|---:|---:|
| 1 | 213 / 210 | **311 / 327** | 1.29 s | 85.5–89.5 % (AR 89.0 %) |
| 2 | 412 / 204 | **583 / 310** | 1.33 s | |
| 4 | 745 / 188 | **966 / 271** | 1.47 s | |
| 8 | 1251 / 168 | **1495 / 222** | 1.85 s | |
| 16 | 1690 / 141 | **1854 / 161** | 2.52 s | runaways ≤ 0.5 % |

Speculative rejection-sampling verify (same output distribution as sampling the causal model), ≈ 2.27 tokens per
forward; the speed comes from the step cost — FULL cuda graphs at every concurrency plus the fused Triton
draft-proposal / index-prep kernels (`TRIDA_SS_DRAFT_FUSED=1`). Validation: greedy C=1 byte-identical to the
previous build; sampled path distributionally equal to it and to causal sampling (unigram/bigram JSD 0.029/0.140
vs 0.026/0.135). Job 4216, 2026-09-18.

### Self-spec — launch by hand

`serve_diffusion.sh` (default mode) expands to this; shown for when you drive `vllm serve` yourself:

```bash
# N = draft block (gen) size; canvas is 2N-1 slots (= the model's block of 8 for N=4), and the config
# value is 2N-2 because vLLM adds the bonus token. FULL_AND_PIECEWISE for one stream; PIECEWISE for --max-num-seqs > 1.
TRIDA_SELFSPEC_N=4 vllm serve <ckpt> --served-model-name trida \
  --hf-overrides '{"architectures":["Qwen3_5ForBlockDiffusion"],"canvas_length":6,"mask_id":248077,"confidence_threshold":0.90}' \
  --diffusion-config '{"canvas_length":6,"max_denoising_steps":8}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --max-num-seqs 1 --trust-remote-code
# chat with tool calling / thinking:
#   --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

Full eval on the same checkpoint (greedy, thinking on) — GSM8K / FunctionChat singlecall·dialog·calldecision / Ko-AgentBench task-weighted SR:
AR 78.2 / 0.922·0.905·0.947 / 0.582; N=4 77.8 / 0.920·0.905·0.941 / 0.615; N=8 77.0 / 0.918·0.895·0.946 / 0.615;
N=32 77.6 / 0.908·0.885·0.942 / 0.549.

### Batched serving: what we fixed in the plugin (2026-09-18)

Until now batched self-spec on vLLM corrupted the second co-scheduled request (fixed token `14162`, seq_len
rewound) and crashed at ~8 rows. Both were traced to the **mixed step** — a request finishing its prompt in
the same forward as other requests' canvases — and fixed in `qwen3_5_diffusion.py` (4 hunks, +39/−3):

1. **Sampler had no mixed-batch path.** The 1-logit prompt row was never recognised as a prefill, so the
   request never received its seed, block init, or GDN ring import and decoded off stale warmup state.
   Rows are now classified with the scheduler's own `is_prefilling_np`; prompts are finished (seed → bonus,
   `init_block`) and canvases re-proposed unchanged on mixed steps.
2. **Stock spec-decode GDN kernel fed a 1-column state-index tensor.** On mixed steps the canvas rows fall
   back to vLLM's stock spec kernel, which expects `num_spec+1` per-step state slots; our block table has one
   block per request, so the builder produced `[nspec, 1]` while the kernel indexed 7 columns → illegal
   memory access (the old "N=8 crash", which was never graph-only). The plugin now hands it a proper
   `[nspec, 2N−1]` tensor (column 0 = the real state row, the rest the null row, whose stores are skipped).

Validated greedy, byte-for-byte against C=1: C=2/4/8 identical under **PIECEWISE** and eager, including an
8→1 drain over 200–900-token outputs; C=8 GSM8K 60 problems ran with 0 errors.

### Known limits

- **FULL cuda graphs at C>1 — fixed 2026-09-18.** Root cause: the plugin handed FlashAttention a *bool*
  per-sequence causal buffer; the builder's `.to(torch.int32)` allocated a fresh tensor every step and a FULL
  graph baked the capture-time pointer, so at replay row 1+ read stale memory as `dynamic_causal` (row 0 stayed
  correct, which is why C=1 always worked). `_causal_buf` is now int32 (persistent), one-line fix. Validated
  greedy byte-for-byte FULL == PIECEWISE at C=1/2/4 and all 8 rows of an 8→1 drain (posdecay; N=4 and N=8), FULL
  C=1 unchanged. `serve_diffusion.sh` still defaults batched serving to PIECEWISE until the sampled grid below
  has been repeated on a second checkpoint; set `CUDAGRAPH=FULL_AND_PIECEWISE` to take the speedup now.
  Sampled matched grid (canonical sampling, GSM8K 200, 8192 budget, one H100, fullmask-bd8):

  | C | self-spec FULL tok/s (acc) | self-spec PIECEWISE tok/s (acc) | causal AR tok/s (acc) |
  |---:|---:|---:|---:|
  | 1 | 211 (85.5%) | 158 (88.5%) | 213 (85.5%) |
  | 2 | 399 (88.0%) | 319 (87.5%) | 413 (90.0%) |
  | 4 | 688 (88.0%) | 511 (87.0%) | 752 (84.5%) |
  | 8 | 1181 (86.5%) | 942 (88.0%) | 1158 (85.0%) |
  | 16 | 1633 (86.5%) | 1646 (84.5%) | 1779 (89.0%) |

  FULL is ≈1.3× PIECEWISE up to C=8 and converges at C=16. Under canonical sampling tok/fwd is ≈2.28 (vs
  2.81 greedy; the server counter excludes the bonus token, `eval.py` adds it), which puts self-spec at **AR parity**, not above it — acceptance under sampling is the lever
  from here, not graphs.
- **Self-spec sampling = SGLang's (since 2026-09-18).** Per-request `temperature`/`top_k`/`top_p` are honored
  through vLLM's own `RejectionSampler` (speculative rejection sampling, drafts sampled from the truncated
  top-k/top-p proposal), so vLLM and SGLang self-spec are now the *same decoder*. `TRIDA_SS_SAMPLE=0` restores
  the exact-greedy verify (byte-identical to the previous build at temperature 0). Matched run, GSM8K 200
  prompts, C=16, canonical sampling (temp 1.0 / top_k 50 / top_p 0.95), 8192 budget, one H100:

  | server | accuracy | hit max_tokens | agg tok/s | per-request tok/s | tok/fwd |
  |---|---:|---:|---:|---:|---:|
  | vLLM causal | 86.5% | 0.5% | 1828 | 139 | 1.0 |
  | **vLLM self-spec N=4 (block 8), sampling** | **87.5%** | **0.0%** | 1323 | 123 | 2.29 |
  | SGLang self-spec b7/g4 | 80.0% | 5.5% | 1320 | 78 | 2.37 |

  Under PIECEWISE (required for C>1 today) batched self-spec sits at SGLang's aggregate throughput and below
  AR; tok/fwd drops from ~2.8 (greedy) to ~2.3 under sampling, as stochastic acceptance implies; per-request forward cost is the same on both engines at C=1 (≈10.3 ms), so the remaining C=16 gap is SGLang's batched GDN state commit (~28% of its step).
- **Greedy is exact only up to bf16 numerics.** Teacher-forcing self-spec output through stock causal shows
  agreement with AR's argmax at all but 1–4 positions per 2048 tokens, each with a top-2 margin ≤ 0.125 nats
  (several exact ties); the flips depend on the kernel path (eager ≠ graphed), and stock causal AR is not
  batch-invariant either. On this checkpoint a near-tie can tip a greedy trajectory into a deliberation loop
  that runs to `max_tokens` (~17% of GSM8K prompts, no-think), which is why sampling — not EOS handling —
  is the remedy.
- Prefix caching is **not** supported by the plugin (enabling it corrupts the GDN state rows).
- Canvas attention is fully causal on FlashAttention 3 (MASK rows do not see later drafts), so drafts are
  on par with the SGLang reference once measured per request (2.27 vs 2.36 tokens per forward at block 8 under
  the same sampling).
- Pure diffusion: PIECEWISE cuda graphs (the GDN two-stream op runs eager between graphed segments); not
  supported by the Trida2.0-4B full-mask recipe.

## Test

CPU-only, no GPU required — pins the two-stream GDN snapshot/restore invariant and
the commit gate:

```bash
python -m pytest inference/vllm/vllm_native_diffusion/test_two_stream_cpu.py
```

## Layout

```
inference/vllm/
  README.md
  DESIGN.md                       # architecture: two-stream GDN → vLLM ModelState
  pyproject.toml                  # the plugin package
  serve_diffusion.sh              # env-driven `vllm serve` launcher
  vllm_native_diffusion/
    __init__.py
    plugin.py                     # vllm.general_plugins entry point (register_trida)
    qwen3_5_diffusion.py          # the ModelState + diffusion sampler + GDN patch
    block_causal_readout.py       # Triton kernels: block-end readout, fused self-spec GDN layer (derived from FLA)
    test_two_stream_cpu.py        # CPU correctness test for the decode crux
```
