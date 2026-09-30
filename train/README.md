# `train/` — block-diffusion SFT

Converts a pretrained **autoregressive** HF checkpoint into a **block-diffusion** LLM by SFT, with an
AR auxiliary loss so the same weights keep a working causal head. Native-torch **FSDP2** (bf16
compute / fp32 grad reduce), per-layer gradient checkpointing, manual grad accumulation, HF
`save_pretrained` checkpoints. See the repo root README for the concept; this file is the practical
map of the stack.

## Launching

Always launch through the **top-level `train.py` shim** (running `train/train.py` directly puts
`train/` itself on `sys.path[0]`, where the file shadows the package):

```bash
torchrun --nnodes=1 --nproc_per_node=8 train.py \
  --model_id Qwen/Qwen3.5-4B \
  --dataset <hf-dataset-or-local-jsonl-dir> \
  --pack --multiturn --keep_all_reasoning \
  --linear_block_mode bidirectional \
  --bd_size 32 --ar_loss_weight 0.1 --fused_ce --compile_mlp \
  --max_length 32768 --length_buckets 2048,4096,8192,16384,20480,24576,28672,32768 \
  --micro_batch_size 1 --grad_accum 2 \
  --save_dir checkpoints/<run> --save_every 1000 --save_optim_state
```

**Resume:** each checkpoint dir is self-contained (resized weights, tokenizer with the added
`<|mask|>` token, `block_diffusion.json`, and — with `--save_optim_state` — a DCP optimizer state +
`global_step`). Resume exactly with `--model_id <ckpt_dir> --resume <ckpt_dir>`; the `<|mask|>`
re-add is idempotent on reload.

## Model families & routing

`train.py` auto-routes on the model config (`is_hybrid_model_id`):

| family | trainer class | layout |
|---|---|---|
| dense (full attention, e.g. **Qwen3**) | `HFBlockDiffusion` | `[S \| x_t \| x_0]` per row (root README) |
| hybrid GDA (**Qwen3.5**: gated-delta linear layers + periodic full attention) | `HFBlockDiffusionHybrid` | FLARE two-stream, below |

### The hybrid path (FLARE two-stream)

A recurrent linear-attention layer cannot be masked into within-block bidirectionality, so the
hybrid trainer runs **two streams** through one layer call, layout `[x0 ; xt_1 ; … ; xt_V]`:

- **clean `x0`** — ordinary causal gated-delta pass; provides the AR loss and the per-block
  boundary states;
- **noisy views `xt_v`** — each block re-initialized from the preceding clean boundary state and
  read out bidirectionally within the block. The V complementary views **share one clean stream**
  (computed once — projections/MLP/attention/AR head, and inside the recurrence one clean pass +
  one set of refined boundary states feed all views; exact, fp32-verified).

Implementation seam: the layers' `chunk_gated_delta_rule` / `causal_conv1d_fn` are rebound per
layer (`_install_two_stream_scan`) to the vendored kernels; full-attention layers get a flex
`BlockMask` built by `_flare_block_diff_mask_mod`. `linear_block_mode`: `bidirectional` = the real
method; `causal` = the legacy approximation.

## Data pipeline (`data/text_sft_data.py`)

Chat-format JSONL (`messages`, optional `tools`) → chat-template rendering → length-bucketed rows.
Collators, selected by flags:

| flags | collator | rows |
|---|---|---|
| *(none)* | `TextSftCollator` | single-turn, one convo/row |
| `--multiturn` | `MultiTurnTextSftCollator` | every assistant turn supervised |
| `--pack` | `PackedRowDataset` + `StreamPackedCollator` | packed single-turn |
| `--pack --multiturn` | `PackedMultiTurnCollator` | **packed whole conversations** (hybrid/bidirectional only) |

Packing is **document-isolated** three ways, all keyed off `seg_id`: the gated-delta recurrence +
short-conv reset at every `cu_seqlens` boundary, the flex mask confines attention per document, and
RoPE positions restart per document. Docs are padded to `bd_size` multiples so packed doc starts
land on block boundaries (kernel requirement). Verified: perturbing one packed document changes the
other's logits by exactly 0.

## Flags that matter

- `--bd_size` block length; `--ar_loss_weight` AR aux weight (`loss=(diff+w·ar)/(1+w)`);
  `--loss_weighting weighted` = 1/γ ELBO weighting of the diffusion CE.
- `--linear_block_mode {causal,bidirectional}` (hybrid only).
- Throughput (all numerically exact, measured on 4B/32k/16×H100 — ~×1.7 supervised tok/s per GPU
  over the naive config): `--fused_ce` (Liger fused CE), `--compile_mlp` (torch.compile of MLP+norms
  only — whole-layer compile recompiles per batch on packed data; `--compile_glue` exists but
  measured no further gain), fine-grained `--length_buckets` (padding was ~17% of compute),
  **omit** `--activation_offload` unless memory-bound (the CPU round-trip costs ~20%).
- `--save_optim_state` for exactly-resumable checkpoints; `--fsdp_keep_params` keeps gathered
  params between fwd/bwd (measured unnecessary on 2 nodes — inter-node comm ≈ 0).

## Third-party kernels — `block_gated_delta_rule/` (fetch these first)

The two-stream block-causal gated-delta kernels are **not vendored here**. They are
**PolyForm Noncommercial 1.0.0** upstream (FLARE/HybridDiffusion), which this Apache-2.0 repo
cannot redistribute, so the directory ships a pinned fetch-and-patch recipe instead:

```bash
bash train/block_gated_delta_rule/fetch_kernels.sh
```

It clones `yuchen-zhu-zyc/HybridDiffusion@6ca547a`, copies the kernel package in, and applies
`trillion_mods.patch`. Until you run it, the hybrid path (`hf_block_diffusion_hybrid.py`,
`forward_flare`) will not import. See
[`block_gated_delta_rule/README.md`](block_gated_delta_rule/README.md) for the by-hand equivalent,
and `LICENSE.HybridDiffusion` / `NOTICE.HybridDiffusion` / `VENDORED.md` for the license terms —
the fetched code is noncommercial and not covered by this repo's license.

Once fetched, the kernels are relative-import-only and self-contained (torch + triton + fla). Note
for anyone debugging them: test through the real model path — synthetic random `g` (the log-decay
gate must be ≤ 0) produces garbage that looks like kernel bugs.
