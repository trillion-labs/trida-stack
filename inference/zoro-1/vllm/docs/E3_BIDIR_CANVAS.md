# E3 — bidirectional mask slots in the self-spec canvas

Status 2026-09-16: **core math implemented and CPU-verified; integration hook identified; kernel
call not yet written.** Worth ~19% (2.31 → 2.74 tok/fwd on identical weights, engine difference only).

## The problem

Canvas is `B = 2N-1` slots: `[t0, spec_1..spec_{N-1}, MASK_1..MASK_{N-1}]`.

- Slots `0..N-1` carry real tokens and their logits **verify** the specs. They must stay strictly
  causal. If slot `j` could see slot `j+1`, its logit would see the very token it is asked to
  predict, the accept test becomes vacuous and the output stops being AR-greedy.
- Slots `N..B-1` are MASK and only **produce** next-step drafts. They are verified next step anyway,
  and training gave them bidirectional attention within the block. Running them causally is a
  train/inference mismatch and costs draft quality — measured offline at N=4:

| | slot 1 | slot 2 | slot 3 |
|---|---|---|---|
| bidirectional (as trained) | 0.795 | 0.616 | 0.474 |
| causal (what vLLM runs) | 0.783 | 0.585 | 0.391 |

vLLM's FA backend takes `causal` as a bool **or a per-request tensor** — but never per *position*
within one request's query block, which is exactly what we need.

## The mechanism

The causal pass already gives each mask slot the prefix, the verify slots, and all **earlier** mask
slots. The only missing edges are mask → **later** mask. Those keys are in the canvas itself, so the
correction is an `[M x M]` strictly-upper-triangular attention over `M = N-1` rows, merged into the
causal result by log-sum-exp:

```
out = (o_causal * e^{l_causal} + o_future * e^{l_future}) / (e^{l_causal} + e^{l_future})
```

Exact, not approximate: it is one softmax over the union of two disjoint key sets. Cost is O(N²·d)
on a 7- or 15-token block.

**Verified on CPU** (`tools/experiments/bidir_merge.py`, run it): merged output matches a
ground-truth mixed mask to 7e-16 at N ∈ {4, 8, 16}; the verify half is **bit-identical** to the pure
causal result (0.0 error), so losslessness holds by construction rather than by testing; and the
mask half really does change (0.879 max delta), so the change is not a no-op.

## Integration

Everything needed already exists in vLLM 0.27.

- `vllm/_custom_ops.py: merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
  suffix_lse, ...)` — a CUDA kernel doing precisely this merge, used for cascade attention. Use it
  rather than the Python reference.
- `flash_attn_varlen_func(..., return_softmax_lse=True)` returns `(out, lse)`; the decode path in
  `v1/attention/backends/flash_attn.py` already calls it that way, and
  `FlashAttentionImpl.can_return_lse_for_decode = True`.
- Hook: `FlashAttentionImpl.forward(self, layer, query, key, value, kv_cache, attn_metadata,
  output, output_scale=None, output_block_scale=None)`. Patch it the way `_install_gdn_block_readout`
  patches the GDN core, gated on `_TRIDA_GDN_READOUT["mode"] == "selfspec"` so it is inert everywhere else.

### Steps
1. In the patched `forward`, call the original to fill `output`, requesting the LSE. If the stock
   path does not surface it, call `flash_attn_varlen_func` directly with `return_softmax_lse=True`
   (the metadata needed is already on `attn_metadata`).
2. Reshape the canvas rows of `query/key/value` to `[num_reqs, B, H, D]` — the canvas is a fixed
   `B`-token query block per request under spec-shape, so this is a view.
3. Compute the future-mask correction (see `bidir_merge.future_mask_attention`): scores masked to
   `is_mask[i] & is_mask[j] & j > i`, then softmax and LSE. Rows with no later mask get `lse = -inf`
   and the merge leaves them untouched — which is every verify row, hence the bit-identity above.
4. `merge_attn_states(output, output, causal_lse, correction_out, correction_lse)`.
5. Gate on `TRIDA_SS_BIDIR=1`, default off.

### Watch out
- **`num_clean` is `N`, not `N-1`.** The mask region starts at slot `N`. Getting this off by one
  silently makes verify rows bidirectional and destroys losslessness without crashing.
- **Both branches of `torch.where` evaluate.** Same trap as the carry-forward gather; clamp indices.
- **FULL cuda graphs freeze the branch at capture.** The patch must be shape-stable and take the same
  path every step, exactly like the existing GDN patch.
- **Mixed batches** (prompt chunk + canvases) already fall back to the stock kernel; the patch must
  no-op there rather than reshape a ragged query.

## Test
1. `python tools/experiments/bidir_merge.py` — CPU, already passing.
2. Serve with `TRIDA_SS_BIDIR=0` and `=1` on the same 30 prompts, greedy, single sequence.
   Both must produce **byte-identical token streams** (the mechanism changes drafts, never the
   verified output). If they differ, the mask boundary is wrong.
3. Compare tok/fwd and per-slot acceptance. Prediction: slot 3 acceptance rises from ≈0.39 toward
   ≈0.47, tok/fwd from 2.44 toward ≈2.9. Report the shortfall against that prediction.
