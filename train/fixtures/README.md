# Decision compatibility baseline

`decision_138239d.json` was generated on CPU from the original `train/jev.py`,
`HFBlockDiffusionHybrid.forward_jev`, and `HFBlockDiffusion.param_groups` at
`138239df5c5a305f74cc1dee7b1de206d8ef6438`, before this refactor. It freezes
head/embedding inputs, loss, head gradients, parameter-group ordering, and AdamW
state after one update. It is not produced by the refactored implementation.

The decoder is an identity stub so this checks decision routing/head/loss and
optimizer compatibility without GPU kernels or downloaded models. It does not
replace GPU/FSDP validation. Seed and torch version are recorded in the JSON.

To regenerate deliberately, run from the repository root with that commit available:

```bash
uv run --no-project --with torch --with numpy python train/fixtures/generate_decision_baseline.py
```

Tests consume the frozen values; they never run the generator or require Git history.
