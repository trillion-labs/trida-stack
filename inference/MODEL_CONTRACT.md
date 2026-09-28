# Model contract — what makes a checkpoint servable

nano-inference serves **diffusion LLMs** through the diffusion-serving SGLang
backend. "Bring your own model" means: if your checkpoint satisfies the contract
below, `serve.py --model <your-checkpoint>` just works. The shipped reference
checkpoint (a Qwen-based diffusion model) is the canonical example.

## Requirements

1. **Hugging Face layout** — a directory (or HF repo id) with `config.json`,
   weights (`model.safetensors`, bf16), and a tokenizer. Loaded via
   `--model-path` with `--trust-remote-code`.

2. **A diffusion-capable architecture** the backend recognizes. The reference
   model uses a two-stream (clean + noisy) block-diffusion architecture whose
   modeling code ships *inside the checkpoint* (custom `architectures` +
   `auto_map` in `config.json`, resolved by `trust_remote_code`). Your model must
   expose the same block-diffusion decode interface the backend's
   `--dllm-algorithm` implementations call.

3. **A mask token.** Block-diffusion decoding fills masked positions, so the
   checkpoint's vocabulary must contain a mask token, and its id must match the
   `mask_id` in the decode config (`configs/*.yaml`). The reference model uses
   `mask_id: 248077`.

4. **A chat template** — with optional `enable_thinking` support (the eval and
   chat paths pass `chat_template_kwargs={"enable_thinking": ...}`).

## Decode config

Each `--mode` maps to a YAML in `configs/` describing how to decode *your* model:

| key | meaning |
|---|---|
| `variant` | backend decode variant identifier |
| `block_size` | runtime block width (e.g. 3 → logical block B=4 in the shift variant) |
| `gen_block_size` | committed tokens per block step |
| `threshold` | per-step confidence commit gate (higher = stricter) |
| `mask_id` | must equal your checkpoint's mask token id |

Match these to how your model was trained. A model trained with a different block
scheme needs its own config — copy one in `configs/` and pass `--config`.

## Minimal checklist

- [ ] loads with `AutoModel.from_pretrained(..., trust_remote_code=True)`
- [ ] has a mask token; `mask_id` set correctly in the decode config
- [ ] chat template renders (with/without thinking)
- [ ] `serve.py --model <you> --mode causal --dry-run` prints the expected command
- [ ] serves and answers a prompt via `chat.py`

If all five pass, the eval harness (`eval.py`, `eval_ifeval.py`) will score it with
no changes.
