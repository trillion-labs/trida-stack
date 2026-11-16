# Model contract — what makes a checkpoint servable

This nano-inference serves **`trillionlabs/GravityOCR`**. The contract below is what `serve.sh` and the
patched SGLang expect from a checkpoint, so a local copy, a fine-tune or a new GravityOCR revision is
servable when it satisfies it (`CKPT=<dir or repo id> bash serve.sh`). A different model gets its own
`inference/<model>/`.

## Requirements

1. **Hugging Face layout.** A directory (or HF repo id) with `config.json`, weights as
   `model.safetensors` (bf16), the tokenizer (`tokenizer.json`, `tokenizer_config.json`), the chat
   template (`chat_template.jinja`) and the image processor (`processor_config.json`). SGLang loads it
   with `--model-path` and **without** `--trust-remote-code`.

2. **The GLM-OCR architecture, unchanged.** `architectures: ["GlmOcrForConditionalGeneration"]`,
   `model_type: glm_ocr`, and no `auto_map`. transformers 5.x and SGLang implement this class natively,
   so no modeling code is loaded from the checkpoint. Block-diffusion training changes the weights,
   not the architecture: only the text decoder learns to fill masked blocks, and the vision encoder
   runs as in GLM-OCR.

3. **`block_diffusion.json`** at the checkpoint root:

   | key | meaning |
   |---|---|
   | `bd_size` | the block size the model was trained with (GravityOCR: 32). Serve with `DRAFT` = `bd_size` + 1, the boundary token plus `bd_size` masks; the worker logs a warning when they differ |
   | `mask_id` | the id of the mask token used for drafting (GravityOCR: 59282, `<|mask|>`). Must be inside the text vocabulary; the worker refuses to start otherwise |
   | `ar_loss_weight` | `> 0` means the checkpoint was trained with the AR auxiliary loss and has a working causal head. Both modes decode with that head (`causal` alone, `self-spec` to verify), so a checkpoint with `0` is not servable here |

   The server reads only `bd_size` and `mask_id`. The other keys (`diff_loss_weight`, `c_diff`, `c_ar`,
   `mask_schedule`, ...) record how the model was trained and are ignored.

4. **The mask token in the vocabulary.** `tokenizer.json` lists `<|mask|>` as a special token with id
   `mask_id`, and `text_config.vocab_size` in `config.json` is larger than `mask_id`.

5. **GLM-OCR's prompt format.** The chat template renders an image followed by a text prompt. Requests
   use one of three task prompts, `Text Recognition:`, `Table Recognition:` (HTML) or
   `Formula Recognition:` (LaTeX), with **one layout region per request**.

## How self-spec finds the mask id

`_resolve_block_diffusion` in `sglang/python/sglang/srt/speculative/selfspec_diffusion_worker.py`
tries, in order:

1. the `SELFSPEC_MASK_ID` environment variable;
2. `mask_id` in `<model path>/block_diffusion.json`;
3. otherwise it stops with an error. There is deliberately no silent default: with a wrong mask id
   the verify pass still emits the AR head's tokens, but almost no draft is accepted, which is easy
   to miss.

SGLang's `--dllm-algorithm` (its separate diffusion-LLM path) is refused together with
`--speculative-algorithm SELFSPEC_DIFFUSION`, so it plays no part here.

`serve.sh` downloads a repo id to a local snapshot first, so `block_diffusion.json` is found there. It
exports `SELFSPEC_MASK_ID=59282` only when the served directory has no `block_diffusion.json`. The
server log names the source it used: `[selfspec-diffusion] K=33 mask_id=59282 (from <source>) bd_size=32`.

`serve.sh` also sets `SELFSPEC_POS_OFFSET=1` by default: every generated token (draft window, verify
window, KV cache) sits one RoPE position after the one causal decoding and the model's training give
it. This is the layout GravityOCR's published self-spec results were produced with; keep `1` to
reproduce them. `0` is the causal layout, under which verification computes what causal decoding
computes, up to floating point. In a development check on 200 crops it gave about the same draft
acceptance, but it has not been validated for GravityOCR on this build's optimized path beyond the
correctness tests in `README.md` ("What self-spec reproduces"). Block-diffusion training, GravityOCR's
included, uses the causal layout, so `0` is the setting that matches training.

## Minimal checklist

- [ ] `config.json` has `GlmOcrForConditionalGeneration` and no `auto_map`
- [ ] `block_diffusion.json` has `bd_size`, `mask_id`, and `ar_loss_weight > 0`
- [ ] `tokenizer.json` has `<|mask|>` at `mask_id`
- [ ] `MODE=causal` serves, and `ocr_request.py --synthetic 'Hello GravityOCR'` reads the text back
- [ ] `MODE=self-spec` serves, logs the mask id from your `block_diffusion.json`, and reads the text back

```bash
python -c "import json,sys; d=json.load(open(sys.argv[1]+'/block_diffusion.json')); print(d['bd_size'], d['mask_id'], d['ar_loss_weight'])" <CKPT>
```
