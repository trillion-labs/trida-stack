# ocr

The Python engine behind Gravity. The app starts it as `python -m ocr.service`
and talks to it in JSON lines (`service/protocol.py`).

| Package | Owns | Ships in the app |
|---|---|---|
| `service/` | The process: request loop, cancellation, protocol | yes |
| `engine/` | One page end to end: layout, bounded region batches, recognition, result assembly, generation limits | yes |
| `recognition/` | GravityOCR on MLX: image preprocessing, prompt, block-diffusion decoding | yes |
| `layout/` | PP-DocLayoutV3 on Core ML: pre/post-processing, regions and reading order, pinned assets | yes |
| `resample.py` | torch-exact uint8 bicubic resize shared by recognition and layout | yes |
| `reference/` | The torch reference model, checkpoint tools and CLI, for development and parity tests | no |

Rules:

- Nothing outside `reference/` imports torch, torchvision or transformers.
  `tests/test_import_boundaries.py` and `apps/macos-ocr/scripts/build-runtime.sh`
  enforce it.
- The bundled packages are pinned in `apps/macos-ocr/runtime/requirements.txt`.
- `experiments/` may import from here; nothing here imports from `experiments/`.
