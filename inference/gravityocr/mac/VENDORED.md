# mac/ocr: the Gravity app's MLX engine

`ocr/` is the Python engine of the Gravity app for macOS
([GravityOCR-macOS](https://github.com/trillion-labs/GravityOCR-macOS)), by Trillion Labs. `serve_mac.py`
runs GravityOCR on an Apple Silicon Mac with it, so the Gravity app does not have to be installed.

| | |
|---|---|
| Source | `Gravity.app/Contents/Resources/project/ocr`, Gravity 0.3.4 |
| Copied | 2026-11-16, unchanged |
| Left out | `ocr/service/` (the app's JSON-lines process protocol; `serve_mac.py` is the server here) |
| Checksums | `VENDORED.sha256` (SHA-256 of every file; `test_gravityocr_mac.py` checks it) |
| Packages | `requirements-lock.txt`: the versions bundled in Gravity 0.3.4, installed by `setup.sh`; except Pillow, 13.0.0 instead of 12.3.0 for its security fixes (same predictions on gravityocr-dev) |
| Licence | Apache-2.0, contributed by Trillion Labs (GravityOCR-macOS has no licence file; see `../README.md` → Provenance and license) |

The files are kept byte for byte as in the app, so an update is a copy, not a merge. Do not edit them
here; pre-commit skips `mac/ocr/`.

`ocr/README.md` and some comments refer to paths of the GravityOCR-macOS repository that are not copied
here: `tests/test_import_boundaries.py`, `apps/macos-ocr/scripts/build-runtime.sh`,
`apps/macos-ocr/runtime/requirements.txt` and `reference/`. Here, `test_gravityocr_mac.py` checks the
import boundary instead (no torch, torchvision or transformers under `ocr/`).

What `serve_mac.py` uses: `ocr.engine.pipeline.LocalOcrEngine` (with `structured=False`: one crop per
request, no layout step) and, for `--mode causal`, `ocr.recognition.model.decode_ar_mlx_from_prefix`.
`ocr/layout/` comes along because `pipeline.py` imports it; it is used only for whole pages.

## Updating to a newer Gravity

```bash
cd inference/gravityocr/mac
rm -rf ocr && rsync -a --exclude service --exclude __pycache__ \
  /Applications/Gravity.app/Contents/Resources/project/ocr/ ocr/
(find ocr -type f | LC_ALL=C sort | xargs shasum -a 256) > VENDORED.sha256
ls /Applications/Gravity.app/Contents/Resources/runtime/lib/python3.12/site-packages | grep dist-info
```

Then:
1. Set `requirements.in` to the package versions that listing shows.
2. Regenerate the lock:
   ```bash
   MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile requirements.in --no-deps --generate-hashes \
     --python-version 3.12 --python-platform aarch64-apple-darwin --index-url https://pypi.org/simple \
     --no-header -o requirements-lock.txt
   ```
3. Run `pip-audit -r requirements-lock.txt --no-deps --disable-pip` (CI runs it too). Raise any package with
   an advisory, as was done for Pillow, and check that the predictions don't change.
4. Update the version and date in this file, and `docs/oss/intake/0021-gravityocr-mac-engine.md`.
5. Re-run `setup.sh`, `test_gravityocr_mac.py` and a Mac evaluation (`benchmark/gravityocr/run.sh`).
