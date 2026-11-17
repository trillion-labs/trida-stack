# GravityOCR Mac engine (`inference/gravityocr/mac/`)

| | |
|---|---|
| 구분 | 직접개발 (`mac/ocr`, Trillion Labs code copied from GravityOCR-macOS) · 외부유입 (the packages in `mac/requirements-lock.txt`) |
| 라이선스 | `mac/ocr`: Apache-2.0, contributed by Trillion Labs (GravityOCR-macOS has no licence file) · packages: MIT, BSD, Apache-2.0, MIT-CMU, PSF-2.0, **MPL-2.0 AND MIT** (tqdm), **WTFPL** (pyaml) |
| 획득 경로 | `mac/ocr`: copied from the Gravity app 0.3.4 (`Contents/Resources/project/ocr`), byte for byte (`mac/VENDORED.sha256`) · packages: `mac/setup.sh`, from PyPI only, wheels only, hashes checked, `--no-deps` |
| 사용 위치 | `inference/gravityocr/serve_mac.py` (Apple Silicon Macs), and `benchmark/gravityocr/run.sh` with `BACKEND=mac`; a separate Python 3.12 venv |
| 평가 기록일 | 2026-11-16 |
| 결정 시점 | PR #109 |
| 결정 | 조건부 채택 — pending a maintainer's sign-off on contributing `mac/ocr` under Apache-2.0 (see Decision) |
| 총점 | 85/100 |
| 재평가 예정 | 2027-05-16, or when `mac/ocr` is updated to a newer Gravity |

## Why we needed something here

`serve_mac.py` serves GravityOCR on an Apple Silicon Mac, for the getting-started notebook and for evaluation
without an NVIDIA GPU. Before this, it imported the engine from an installed Gravity app. Copying the engine in
makes trida-stack self-contained, with nothing installed outside the repository but PyPI packages.

| Component | Version | Licence | Used for |
|---|---|---|---|
| `mac/ocr` (Gravity's MLX engine) | Gravity 0.3.4 | Apache-2.0 (contributed; see Decision) | the OCR pipeline `serve_mac.py` runs |
| [mlx](https://github.com/ml-explore/mlx), mlx-metal | 0.32.2 | MIT | the GPU runtime |
| [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) | 0.7.0 (`--no-deps`) | MIT | the GLM-OCR model code the engine imports |
| [coremltools](https://github.com/apple/coremltools) | 9.0 | BSD-3-Clause | the layout model (whole pages only; `serve_mac.py` sends crops) |
| [protobuf](https://github.com/protocolbuffers/protobuf) | 7.36.0 | BSD-3-Clause | coremltools |
| [tokenizers](https://github.com/huggingface/tokenizers) | 0.23.1 | Apache-2.0 | the tokenizer |
| numpy · sympy · mpmath | 2.5.2 · 1.14.0 · 1.3.0 | BSD (numpy: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0) | arrays; coremltools' dependencies |
| [Pillow](https://github.com/python-pillow/Pillow) | **13.0.0** (Gravity bundles 12.3.0) | MIT-CMU | images |
| attrs · cattrs · PyYAML · packaging · typing-extensions | see the lock | MIT · MIT · MIT · Apache-2.0 OR BSD-2-Clause · PSF-2.0 | coremltools' and the engine's dependencies |
| [tqdm](https://github.com/tqdm/tqdm) | 4.70.0 | **MPL-2.0 AND MIT** | coremltools' progress bars |
| [pyaml](https://github.com/mk-fg/pretty-yaml) | 26.7.0 | **WTFPL** | coremltools |

## Alternatives considered

| 후보 | 왜 선택하지 않았나 |
|---|---|
| Keep importing the engine from an installed Gravity app | the reviewer asked for no runtime dependency on the app or on GravityOCR-macOS |
| Install mlx-vlm with its dependencies | pulls in transformers and torch, which the engine does not import |
| Keep Pillow 12.3.0, as the app bundles it | 42 advisories (pip-audit, 2026-11-16), all fixed in 13.0.0; 13.0.0 gives the same 402 predictions on `trillionlabs/gravityocr-dev` |

## Score

| 기준 | 점수 | 근거 |
|---|---|---|
| A 기능 적합성 | 19/20 | the engine Gravity ships; predictions identical to the app's on 402 crops |
| B 라이선스 적합성 | 14/20 | our own code, but its source repository states no licence; tqdm is file-level copyleft (unmodified); pyaml's WTFPL is not on the policy's lists |
| C 유지보수 활성도 | 17/20 | maintained by Trillion Labs (engine) and active upstreams (MLX, mlx-vlm) |
| D 보안 대응 | 17/20 | hashed lock from PyPI only, wheels only; `pip-audit` clean after the Pillow bump; scanned in CI |
| E 통합·이탈 비용 | 18/20 | a separate venv, used only on a Mac; removing it leaves the SGLang path untouched |
| **총점** | **85/100** | |

## Decision

- **`mac/ocr`:** Trillion Labs code. GravityOCR-macOS is public but has no licence file, which
  [`LICENSE_POLICY.md`](../LICENSE_POLICY.md) treats as all-rights-reserved until a licence is stated in writing.
  PR #109 states it: Trillion Labs contributes `mac/ocr` to this repository under Apache-2.0, recorded in
  `inference/gravityocr/README.md`, `inference/gravityocr/NOTICE`, `NOTICE` and `COMPLIANCE.md` §1.
  **Needs a maintainer's sign-off on PR #109** (the owner of GravityOCR-macOS or a trida-stack maintainer);
  record the name and date here when given.
- **tqdm (MPL-2.0 AND MIT):** conditional licence, met: installed unmodified as a dependency, never vendored.
- **pyaml (WTFPL):** a permissive licence not listed in the policy; it imposes no obligations. Installed
  unmodified as a dependency, never vendored. Recorded here for the same sign-off.

## Obligations we take on

- Keep `mac/ocr` byte-identical to a Gravity release (`mac/VENDORED.sha256`, checked by a test) and this
  record, both registers and `mac/VENDORED.md` in step with it.
- Keep `mac/requirements-lock.txt` free of open advisories (CI scans it) and re-check on the date above.
