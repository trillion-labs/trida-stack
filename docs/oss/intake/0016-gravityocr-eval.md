# GravityOCR evaluation toolchain (`benchmark/gravityocr/`)

| | |
|---|---|
| 구분 | 외부유입 |
| 라이선스 | OmniDocBench Apache-2.0 · sacrebleu Apache-2.0 · datasets Apache-2.0 · **PyMuPDF AGPL-3.0** · **Levenshtein GPL-2.0-or-later** · TeX Live (LPPL and others) |
| 획득 경로 | `benchmark/gravityocr/setup.sh` installs everything into `$EVAL_HOME`, outside the repository |
| 사용 위치 | `benchmark/gravityocr/` only (scoring venv, Python 3.10) |
| 평가 기록일 | 2026-11-16 |
| 결정 시점 | PR #95 |
| 결정 | 조건부 채택 (maintainer exception for PyMuPDF and Levenshtein) |
| 총점 | 78/100 |
| 재평가 예정 | 2027-05-16, or when OmniDocBench drops Levenshtein, or when this tool starts shipping anything |

## Why we needed something here

To report GravityOCR's accuracy the way the field does, the scores have to come from OmniDocBench's own
metric code (text edit distance, table TEDS, formula CDM), not a reimplementation. That code, and the
pipeline around it, needs the components below.

| Component | Version | Licence | Used for |
|---|---|---|---|
| [OmniDocBench](https://github.com/opendatalab/OmniDocBench) | `0604d98` | Apache-2.0 | the metric code `score.py` imports |
| [Levenshtein](https://github.com/rapidfuzz/Levenshtein) | `0.25.1`, pinned by OmniDocBench's `pyproject.toml` | **GPL-2.0-or-later** | edit distance, in OmniDocBench and in `score.py` |
| [PyMuPDF](https://github.com/pymupdf/PyMuPDF) | `1.28.2` | **AGPL-3.0** (or commercial) | renders the CDM formula PDFs to images (the `magick` shim in `run.sh`) |
| [sacrebleu](https://github.com/mjpost/sacrebleu) | `2.6.0` | Apache-2.0 | BLEU |
| [datasets](https://github.com/huggingface/datasets) | `5.2.0` | Apache-2.0 | exporting the evaluation set |
| TeX Live, via upstream `install-tl` | 2022.06 or newer | LPPL and other free licences | `pdflatex` for CDM, run as an external program |

## Alternatives considered

| 후보 | 왜 선택하지 않았나 |
|---|---|
| `rapidfuzz` instead of `Levenshtein` in `score.py` | doesn't remove the GPL dependency: OmniDocBench itself pins `Levenshtein==0.25.1` |
| `pypdfium2` instead of PyMuPDF | possible later; the shim was validated with PyMuPDF on H100 and L40S |
| Reimplementing the metrics | scores would drift from the published OmniDocBench numbers we compare against |

## Score

| 기준 | 점수 | 근거 |
|---|---|---|
| A 기능 적합성 | 19/20 | the reference metric code; guks reproduced the L40S reference on H100 (92.62 vs 92.57) |
| B 라이선스 적합성 | 8/20 | two dependencies are on the prohibited list; held to a local-tool exception below |
| C 유지보수 활성도 | 17/20 | OmniDocBench and the libraries are actively maintained |
| D 보안 대응 | 16/20 | pinned versions; PyMuPDF parses only PDFs the tool itself produces |
| E 통합·이탈 비용 | 18/20 | isolated in its own venv under `$EVAL_HOME`; removable without touching serving or training |
| **총점** | **78/100** | |

## Decision

**Licence exception approved by:** Luke Kim (@usik-luke-trl), trida-stack maintainer, 2026-11-16 (PR #95).
It is an exception to the GPL / AGPL row of [`LICENSE_POLICY.md`](../LICENSE_POLICY.md), limited to this
evaluation tool:

- PyMuPDF and Levenshtein are installed by `setup.sh` into a separate venv under `$EVAL_HOME`, outside the
  repository. They are never committed, bundled or redistributed.
- No serving, training or library code imports them; only `benchmark/gravityocr/` does, as a local tool run
  by the person evaluating.
- Nothing from this tool is shipped or offered as a network service, so AGPL's network clause doesn't come
  into play.
- If the tool ever needs to ship, or run as a service for others, this exception ends and the dependencies
  must be replaced (`pypdfium2` for PyMuPDF; Levenshtein needs OmniDocBench to drop it first).

## Obligations we take on

- Keep the versions pinned in `setup.sh`, and this record and both registers in step with them.
- Re-check on the date above.
