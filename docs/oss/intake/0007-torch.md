# torch 2.11.0

| | |
|---|---|
| 구분 | 외부유입 |
| 라이선스 | BSD-3-Clause |
| 획득 경로 | PyPI / pytorch.org |
| 사용 위치 | `train/`, `inference/`, `benchmark/`; 2.9.1 in the separate `inference/gravityocr/` server environment (see Scope) |
| 평가 기록일 | 2026-10-13 |
| 결정 시점 | the exact pin predates this record; it is the only `==` pin in `requirements.txt` |
| 결정 | **채택** |
| 총점 | **85 / 100** |
| 재평가 예정 | 2027-10-13, or at any CUDA/driver change on the training boxes |

## Why we needed something here

Everything is written against it. There is no version of this stack where a tensor
library is optional.

## Alternatives considered

None, honestly. JAX and MLX exist and MLX is used on-device, but the kernels, the
serving engines and the checkpoints are all PyTorch. Recording "no alternative was
considered" is more useful than inventing a comparison we did not make.

## Score

| 기준 | 점수 | 근거 |
|---|---|---|
| A 기능 적합성 | 20 | complete fit |
| B 라이선스 적합성 | 20 | BSD-3-Clause, allowed, attribution only |
| C 유지보수 활성도 | 19 | large maintainer group, predictable release cadence |
| D 보안 대응 | 18 | published advisories, a clear reporting path, fixes land |
| E 통합·이탈 비용 | 8 | **effectively irreplaceable.** Leaving PyTorch is rewriting the project. |
| **총점** | **85** | |

The 8 on E is the honest number and it did not change the decision. Scoring it
higher to make the total look better would hide the one fact a reader should take
from this record.

## Decision

Adopt, pinned exactly. `torch==2.11.0` is the only `==` pin in `requirements.txt`
because the CUDA build must match the driver and the compiled kernels on the
training boxes; a floating torch silently breaks a cluster.

## Scope (amended 2026-10-30)

This decision covers the environments resolved from `requirements.txt` and
`inference/zoro-1/requirements.txt`. The `==2.11.0` pin and its reasoning stand there
unchanged.

`inference/gravityocr/` runs in a separate Python 3.12 environment, built from
`inference/gravityocr/sglang/requirements-lock.txt` and never installed together with
`requirements.txt`. There torch is **2.9.1**, because SGLang v0.5.10 pins it exactly and
the kernels in that environment (sglang-kernel, FlashInfer, the CUDA builds of
torchvision and torchaudio) are built against it. That pin is recorded and owned in
[`0010-sglang-gravityocr.md`](0010-sglang-gravityocr.md). Its advisories are in
[`exceptions.md`](../../supply-chain/exceptions.md) (PYSEC-2026-2286, PYSEC-2025-195,
PYSEC-2026-139 and the existing PYSEC-2025-194). It moves only with an SGLang rebase.
That rebase is not a reason to change the 2.11.0 pin here, and the 2.11.0 pin is not a
reason to change that one.

## Obligations we take on

BSD-3-Clause attribution in [`NOTICE`](../../../NOTICE), including the
no-endorsement clause.

## Re-evaluation trigger

Annually, and on any driver or CUDA change on the training nodes — the pin exists
to be re-examined at exactly that moment, not to be permanent.
