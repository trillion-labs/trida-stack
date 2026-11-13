# transformers >=4.57,<5

| | |
|---|---|
| 구분 | 외부유입 |
| 라이선스 | Apache-2.0 |
| 획득 경로 | PyPI |
| 사용 위치 | `train/`, `inference/`, `benchmark/`; 5.3.0 in the separate `inference/gravityocr/` server environment (see Scope) |
| 평가 기록일 | 2026-10-13 |
| 결정 시점 | the `<5` ceiling was set when 5.x broke checkpoint loading |
| 결정 | **조건부 채택** |
| 총점 | **84 / 100** |
| 재평가 예정 | 2027-04-13, or when our checkpoints load under 5.x — whichever is first |

## Why we needed something here

Model definitions, tokenizers and checkpoint loading. Our checkpoints are published
in the Hugging Face format, so this is also our interoperability surface.

## Alternatives considered

| 후보 | 왜 선택하지 않았나 |
|---|---|
| Our own loader | we would then publish checkpoints nobody else can load |
| Pin to 5.x and port the checkpoints | this is the eventual path, but it is a migration with its own schedule, not a version bump |

## Score

| 기준 | 점수 | 근거 |
|---|---|---|
| A 기능 적합성 | 18 | complete fit for loading and tokenization; the remote-code path is the part we have to reason about |
| B 라이선스 적합성 | 20 | Apache-2.0, allowed |
| C 유지보수 활성도 | 19 | very active, frequent releases — which is also why the ceiling is load-bearing |
| D 보안 대응 | 17 | advisories published and acted on. Several of the open entries in [`exceptions.md`](../../supply-chain/exceptions.md) are in this component, with the mitigation recorded as "load only from trusted repositories". |
| E 통합·이탈 비용 | 10 | the checkpoint format is the interoperability contract; leaving means changing what we publish |
| **총점** | **84** | |

## Decision

**Conditional**, despite a score in the adopt band. The condition is the `<5`
ceiling: our checkpoints' remote code does not load under 5.x. This is the override
in [`SCORECARD.md`](../SCORECARD.md) — the score measures the component, the
condition describes our situation.

The ceiling is not arbitrary and it is not free. Staying under 5 means some upstream
security fixes land only in 5.x releases, which is why the relevant findings are
recorded in [`exceptions.md`](../../supply-chain/exceptions.md) with an approver and
an expiry that CI enforces, rather than being silently carried.

## Scope (amended 2026-10-30)

The `<5` ceiling applies to the environments resolved from `requirements.txt` and
`inference/zoro-1/requirements.txt`, where our checkpoints' remote code has to load. It
stands unchanged.

`inference/gravityocr/` runs in a separate Python 3.12 environment, built from
`inference/gravityocr/sglang/requirements-lock.txt` and never installed together with
`requirements.txt`. There transformers is **5.3.0**, because SGLang v0.5.10 pins it
exactly. That environment needs 5.x anyway: GLM-OCR's architecture is built into
transformers 5.x, and the checkpoint carries no remote code. The pin is recorded and
owned in [`0010-sglang-gravityocr.md`](0010-sglang-gravityocr.md).

One exception now spans both environments. PYSEC-2026-4174 (custom_generate fetched
before the trust check) affects 4.49.0 through 5.8.1, so its entry in
[`exceptions.md`](../../supply-chain/exceptions.md) is written for root and zoro-1 as well
as for `inference/gravityocr/`. Until 2026-10-30, root's 4174 finding was matched only by
PYSEC-2025-217's entry, which listed 4174's aliases by mistake. PYSEC-2026-3929 and
PYSEC-2026-2290 now record the 5.3.0 environment too.

## Obligations we take on

Apache-2.0 attribution in [`NOTICE`](../../../NOTICE). Plus the real one: keeping
the exception entries current, because the ceiling is what makes them necessary.

## Re-evaluation trigger

Whichever comes first — the review date, our checkpoints loading under 5.x, or an
exception for this component expiring. The third is the one CI will tell us about.
