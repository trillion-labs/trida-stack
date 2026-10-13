# IFEval scorer (vendored snapshot)

| | |
|---|---|
| 구분 | 외부유입 (vendored) |
| 라이선스 | Apache-2.0 |
| 획득 경로 | google-research/instruction_following_eval |
| 사용 위치 | `benchmark/` |
| 평가 기록일 | 2026-10-13 |
| 결정 시점 | vendored when the evaluation harness was first assembled |
| 결정 | **채택** |
| 총점 | **82 / 100** |
| 재평가 예정 | 2027-10-13 |

## Why we needed something here

Instruction-following scores are only comparable to published numbers if the scorer
is the same scorer. A reimplementation produces numbers nobody can check against the
literature.

## Alternatives considered

| 후보 | 왜 선택하지 않았나 |
|---|---|
| Depend on it from PyPI | the published packaging has not tracked the reference implementation reliably; a drifting scorer silently changes our reported numbers |
| Reimplement | defeats the purpose — see above |

## Score

| 기준 | 점수 | 근거 |
|---|---|---|
| A 기능 적합성 | 18 | it is the reference scorer; that is the whole requirement |
| B 라이선스 적합성 | 20 | Apache-2.0, allowed |
| C 유지보수 활성도 | 12 | quiet upstream — but **frozen is the point here**. A scorer that changes makes our historical numbers incomparable, so low commit activity scores as stability, not neglect. |
| D 보안 대응 | 14 | no stated policy; it is a pure scoring function over our own evaluation outputs, with no network or deserialisation surface |
| E 통합·이탈 비용 | 18 | self-contained; replacing it is one module and a re-run |
| **총점** | **82** | |

## Decision

Adopt as a **vendored snapshot** — the only third-party source committed into this
repository. Vendoring is deliberate: pinning a version does not protect a score from
a repackaging change, but a committed snapshot does.

## Obligations we take on

The Apache-2.0 licence text and the original copyright headers stay with the
vendored files, with the entry in [`NOTICE`](../../../NOTICE) and the record in
[`docs/supply-chain/asset-register.csv`](../../supply-chain/asset-register.csv).
Because it is vendored, no scanner will tell us about a finding in it — that is the
cost we accepted, and the register row is where a human is supposed to look.

## Re-evaluation trigger

Annually, or when the upstream reference implementation changes in a way that would
change scores — in which case the decision is whether to re-vendor and re-run, not
whether to drift.
