# SGLang

| | |
|---|---|
| 구분 | 외부유입 |
| 라이선스 | Apache-2.0 |
| 획득 경로 | https://github.com/sgl-project/sglang |
| 사용 위치 | `inference/zoro-1/sglang/` |
| 평가 기록일 | 2026-10-13 |
| 결정 시점 | adopted as the first diffusion serving backend, ahead of the vLLM plugin |
| 결정 | **채택** |
| 총점 | **85 / 100** |
| 재평가 예정 | 2027-10-13 |

## Why we needed something here

The diffusion serving path needed a second independent implementation. One engine
cannot tell you whether a decoding result is a property of the model or a property
of the engine — and in practice this repository has resolved several
"model" findings that turned out to be engine-specific by comparing the two.

## Alternatives considered

| 후보 | 왜 선택하지 않았나 |
|---|---|
| vLLM only | a single backend gives no cross-check; see above |
| HF `generate` | not a serving engine; no batching or cache management at the scale we test |

## Score

| 기준 | 점수 | 근거 |
|---|---|---|
| A 기능 적합성 | 18 | covers the diffusion path, but we carry a patch rather than a plugin |
| B 라이선스 적합성 | 20 | Apache-2.0, allowed |
| C 유지보수 활성도 | 17 | active project, fast-moving; the pace is itself a cost to us |
| D 보안 대응 | 15 | advisories exist; the policy is less formalised than vLLM's |
| E 통합·이탈 비용 | 15 | the diffusion backend is applied as a patch against a pinned upstream, so an upstream move costs us work |
| **총점** | **85** | |

## Decision

Adopt as the second serving backend. The patch-against-pinned-upstream arrangement
is the reason `tools/verify_fetch.sh` runs weekly: if upstream moves or rewrites
history, we find out from CI rather than from a bug report.

## Obligations we take on

Apache-2.0 attribution in [`NOTICE`](../../../NOTICE), and state changes for the
patched paths — recorded in [`COMPLIANCE.md`](../../../COMPLIANCE.md).

## Re-evaluation trigger

Annually, or when the weekly fetch verification fails for a structural reason
(subpath gone, pin unreachable) rather than a transient one.
