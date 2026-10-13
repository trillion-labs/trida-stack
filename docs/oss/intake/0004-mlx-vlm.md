# mlx-vlm (adapted)

| | |
|---|---|
| 구분 | 수정포함 |
| 라이선스 | MIT |
| 획득 경로 | https://github.com/Blaizzy/mlx-vlm |
| 사용 위치 | vision encoder on the MLX backend (PR #4) |
| 평가 기록일 | 2026-10-13 |
| 결정 시점 | PR #4, open |
| 결정 | **조건부 채택** |
| 총점 | **78 / 100** |
| 재평가 예정 | when PR #4 lands, and otherwise 2027-04-13 |

## Why we needed something here

The MLX backend needs a vision encoder whose parameter names line up with our
checkpoints. Writing one is possible but duplicates work that exists.

## Alternatives considered

| 후보 | 왜 선택하지 않았나 |
|---|---|
| Write the encoder from scratch | more code to maintain for no research value |
| Convert the HF encoder at load time | parameter-name mapping at runtime is exactly the fragile thing we want to avoid |

## Score

| 기준 | 점수 | 근거 |
|---|---|---|
| A 기능 적합성 | 16 | the encoder fits, but we adapt it for parameter-name compatibility rather than using it as published |
| B 라이선스 적합성 | 20 | MIT, allowed |
| C 유지보수 활성도 | 14 | active, but the bus factor is low — effectively a single-maintainer project |
| D 보안 대응 | 13 | no stated policy; same local-inference surface as MLX |
| E 통합·이탈 비용 | 15 | adapted code lives in our tree, so leaving means deleting a module |
| **총점** | **78** | |

## Decision

**Conditional.** 78 falls in the conditional band, and the condition is concrete:
the adapted code must arrive with its tests, and the attribution for the adapted
sources must be correct before it merges. Both are open review items on PR #4 at the
time of writing.

Adapting MIT code into our tree means this is a **수정포함** component, not a plain
dependency: the original copyright notice stays with the adapted files.

## Obligations we take on

MIT attribution in [`NOTICE`](../../../NOTICE), the original copyright header
retained in the adapted sources, and the adaptation described in
[`COMPLIANCE.md`](../../../COMPLIANCE.md).

## Re-evaluation trigger

PR #4 landing. If PR #4 is closed without merging, this record is closed with it.
