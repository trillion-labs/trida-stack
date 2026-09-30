# 외부 OSS 도입 평가서 — HybridDiffusion

> **Record date 2026-09-30.** This form documents a decision taken earlier, during the
> two-stream port and the pre-public-release compliance work (PR #40, September 2026).
> The evaluation and its reasoning are contemporaneous; **this written record is not** —
> it was produced on the record date above. Nothing here is backdated.

## 기본정보

- 신청 프로젝트: trida-stack (`train/`, `inference/sglang/`, `inference/vllm/`)
- 후보명 / 버전 / commit: HybridDiffusion / — / `6ca547a` ("Initial public release")
- 공식 저장소·패키지 URL: https://github.com/yuchen-zhu-zyc/HybridDiffusion
- 신청자 / 평가자 / 평가일: Trillion Labs research team / same / decision Sept 2026, recorded 2026-09-30
- 사용 기능과 결합 방식: two-stream Gated-DeltaNet + ShortConv Triton kernels and the
  block-end readout. **Fetched at build time and patched; never committed to this repo.**
  Used by `train/hf_block_diffusion_hybrid.py` (`forward_flare`), the SGLang backend, and
  the vLLM plugin's `block_causal_readout.py`.
- 배포 시나리오: 소스 (recipe only — we distribute instructions, not the code)
- 예상 사용기간: indefinite, until a 대체전략 route is funded

## 치명 조건

- [x] 공식 출처와 버전을 확인했다. — pinned `6ca547a` in three independent recipes
- [x] 라이선스가 식별되고 정책상 검토 가능하다. — PolyForm Noncommercial 1.0.0
- [ ] **상업 이용·배포를 금지하는 조건이 없다. — FAIL**
- [x] 해결 불가능한 치명적 보안 위험이 없다.
- [x] 필수 플랫폼과 기능을 충족한다. — no substitute implementation exists
- [x] 출처 불명 코드·바이너리·데이터가 없다.

**한 조건이 충족되지 않았다.** Per the template, 점수평가는 중단된다. The score below is
recorded for completeness only and is **not** an adoption score.

## 점수 (참고용 — 치명 조건 미충족으로 평가 중단)

| 영역 | 배점 | 점수 | 사실·근거 URL | 판단 |
|---|---:|---:|---|---|
| 라이선스·권리 | 20 | **0** | PolyForm-NC 1.0.0 — 정책 §4-2 기준 목록 외 | 상업 이용 금지 |
| 보안·공급망 | 20 | 14 | no known advisories; single-maintainer research repo | pinned commit mitigates drift |
| 기능·성능 | 15 | 15 | the only implementation of the FLARE two-stream readout | no alternative |
| 유지보수성 | 15 | 6 | research repo, one public release, low commit activity | upstream may go dormant |
| 아키텍처·운영 | 10 | 8 | kernels are self-contained (torch/triton/fla) | patch surface is small |
| 커뮤니티·거버넌스 | 10 | 2 | no governance, no release cadence | single author |
| 지속가능성·대체성 | 10 | 2 | no drop-in replacement; reimplementation is months of work | high lock-in |
| 합계 | 100 | **47** | | **게이트 불통과** |

Under `프로젝트평가` §4-1 the decision rule is 70점 이상 **+ 라이선스 15점 이상**.
This fails both.

## 결정

- **결정: 조건부 승인 (research / noncommercial only)** — adopted under the explicit
  carve-out in [`../LICENSE_POLICY.md`](../LICENSE_POLICY.md), **not** by passing the gate.
  We adopt a dependency our own policy scores as non-adoptable, because no alternative
  exists and the research work requires it. Recording that honestly is the point of this form.
- 승인 버전과 사용범위: `6ca547a` only. Two-stream training path and both diffusion-serving
  backends. **The causal (AR) path must remain free of this dependency.**
- 조건·완화조치 / 담당 / 기한:
  1. never vendored — fetch recipe only · research team · ongoing (in force)
  2. noncommercial restriction declared in 4 READMEs + NOTICE + COMPLIANCE · done 2026-09-30 (PR #1)
  3. AR path kept clean — re-approval required if this changes · ongoing
- **대체전략:** (a) upstream relicensing — not approached; (b) clean-room reimplementation
  of the block-end readout from the published FLARE method rather than from PolyForm-NC
  source — estimated months, unfunded; (c) commercially-clean build shipping only the
  causal path — loses the product's differentiator. **No route currently funded.**
- 재평가일과 트리거: 2026-12-31, or on pinned-commit drift / upstream license change /
  any proposal to distribute Trida commercially
- 승인자: Trillion Labs research team

## 미해결 사항 (escalation)

**trida-stack is not commercially licensable through its two-stream paths.** The
restriction reaches every downstream user of two-stream training or diffusion serving.
This is a product decision that has not been formally taken by anyone outside the
research team. It is recorded here so that it is taken deliberately rather than by default.
