# Open-source license policy — trida-stack

Status: active · Owner: Trillion Labs research team · Last reviewed: 2026-09-30
Applies to: every third-party component reachable from this repository, including
components fetched at build time rather than committed.

This is the 허용/조건부/금지 list required for OSS intake evaluation. It adapts the
NIPA/오픈업 기본안 (`프로젝트평가` guideline §4-2), which explicitly permits adaptation
("기본안 — 자사 정책으로 조정"). Where we depart from the 기본안, the departure is
stated and justified rather than left implicit.

## 허용 — use freely, attribution only

| license | condition |
|---|---|
| Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, ISC | Attribution in `NOTICE`. No further review. |
| MPL-2.0, EPL-2.0 | Permitted; file-level modification notices required. |

## 조건부 — permitted only inside a declared boundary

| license | boundary |
|---|---|
| LGPL-2.1 / LGPL-3.0 | Dynamic linking only. Static linking requires review. |
| **PolyForm Noncommercial 1.0.0** | **See "The PolyForm-NC carve-out" below. This is a deliberate departure from the 기본안.** |

## 금지 — not adoptable

| license | reason |
|---|---|
| No license / unlicensed source | No grant of rights. |
| SSPL, BUSL, and other source-available non-OSI licenses | Service-use restrictions incompatible with our distribution. |
| GPL-2.0 / GPL-3.0 / AGPL-3.0 linked into the shipped tree | Copyleft propagation into an Apache-2.0 distribution. Legal review required before any exception. |
| Model weights whose license restricts outputs or downstream use, where unreviewed | Requires separate AI-compliance review; software license terms do not carry over. |

## The PolyForm-NC carve-out (departure from the 기본안)

The 기본안 places noncommercial-licensed code in **금지**, with the note
"상업 이용 불가·분쟁 여지". We accept that characterisation as accurate. We
nonetheless permit one specific noncommercial dependency, under a boundary, and we
state the cost openly rather than scoring around it.

**What is permitted.** The two-stream Gated-DeltaNet Triton kernels from
`yuchen-zhu-zyc/HybridDiffusion@6ca547a` (PolyForm Noncommercial 1.0.0), used by the
two-stream training path and both diffusion-serving backends.

**The boundary — all five conditions must hold:**

1. **Never vendored.** The kernels are fetched at build time from the pinned upstream
   commit and patched. No PolyForm-NC source is committed to this Apache-2.0 repository.
2. **Declared at every entry point.** `README.md`, `train/README.md`,
   `inference/README.md`, `inference/vllm/README.md`, `NOTICE` and `COMPLIANCE.md`
   each state that the fetched kernels are noncommercial and not covered by this
   repository's license.
3. **The restriction propagates and we say so.** Anyone using the two-stream training
   path or a diffusion-serving backend inherits the noncommercial restriction. This is
   not cured by our Apache-2.0 license on the surrounding code.
4. **The causal (AR) path stays clean.** `--mode causal` runs on stock vLLM with no
   PolyForm-NC dependency, and must remain so. Any change that makes the AR path
   depend on the kernels requires re-approval under this policy.
5. **A replacement strategy exists and is reviewed quarterly.** See below.

**The cost we are accepting.** Under this policy trida-stack is **not commercially
licensable through its two-stream paths**. That is a product constraint, not a
paperwork detail, and it is owned rather than hidden.

**대체전략 (replacement strategy).** Three routes out, in increasing cost:
upstream relicensing; a clean-room reimplementation of the block-end readout kernel
from the published FLARE method rather than from PolyForm-NC source; or a
commercially-clean build that ships only the causal path. No route is currently
funded. Reviewed each quarter against whether a commercial Trida is on the roadmap.

## Re-evaluation

Quarterly, or immediately on any of these triggers:

- the pinned upstream commit `6ca547a` moves in any of the three fetch recipes
  (`train/block_gated_delta_rule/fetch_kernels.sh`, `inference/sglang/`,
  `inference/vllm/vllm_native_diffusion/KERNELS.md`)
- any upstream changes its license
- a new dependency enters the shipped tree
- a commercial distribution of Trida is proposed
