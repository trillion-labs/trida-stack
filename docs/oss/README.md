# Open-source governance records

Evidence for the OSS maturity domains, kept in the repository so every claim has a URL.
Structure follows the NIPA/오픈업 오픈소스 성숙도 guidelines (TTAK.KO-11.0246_R1).

| file | domain | what it is |
|---|---|---|
| [`LICENSE_POLICY.md`](LICENSE_POLICY.md) | 프로젝트평가 | 허용/조건부/금지 license list, and the PolyForm-NC carve-out |
| [`INTAKE_REGISTER.csv`](INTAKE_REGISTER.csv) | 프로젝트평가 | one row per adopted third-party component |
| [`intake/`](intake/) | 프로젝트평가 | the completed 도입 평가서 behind each register row |
| [`KPI_DEFINITION.csv`](KPI_DEFINITION.csv) | 성과관리 | 6 KPIs + 2 guardrails: formula, scope, filters, limitations |
| [`MONTHLY_METRICS.csv`](MONTHLY_METRICS.csv) | 성과관리 | monthly measurements with data-quality status |
| [`EVIDENCE_REGISTER.csv`](EVIDENCE_REGISTER.csv) | 성과관리 | claim → evidence URL |
| [`MATURITY_ASSESSMENT.csv`](MATURITY_ASSESSMENT.csv) | 성과관리 | 8-dimension self-assessment with next-level gaps |
| [`QUARTERLY_REVIEW.md`](QUARTERLY_REVIEW.md) | 성과관리 | quarterly review record |

## How to read these honestly

**Missing is not zero.** Where no measurement was possible the value is `결측`, not `0`.
Three of eight KPIs are currently 결측 because the thing they measure does not yet exist
(no external contributors, one release). Recording that as `0` would be false.

**Scope is declared per KPI.** The public repository was created 2026-09-28 with a
squashed history, so repository-derived contribution metrics do not reflect the project's
real history. KPI-006 is marked `degraded_scope_mismatch` for this reason and should not
be interpreted.

**These metrics are not for evaluating individuals.** The guideline is explicit:
"개인을 감시하거나 프로젝트를 줄 세우는 용도로 사용하지 않는다." With a single maintainer,
every project metric is also an individual metric — which is a reason for care, not a
reason to collect more.

**Current maturity is not Lv3 everywhere, and the files say so.** `MATURITY_ASSESSMENT.csv`
records four dimensions at Lv1. 성과관리 requires two measurement periods and we have one.
