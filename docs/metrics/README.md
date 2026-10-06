# Project metrics

Three metrics and one context signal, collected monthly by
[`tools/collect_metrics.py`](../../tools/collect_metrics.py) into
`metrics/snapshots/<YYYY-MM>.json`.

## We are in an observation period, and there are no targets

The repository has been public since **2026-09-28**. The 성과관리 guideline puts a new
project in a **4–8 week observation period** whose job is to confirm data quality rather
than hit numbers, and wants a **90-day baseline** before any target is committed:

> 새 프로젝트는 첫 4~8주를 관찰기간으로 두고 목표 대신 데이터 품질을 확인한다.

So no metric here has a target. The SLA threshold below is **provisional** — it exists so
the measurement has a shape, not as a commitment. Setting real targets is a decision for
after the baseline, and the guideline is explicit that targets come as three values
(목표 / 경고 / 중단 조건), not one.

## What is measured

| id | metric | definition |
|---|---|---|
| **M-001** | 첫 사람 응답시간 | `first_human_response_at − created_at`, median and p90, in business hours |
| **M-002** | 미응답률 | items with no human response ÷ items × 100 |
| **M-003** | SLA 달성률 | responses within the threshold ÷ measured responses × 100 |
| **C-001** | 관심 신호 | stars / forks / watchers |
| **C-002** | open items | `open_issues_count` — **includes PRs** |

**M-001 and M-002 belong together.** An item that is never answered drops out of a
response-time sample entirely, so a good median can coexist with widespread neglect.
Reading M-001 without M-002 is the mistake the pairing exists to prevent.

**Median and p90, never the mean.** An average hides a single item abandoned for months.

**C-001 is interest, not adoption.** Stars and forks are attention signals. The guideline
is blunt that they are not evidence of use, and real adoption needs overlapping evidence —
production use at distinct organisations, dependent packages, repeat users. We have none
of that yet and do not claim otherwise.

## Scope: external contributors only

The metrics measure the **external contributor experience**. Items opened by repository
collaborators are counted separately and excluded.

This matters more than it sounds. Every item in the repository today is a first-party PR
opened and merged by a maintainer. Counting those produced a **100% no-response rate** —
arithmetically true, and meaningless. The current honest reading is `no_data`: there have
been no external contributions yet.

## Missing is not zero

Values with no measurable population are `null` with `status: "no_data"`, never `0`.
Every value carries `good` / `partial` / `estimated` / `no_data`, and `partial` on a
sample under 10 means it is not yet a trend.

## Known limitations

Recorded in every snapshot rather than only here:

- Business hours exclude weekends but **not public holidays**.
- Bots are filtered by account type and a `[bot]` login suffix; a human account used for
  automation is not caught.
- A response is a comment or review by someone other than the author. **A fast formal
  reply does not mean the issue was resolved.**
- The external/internal split uses the collaborator list, so a contributor who later gains
  write access moves between populations across periods.

## Not individual measurement

Aggregated at project level only. The guideline is explicit that public GitHub data is not
a licence for profiling — no per-person rankings, no night-activity or sentiment inference:

> 개인보다 프로젝트·팀 수준으로 집계하고 작은 집단은 합친다.

With a small maintainer group, a per-person metric would effectively name individuals.
That is a reason for care, not a reason to collect more.

## Running it

```bash
GITHUB_TOKEN=$(gh auth token) python tools/collect_metrics.py
python -m pytest tools/test_collect_metrics.py -q
```

The response-time path cannot be exercised live while there are no external contributors,
so its logic is covered by deterministic tests instead — a metric that is correctly empty
and one that is broken look identical from the outside.
