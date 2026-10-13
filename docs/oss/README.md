# Open-source intake and evaluation

How a third-party component gets into `trida-stack`, what we decided about the ones
already here, and when each decision is looked at again.

| file | what it is |
|---|---|
| [`LICENSE_POLICY.md`](LICENSE_POLICY.md) | 허용 / 조건부 / 금지 licence lists — the first gate |
| [`SCORECARD.md`](SCORECARD.md) | the five criteria, 20 points each, and the decision bands |
| [`INTAKE_REGISTER.csv`](INTAKE_REGISTER.csv) | one row per evaluated component: score, decision, condition, review date |
| [`intake/`](intake/) | the evaluation record behind each row |

## The procedure

1. **Licence gate.** Check the licence against [`LICENSE_POLICY.md`](LICENSE_POLICY.md).
   Prohibited, or no licence stated, ends it — no score is produced.
2. **Score.** Fill in [`intake/TEMPLATE.md`](intake/TEMPLATE.md) using
   [`SCORECARD.md`](SCORECARD.md). Alternatives considered go in the record, including
   "none, and here is why".
3. **Decide.** 80+ adopt, 60–79 adopt with a written condition, below 60 reject. A
   component can be conditional at any score when we carry a version constraint or
   depend on something else landing first.
4. **Record.** The intake file plus one row in
   [`INTAKE_REGISTER.csv`](INTAKE_REGISTER.csv), in the same PR as the code that
   introduces the component.
5. **Register it as an asset.** Add it to
   [`docs/supply-chain/asset-register.csv`](../supply-chain/asset-register.csv) and
   to [`NOTICE`](../../NOTICE).

Steps 4 and 5 are not duplicates. The asset register answers *what is in the
product*; the intake record answers *why we chose it and what we would accept
instead*. A generated SBOM answers neither.

## Re-evaluation

| Decision | Cycle |
|---|---|
| 채택 | annually |
| 조건부 채택 | every six months, or when the condition resolves — whichever is first |
| Any component | immediately on a licence change, a security finding upstream does not fix, or the component becoming unmaintained |

Review dates live in the register's `재평가 예정` column. A review is a PR that
either updates the record with what changed or restates the decision with a new
date — restating is a valid outcome, silence is not.

## What is recorded here, and what is not

These records cover components we **chose**. The fetched kernel recipes are a
different kind of entry: they are an install path assembled at build time from a
pinned upstream commit, not a dependency we select between. Their licence
arrangement, the paths they cover and the obligations they create are documented in
[`COMPLIANCE.md`](../../COMPLIANCE.md) §1b, with the pins enforced by
`tools/check_supply_chain.py` and verified weekly by `tools/verify_fetch.sh`.

## Honest note on dates

Seven of the eight records below were written on **2026-10-13**, after the decisions
they describe. The decisions are real and the reasoning is the reasoning that was
actually used — but the record postdates them, and each file says so in its
`결정 시점` row. We are not backdating evaluation dates to look like a process that
ran earlier than it did. From here on the record is written in the PR that
introduces the component.

## Who decides

Maintainers, per [`GOVERNANCE.md`](../../GOVERNANCE.md). A licence-policy change or
a new conditional adoption needs a code-owner review.
