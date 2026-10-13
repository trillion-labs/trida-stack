# Selection scorecard

How a third-party component is scored before it is adopted. Five criteria, 20 points
each, 100 total.

The score is not the decision. It makes the reasoning comparable between components
and visible a year later, when the person who made the call has forgotten why.

## Criteria

### A. Functional fit — 20

Does it do what we need, at the scale we need, without us rewriting it?

| | |
|---|---|
| 17–20 | covers the requirement directly; the extension points we need are public API |
| 12–16 | covers it, but we carry a patch, a plugin, or a wrapper |
| 6–11 | partial fit; a meaningful piece is ours to build |
| 0–5 | we would be using it for one function out of many |

### B. Licence fit — 20

Scored against [`LICENSE_POLICY.md`](LICENSE_POLICY.md).

| | |
|---|---|
| 20 | allowed licence, obligations are attribution only |
| 12–16 | allowed, but obligations need active upkeep (state changes, notice files) |
| 6–11 | conditional licence, condition is met and recorded |
| 0 | prohibited, or no licence stated — **the evaluation stops here** |

### C. Maintenance activity — 20

| | |
|---|---|
| 17–20 | regular releases, issues answered, more than one active maintainer |
| 12–16 | active, but the bus factor is low or releases are irregular |
| 6–11 | quiet; last release over a year ago, or effectively one maintainer |
| 0–5 | abandoned |

A frozen component can still score well here if freezing is the point — a reference
scorer that must not change scores on its stability, not its commit rate. Say so in
the record when that applies.

### D. Security posture — 20

| | |
|---|---|
| 17–20 | documented reporting path, advisories published, fixes land promptly |
| 12–16 | advisories exist, response time uneven |
| 6–11 | no stated policy, but the project is small and the attack surface is understood |
| 0–5 | known unfixed issues in the paths we use |

Open findings in the component at the time of evaluation are listed in the record and
cross-referenced to [`docs/supply-chain/exceptions.md`](../supply-chain/exceptions.md)
where one applies.

### E. Integration cost and exit cost — 20

How much work to adopt, and how much to leave.

| | |
|---|---|
| 17–20 | swappable behind our own interface; leaving is a day |
| 12–16 | adoption is contained; leaving means rewriting a module |
| 6–11 | it shapes our architecture; leaving is a project |
| 0–5 | effectively irreplaceable |

A low score here is not a veto. `torch` scores low and we adopted it anyway — but
recording it means nobody is surprised later.

## Decision

| Total | Decision |
|---|---|
| 80–100 | **채택 (adopt)** |
| 60–79 | **조건부 채택 (adopt with condition)** — the condition and its review date go in the record |
| below 60 | **반려 (reject)** — with the reason, so the same component is not re-proposed blind |

Two overrides, in both directions:

- **Any B of 0 is a rejection regardless of total.** A licence failure is not
  outweighed by a good fit.
- **A component may be 조건부 at any score** when we carry a version constraint, an
  unresolved upstream issue, or a dependency on someone else's work landing first.
  `transformers` scores 84 and is conditional, because our checkpoints do not load
  under 5.x. The score measures the component; the condition describes our situation.

## Using this template

Copy [`intake/TEMPLATE.md`](intake/TEMPLATE.md), fill it in, open a PR, and add one
row to [`INTAKE_REGISTER.csv`](INTAKE_REGISTER.csv). The PR thread is where the
disagreement, if any, is recorded.
