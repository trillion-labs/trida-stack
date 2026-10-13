# Licence policy

Which licences may enter `trida-stack`, which may enter only with a recorded
condition, and which may not enter at all.

This list is the first gate in [the intake procedure](README.md). A component that
fails it is not scored — the evaluation stops there.

## Allowed

Permissive licences whose obligations we discharge by keeping attribution in
[`NOTICE`](../../NOTICE) and shipping the licence text.

| Licence | Obligation we take on |
|---|---|
| Apache-2.0 | attribution, state changes, keep the patent grant intact |
| MIT | attribution |
| BSD-2-Clause, BSD-3-Clause | attribution; no endorsement claims |
| ISC | attribution |
| Python-2.0, PSF | attribution |
| Unlicense, CC0-1.0 | none |

A component under an allowed licence still has to be recorded in
[`INTAKE_REGISTER.csv`](INTAKE_REGISTER.csv) and in
[`docs/supply-chain/asset-register.csv`](../supply-chain/asset-register.csv). Allowed
is not the same as automatic.

## Conditional

These enter only with the condition written into the intake record, and only where
the condition can actually be met.

| Licence | Condition |
|---|---|
| MPL-2.0, EPL-2.0, CDDL | file-level copyleft. Allowed as an **unmodified** dependency. If we modify a covered file, that file's source must be published under the same licence — so a modification needs a decision, not a commit. |
| LGPL-2.1, LGPL-3.0 | allowed only as a dynamically linked runtime dependency, never vendored or statically linked. |
| BSD-4-Clause | the advertising clause is incompatible with how we distribute; allowed only if upstream relicenses or grants a waiver in writing. |
| Dual-licensed (e.g. GPL-or-commercial) | allowed only if we take, and record, the permissive side. |
| No licence file at all | treated as all-rights-reserved. Allowed only after upstream states a licence in writing; the statement is linked from the intake record. |

## Prohibited

| Licence | Why |
|---|---|
| GPL-2.0, GPL-3.0, AGPL-3.0 | repository-wide copyleft. `trida-stack` is distributed under Apache-2.0; linking these would compel us to relicense, and AGPL extends that to network use. |
| SSPL, BUSL, Elastic License, Commons Clause | source-available, not open source. They restrict fields of use we cannot bound in advance. |
| Noncommercial licences (CC-BY-NC, PolyForm Noncommercial) | may not be **committed to this repository**. See below. |
| Any licence with a field-of-use or user restriction (most "responsible AI" licences, Llama-style acceptable-use terms) | we cannot pass a restriction we did not write on to downstream users of an Apache-2.0 repository. |

### The noncommercial boundary is a real line, not a grey area

Noncommercial-licensed upstream code is **never committed here**. Where such code is
part of an install path, it is fetched and patched at setup time from a pinned
upstream commit, and the repository carries only the recipe. That arrangement, the
paths it applies to and the obligations it creates are documented in
[`COMPLIANCE.md`](../../COMPLIANCE.md) §1b and
[`NOTICE`](../../NOTICE). `tools/check_supply_chain.py` fails CI if a pinned commit
quoted anywhere in the repository disagrees with the recipe.

## Licence of our own contributions

Everything first-party in this repository is Apache-2.0, including contributions
from outside — see [`CONTRIBUTING.md`](../../CONTRIBUTING.md). We do not accept
code into this repository under any other licence.

## When this list changes

By pull request, with a code-owner review ([`GOVERNANCE.md`](../../GOVERNANCE.md)).
Moving a licence from prohibited to conditional, or conditional to allowed, requires
the agreement of all maintainers, because the change applies retroactively to
everything already in the tree.
