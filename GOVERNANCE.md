# Governance

Who decides what in `trida-stack`, how a decision gets made, and where it is
written down. This document describes what we actually do. Where something is not
yet in force, it says so rather than describing an aspiration.

`trida-stack` is maintained by Trillion Labs as a research project. It is open to
outside contributions under [Apache-2.0](LICENSE), and this document applies to
everyone equally, maintainers included.

---

## Roles

| Role | Who | What they do |
|---|---|---|
| Contributor | anyone who opens an issue, discussion or PR | proposes changes; no permissions needed |
| Reviewer | any maintainer, or a contributor invited to a specific review | reviews PRs, approves or requests changes |
| Maintainer | the people listed below | merges PRs, cuts releases, owns the protected paths |
| Security contact | [security@trillionlabs.co](mailto:security@trillionlabs.co) | receives private vulnerability reports ([SECURITY.md](SECURITY.md)) |
| Conduct contact | [admin@trillionlabs.co](mailto:admin@trillionlabs.co) | receives conduct reports ([CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)) |

A role is a set of responsibilities, not a rank. The same person usually holds
several.

### Maintainers

| GitHub | Area |
|---|---|
| [@guks-trl](https://github.com/guks-trl) | training, two-stream kernels, model contract |
| [@hongjoonahn-tri](https://github.com/hongjoonahn-tri) | serving backends, evaluation |
| [@usik-luke-trl](https://github.com/usik-luke-trl) | release, supply chain, repository policy |

These three are also the code owners for every protected path in
[`.github/CODEOWNERS`](.github/CODEOWNERS): licensing, security policy, CI
workflows, the supply-chain documents, the tooling under `tools/`, and the kernel
fetch recipes.

Community-facing duty roles — community lead, technical duty, moderator, security
duty — are defined in
[`docs/community/CHANNEL_POLICY.md`](docs/community/CHANNEL_POLICY.md). They are
**not yet assigned by name**, which is why the chat channel has not launched.

---

## How a decision is made

**Default: lazy consensus on the pull request.** A PR with an approving review from
a maintainer, green CI, and no outstanding request for changes may be merged. If no
maintainer has responded within **two business days**, the author may ping on the PR;
silence is not approval, but it is also not a veto.

**Protected paths need a code-owner review.** Licence, security policy, CI, supply
chain, tooling and the kernel recipes are listed in `.github/CODEOWNERS` precisely
because a mistake there is quiet and expensive. The same rule keeps an AI coding
agent from changing licensing or security configuration without a human in the loop.

> **In force?** Not yet. The branch ruleset currently has
> `require_code_owner_review` set to `false`, so CODEOWNERS is advisory and GitHub
> will not block a merge. Maintainers follow it by convention until the setting is
> turned on. We state this rather than implying an enforcement we do not have.

**A request for changes blocks the merge** until the reviewer resolves it or another
maintainer explains, on the PR, why it is being overridden. Overriding in silence is
not available to anyone.

**Disagreement between maintainers** is settled on the PR or issue thread, in
writing, in English. If two maintainers disagree and neither moves, the change does
not land; either it is reshaped into something both accept, or it is closed with the
reason recorded. There is no tie-break vote and no casting vote, because with three
maintainers a majority rule would let two people quietly outvote the person who owns
the area.

**Anything irreversible gets a second maintainer.** Publishing a release, moving or
deleting a tag, changing the licence or `NOTICE`, rotating a credential, altering
branch protection, or making the repository's visibility change — none of these are
done by one person acting alone.

---

## Where decisions are recorded

A decision that lives only in someone's memory is not a decision anyone can audit.

| Decision | Recorded in |
|---|---|
| Why a change landed | the PR thread and the squashed commit message |
| What shipped in a version | [`CHANGELOG.md`](CHANGELOG.md) |
| Why a third-party component was adopted, and what we would accept instead | [`docs/oss/INTAKE_REGISTER.csv`](docs/oss/INTAKE_REGISTER.csv) and the record under [`docs/oss/intake/`](docs/oss/intake/) |
| What third-party code is in the product | [`docs/supply-chain/asset-register.csv`](docs/supply-chain/asset-register.csv) |
| Which licences may enter at all | [`docs/oss/LICENSE_POLICY.md`](docs/oss/LICENSE_POLICY.md) |
| What we decided about a vulnerability with no fix | [`docs/supply-chain/exceptions.md`](docs/supply-chain/exceptions.md) — with an approver and an expiry date that CI enforces |
| Licence obligations and the noncommercial boundary | [`COMPLIANCE.md`](COMPLIANCE.md) and [`NOTICE`](NOTICE) |
| Which channel an issue belongs in | [`COMMUNITY.md`](COMMUNITY.md) |

Issues and PRs are the record. We do not keep a separate decision log, because a
second place to write things down is a second place to forget.

---

## Releases

Versions follow [Semantic Versioning](https://semver.org/). Only the **latest
release** is supported — see [SECURITY.md](SECURITY.md), which also explains why
that line is the denominator of our SBOM-coverage metric.

Cutting a release:

1. The change is on `main` and CI is green.
2. `CHANGELOG.md` gets the version's entry in its own PR.
3. A maintainer tags and publishes the release.
4. The `release: published` event runs
   [`release-sbom.yml`](.github/workflows/release-sbom.yml), which builds the source
   archive, generates SPDX and CycloneDX SBOMs, attaches a SLSA provenance
   attestation, and uploads the four assets.

The pipeline refuses to overwrite assets on a release that already has them. A
published tag is never moved; if a tag was cut too early, the fix is the next patch
release. That is not a hypothetical — `v0.1.2` exists because `v0.1.1` was tagged
twenty-four minutes before a security dependency bump merged.

A security fix may skip the usual review latency, but never the second maintainer.

---

## Becoming a maintainer

There is no application form and no fixed quota. The path is a track record: several
merged non-trivial PRs, reviews that caught real problems, and sustained engagement
in issues or discussions over a few months. Any maintainer may propose a contributor;
the appointment needs the agreement of all current maintainers, and is recorded by a
PR that adds the person to the table above and to `.github/CODEOWNERS` where
relevant.

Maintainers who have been inactive for roughly six months are moved to an emeritus
line in the same PR-recorded way, with write access removed. This is housekeeping,
not a judgement — the point is that the list of people who can merge matches the
list of people who actually will. Anyone may step down at any time by opening that
PR themselves.

---

## Conduct

The [Code of Conduct](CODE_OF_CONDUCT.md) applies in every project space, including
pull requests, issues, discussions and any chat channel we open. Reports go to
[admin@trillionlabs.co](mailto:admin@trillionlabs.co), not to a public thread.
A maintainer who is the subject of a report recuses themselves from handling it.

---

## Changing this document

By pull request, like everything else, with the code-owner rule above. If what this
file says stops matching what we do, the file is the thing that is wrong and it gets
corrected — a governance document describing a process nobody follows is worse than
none.
