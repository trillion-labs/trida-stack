# Supply chain

What third-party code enters trida-stack, how it is checked, and what happens when a
finding has no fix. Applies the NIPA/오픈업 오픈소스 공급망 관리 가이드라인 to this
repository rather than reproducing it.

| file | what it is |
|---|---|
| [`asset-register.csv`](asset-register.csv) | 통합자산목록 — one row per component, 직접개발 / 외부유입 / 수정포함 |
| [`exceptions.md`](exceptions.md) | approved vulnerability exceptions; CI enforces them |
| [`../../COMPLIANCE.md`](../../COMPLIANCE.md) | the licence narrative and redistribution obligations |
| [`../oss/`](../oss/) | 프로젝트 평가 — licence policy, selection scorecard, and the intake record behind each adopted component |

The register and the SBOM are not duplicates. A generated SBOM carries components and
versions; it cannot carry why something was adopted, who owns it, or what was decided
about a finding. The register holds that, which is why both exist.

## What is unusual about this repository

The install path is not only `pip install`. Three code paths assemble third-party
sources **at build time** from a pinned upstream commit:

| path | recipe |
|---|---|
| two-stream training kernels | `train/block_gated_delta_rule/fetch_kernels.sh` |
| SGLang diffusion backend | `inference/sglang/` |
| vLLM block-end readout kernel | `inference/vllm/vllm_native_diffusion/KERNELS.md` |

All three resolve to `yuchen-zhu-zyc/HybridDiffusion@6ca547a`, which is
PolyForm Noncommercial 1.0.0 and is therefore **never committed here**. See
[`../../COMPLIANCE.md`](../../COMPLIANCE.md) §1b.

Two consequences shape the CI below.

**Dependency scanners cannot see these components.** They are not in any manifest, so
`dependency-review` and the dependency graph are blind to them. The register records
them by hand, and `tools/check_supply_chain.py` fails CI if the pinned commit quoted
anywhere in the repository disagrees with what `fetch_kernels.sh` checks out.

**The recipes are the install path.** If upstream moves, renames, force-pushes, or
changes the subpath, nobody can build either two-stream path — and we would normally
find out from a bug report. `tools/verify_fetch.sh` runs the recipe weekly against real
upstream: the pin resolves, the subpath exists, the 18 sources match
`tools/kernel-checksums.txt`, the patch applies, the result parses.

The workflows are a supply chain of their own. A tag like `@v4` is a pointer upstream
can move, and a compromised account moves it silently -- this is how
`tj-actions/changed-files` was used to exfiltrate secrets from thousands of
repositories in March 2025. The actions that generate our SBOM and sign its
attestation are the ones a reader is trusting most, so every action is pinned to a
40-character commit SHA with the human-readable version in a trailing comment, and
`tools/check_action_pins.py` fails CI if a tag comes back.

### Updating an action pin

The SHA comes from asking GitHub what the tag pointed at, at that moment:

```bash
gh api repos/<owner>/<repo>/git/ref/tags/<tag> --jq '.object.sha'
# annotated tag -> one more hop:
gh api repos/<owner>/<repo>/git/tags/<sha> --jq '.object.sha'
```

**A pin freezes a point in time; it does not verify it.** If the tag had already been
moved before the lookup, the pin records the moved commit. So cross-check the SHA
against the release page for the version in the trailing comment before trusting a new
pin, and treat the first pin of an action as the moment you decided to trust it.

What the pin does buy is that nothing changes underneath us afterwards, and that any
later change is a visible diff in a pull request rather than a silent one.

The checksums matter beyond the pin. A pinned commit protects against upstream *moving*;
recorded hashes additionally detect the history being rewritten under the same ref.

## CI

| check | when | fails on |
|---|---|---|
| Pinned-commit drift | PR, push | the recorded pin disagreeing with `fetch_kernels.sh` |
| Action pins | PR, push | a workflow running a GitHub Action from a tag instead of a commit SHA |
| Dependency advisories | PR, push, weekly | a finding with no unexpired exception |
| Fetch recipe | weekly | the recipe or its integrity check failing |
| Published model card | weekly | HF licence or visibility disagreeing with `COMPLIANCE.md` §4b |
| Release SBOM | on release | invalid SBOM, or a finding with no exception |

## Licence policy for new dependencies

Not yet enforced in CI — `actions/dependency-review-action` requires **Settings →
Security → Dependency graph**, which is currently off for this repository, so the check
could not pass and was left out rather than merged red. Once it is enabled, adding the
workflow is a small follow-up.

The policy it will enforce, and which applies to review in the meantime:

| | |
|---|---|
| fail on | severity `high` or above in newly added dependencies |
| deny | `GPL-2.0`, `GPL-3.0`, `AGPL-3.0` — copyleft that would propagate into this Apache-2.0 distribution |
| deny | `SSPL-1.0` and similar source-available licences whose service restrictions do not fit how we distribute |

Note this gate would not see the PolyForm-NC kernels either: they arrive by build-time
fetch, not through the dependency graph. The pin-drift job covers that path.

## Why the scanners do not fail the build directly

Scanners run in report mode and `tools/check_advisories.py` is the gate. That split is
deliberate. The guideline puts it plainly:

> 영구적인 `continue-on-error: true`는 관리가 아니라 무시다.

A scanner that cannot fail trains people to ignore it. A scanner wired straight to
`fail-build` on a project with an unfixable pin fails every run until someone disables
it — same outcome, more steps. So every finding must be fixed or carry an entry in
[`exceptions.md`](exceptions.md) with a reason, a mitigation, an approver and an expiry,
and CI fails when an entry expires.

That last part is the point. It turns an accepted risk from a note that outlives its
own reasoning into a decision that comes back around.

## Supported versions, and why it is a number

[`SECURITY.md`](../../SECURITY.md) supports **the latest release only**. That is the
denominator of the guideline's SBOM-coverage metric -- *releases with a valid SBOM /
supported releases* -- so the older tags that predate this machinery are out of scope
by policy rather than counted as gaps.

Published assets are never overwritten: `release-sbom.yml` fails if the tag already
has any. A corrected SBOM means a patch release, because replacing one in place
rewrites what consumers were told a tag contains and leaves no trace.

## What the release carries

`release-sbom.yml` builds a source tarball of the tag, checksums it, generates SPDX and
CycloneDX from the same scan boundary and syft version, validates both, scans the SPDX
through the exception gate, attests provenance and the SBOM, and uploads the tarball,
both SBOMs and `SHA256SUMS`.

SPDX and CycloneDX are kept for different jobs rather than as duplicates: SPDX for the
licence and audit side, CycloneDX for the security side — and as the place VEX will go,
once the triage in `exceptions.md` has settled. VEX would let a consumer's scanner read
our "not affected, here is why" analysis instead of re-deriving it.

The tarball contains no PolyForm-NC source, because none is committed. The SBOM is
therefore accurate about what we distribute, and the register covers what gets fetched.

## The model is not in this repository

`COMPLIANCE.md` §4b records the weights' licence and the base model's, and
`tools/check_model_card.py` checks weekly that those claims still match what Hugging Face
publishes. The repository cannot fix a wrong model card — it can refuse to go on
describing it incorrectly.

Apache-2.0 on the weights does not make the serving stack permissive: the diffusion and
self-speculative decode paths run §1b kernels that are PolyForm Noncommercial, and that
restriction attaches to whoever runs them. The AR path has no such dependency.

## Known gaps

- **No AI-BOM.** Training configuration and evaluation results are not tracked as
  supply-chain artifacts. Training data provenance is deliberately out of scope —
  it is internal and proprietary, as recorded in `data/DATA_CATALOG.md`.
- **Dependency graph is disabled**, so dependency review and Dependabot security updates
  cannot run. One repository setting.
- **Code-owner review is not enforced.** `.github/CODEOWNERS` exists, but the branch
  ruleset has `require_code_owner_review: false` and `required_approving_review_count: 0`,
  so the file currently documents intent rather than gating anything.
- Dependencies are declared as ranges in `requirements.txt`; the resolved set lives in
  `requirements-lock.txt` (85 packages) and `inference/requirements-lock.txt` (49).
  Regenerate with the command in each file's header -- the flags matter, since the
  resolution is platform-specific. Advisory scanning still reads the declared ranges;
  the lockfiles exist so the release SBOM describes something real.
