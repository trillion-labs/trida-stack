# Supply chain

What third-party code enters trida-stack, how it is checked, and what happens when a
finding has no fix. Applies the NIPA/오픈업 오픈소스 공급망 관리 가이드라인 to this
repository rather than reproducing it.

| file | what it is |
|---|---|
| [`asset-register.csv`](asset-register.csv) | 통합자산목록 — one row per component, 직접개발 / 외부유입 / 수정포함 |
| [`exceptions.md`](exceptions.md) | approved vulnerability exceptions; CI enforces them |
| [`../../COMPLIANCE.md`](../../COMPLIANCE.md) | the licence narrative and redistribution obligations |

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

The checksums matter beyond the pin. A pinned commit protects against upstream *moving*;
recorded hashes additionally detect the history being rewritten under the same ref.

## CI

| check | when | fails on |
|---|---|---|
| Pinned-commit drift | PR, push | the recorded pin disagreeing with `fetch_kernels.sh` |
| Dependency advisories | PR, push, weekly | a finding with no unexpired exception |
| Dependency review | PR | new deps at high severity or a denied licence |
| Fetch recipe | weekly | the recipe or its integrity check failing |
| Release SBOM | on release | invalid SBOM, or a finding with no exception |

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

## Known gaps

- **Base-model licence.** Trida is derived from Qwen3/Qwen3.5. `COMPLIANCE.md` covers
  software dependencies and evaluation datasets but does not state the base model's
  licence or what it implies for the derivative. Derived-model licence propagation is a
  real question and this is not yet answered anywhere in the repository.
- **No AI-BOM.** Training data provenance, training configuration and evaluation results
  are not tracked as supply-chain artifacts.
- **Most dependencies are unpinned** and there is no lockfile, so advisory scanning is
  against declared ranges rather than a resolved set.
- **`CODEOWNERS` names a team that may not exist yet** — it does nothing until the
  handle is real.
