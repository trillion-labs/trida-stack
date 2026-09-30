# Supply-chain management

How third-party components enter trida-stack, how they are inventoried, scanned, and
dispositioned, and what a downstream consumer receives. Structure follows the
공급망관리 guideline (Lv3 요건: SBOM 정기 생성 + 자동 취약점 점검 + 납품 고지패키지 +
**조치 이력 관리**).

| artifact | what it is |
|---|---|
| [`ASSET_INVENTORY.csv`](ASSET_INVENTORY.csv) | 통합자산목록 — one row per component, 직접개발 / 외부유입 / 수정포함 |
| [`sbom/external-components.json`](sbom/external-components.json) | components no scanner can see, merged into every generated SBOM |
| [`VULN_DISPOSITION.csv`](VULN_DISPOSITION.csv) | 취약점 조치대장 — every finding and what was decided about it |
| [`../../.github/workflows/release-sbom.yml`](../../.github/workflows/release-sbom.yml) | SPDX + CycloneDX per release, plus the notice package |
| [`../../.github/workflows/supply-chain.yml`](../../.github/workflows/supply-chain.yml) | pin-drift check, pip-audit, dependency review |
| [`../../tools/make_notice_package.sh`](../../tools/make_notice_package.sh) | 납품 고지패키지 assembly |

## Why the SBOM needs a hand-written fragment

A generated SBOM describes what a dependency manifest declares. Four of our components
are invisible to that process:

- the three **PolyForm-Noncommercial** kernel sets, which are *fetched at build time*
  from a pinned upstream commit and never committed here
- the **vendored IFEval scorer** in `inference/ifeval_lib/`, in-tree source with no manifest

An SBOM that omits them would be worse than no SBOM: it would tell a consumer, with
authority, that the build contains no noncommercial code. So `sbom/external-components.json`
describes them by hand and the release workflow merges it in.

A hand-written record is only trustworthy if something stops it going stale. That is
[`tools/check_supply_chain.py`](../../tools/check_supply_chain.py), which fails CI if the
pinned commit recorded anywhere in the repository disagrees with what `fetch_kernels.sh`
actually checks out. It currently guards 18 references across 11 files plus the fragment.

## Scanning, and why it does not block

`pip-audit` runs on every push, every PR, and weekly. It is `continue-on-error` on purpose:
`requirements.txt` is largely unpinned by design (the CUDA build is chosen per cluster), so
a blocking gate would stop unrelated PRs on transitive advisories that cannot be fixed in
that PR. The requirement the standard actually sets is **조치까지, not 스캔까지** — so the
gate that matters is the disposition log, reviewed quarterly, not a red X.

Dependency review does block on PRs, at `fail-on-severity: high`, with a deny-list mirroring
[`LICENSE_POLICY.md`](LICENSE_POLICY.md). Note it cannot see PolyForm-NC: that arrives by
build-time fetch, not through the dependency graph. The pin-drift job covers that path.

## The 2026-09-30 baseline

First scan: **11 findings across 3 packages.** Triaged in `VULN_DISPOSITION.csv`:

- **1 fixed** — setuptools build floor raised to `>=83` (PYSEC-2026-3447)
- **3 not applicable** — verified by checking the call sites, not by assumption: we do not
  use HF `Trainer`, `torch.jit.script`, X-CLIP or LightGlue
- **1 in progress** — torch 2.13.0 upgrade, gated on CUDA build compatibility
- **2 accepted with mitigation** — and these are the ones worth reading

### The accepted risk

`transformers` **PYSEC-2026-2289** is a critical RCE: a malicious `config.json` can set
`_attn_implementation_internal` to an attacker-controlled Hub repo, and
`from_pretrained()` will download and execute it, **bypassing `trust_remote_code`**. There
are 18 `from_pretrained`/`save_pretrained` call sites in this repository.

It is fixed in transformers 5.3.0. **We cannot take that fix**: `requirements.txt` pins
`transformers>=4.57,<5` because the Trida checkpoints' remote code does not work on 5.x.
PYSEC-2026-3929 (path traversal via `save_pretrained`) has the same shape and the same block.

So the disposition is **수용 with mitigation**: load checkpoints only from trusted
repositories. The real fix is to make the checkpoints 5.x-compatible so the pin can be
lifted, which is a product task, not a dependency bump. It is recorded here rather than
left implicit.

## A note on publishing this

Everything in the disposition log is a **publicly known upstream advisory affecting a
version pin that is already visible in `requirements.txt`**. Anyone can run `pip-audit`
against this repository and get the same list. Publishing our triage therefore discloses
no new attack surface; it discloses that we know, and what we decided.

Defects in *our own* code are handled the other way round — privately, via
[`SECURITY.md`](../../SECURITY.md) — and are never triaged in public before a fix ships.
