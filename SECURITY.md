# Security Policy

We take the security of `trida-stack` seriously. Thank you for helping keep the
project and its users safe.

## Supported versions

**The latest release only.** Older tags receive no fixes, no advisories and no SBOM.

| Version | Supported |
|---|---|
| latest release | yes |
| anything older | no |

This is a research project with a small maintainer group, and a support window we
cannot honour is worse than a narrow one stated plainly. If you need a fix on an
older tag, open an issue and say why -- we would rather have that conversation than
imply a guarantee.

This line also defines a measured quantity. The supply-chain guideline computes
SBOM coverage as *releases with a valid SBOM / supported releases*, so "supported"
is the denominator: with the latest release carrying its SBOM, coverage is 100%,
and the un-described older tags are out of scope by policy rather than by omission.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
pull requests, or discussions.**

Instead, use either private channel:

- **GitHub private vulnerability reporting** — the **Report a vulnerability**
  button under the repository's [Security tab](https://github.com/trillion-labs/trida-stack/security/advisories).
  The report, and the whole thread, stay private until an advisory is published.
- **Email** — **[security@trillionlabs.co](mailto:security@trillionlabs.co)**,
  a role alias rather than one person's inbox.

Either reaches the maintainers; the GitHub route keeps the report attached to the
repository and lets us credit you on the advisory.

Please include as much of the following as you can:

- A description of the issue and its potential impact.
- Steps to reproduce, or a proof-of-concept.
- Affected files, commit, branch, or release.
- Any suggested remediation.

You may optionally encrypt sensitive details; ask in your first email and we will
arrange a secure channel. Please give us a reasonable window to investigate and
ship a fix before any public disclosure (coordinated disclosure).

## Response expectations

- **Acknowledgement:** within **3 business days** of your report.
- **Initial assessment / triage:** within **10 business days**.
- **Fix or mitigation plan:** communicated after triage, with timelines that
  depend on severity and complexity.
- We will keep you informed of progress and let you know when a fix is released.
  With your permission, we are happy to credit you in the release notes.

## Scope

In scope:

- Source code in this repository (the Apache-2.0 training and serving stack:
  `train/`, `inference/`, `benchmark/` wrappers, and supporting scripts).
- The fetch/patch recipes and configuration shipped here.
- Build, packaging, and CI configuration in this repository.

Out of scope (report to the respective upstream project):

- Third-party runtime dependencies and backends installed separately — PyTorch,
  Hugging Face Transformers/Datasets/Accelerate, SGLang, vLLM, and the benchmark
  harnesses (see [`COMPLIANCE.md`](COMPLIANCE.md)).
- The PolyForm-Noncommercial two-stream Gated-DeltaNet / block-diffusion kernels,
  which are **not bundled** here but fetched from upstream
  ([yuchen-zhu-zyc/HybridDiffusion](https://github.com/yuchen-zhu-zyc/HybridDiffusion));
  report kernel issues upstream.
- Trida model weights and their inference behavior (model-card issues), unless
  the issue is in this repository's code.

## Reporting exposed secrets

This repository's git history was **scrubbed of secrets** prior to public
release. If you nonetheless spot anything that looks like a credential, API key,
token, private key, or other secret — in the current tree, the git history, CI
logs, or an artifact — **please report it privately** to
[security@trillionlabs.co](mailto:security@trillionlabs.co) rather than opening a
public issue or PR. Do not attempt to use any such credential.

Thank you for practicing responsible disclosure.
