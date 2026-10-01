# Approved vulnerability exceptions

Every advisory the scanners report must either be fixed or appear here. CI fails
otherwise, and CI fails again when an entry passes its `expires` date.

This file exists instead of a permanent `continue-on-error`. A scanner that cannot
fail teaches people to ignore it; an exception with an owner and an expiry turns a
finding into a decision that comes back around.

**Rules**

- An exception needs a reason, an approver and an expiry; `affected` entries also need
  a mitigation. "No impact" on its own is not an exception — environments change and
  the note outlives the judgement that produced it.
- `not_affected` means we checked the call sites, and the check is recorded in
  `evidence`. It is not an assumption.
- Expiry is a re-review date, not a deadline for the upstream fix. Renewing is fine;
  renewing without re-reading is not.

<!-- Machine-readable: tools/check_advisories.py parses the single YAML code fence
     below. Keep every entry inside it - an entry outside the fence is silently
     ignored by the gate. -->

```yaml
- id: PYSEC-2026-2289
  component: transformers
  status: affected
  reason: >-
    Malicious config.json can set _attn_implementation_internal to an attacker-controlled
    Hub repo; from_pretrained() then downloads and executes it, bypassing trust_remote_code.
    Fixed upstream in 5.3.0. We cannot take the fix: requirements.txt pins
    transformers>=4.57,<5 because the Trida checkpoints' remote code does not run on 5.x.
  mitigation: >-
    Load checkpoints and tokenizers only from trusted repositories (trillionlabs/*, and
    vetted upstreams). Prefer safetensors over pickle formats.
  evidence: 18 from_pretrained/save_pretrained call sites; the pin is in requirements.txt
  follow_up: >-
    Make the checkpoints 5.x-compatible so the pin can be lifted. A product task, not a
    dependency bump.
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2026-3929
  component: transformers
  status: affected
  reason: >-
    Path traversal via chat_template keys during save_pretrained(). Fixed upstream in
    5.10.0; blocked by the same <5 pin as PYSEC-2026-2289.
  mitigation: Download and save tokenizers only from trusted repositories.
  evidence: same call sites as PYSEC-2026-2289
  follow_up: shares the 5.x-compatibility task above
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2026-2288
  component: transformers
  status: not_affected
  reason: >-
    HF Trainer._load_rng_state() calls torch.load() without weights_only. We do not use
    HF Trainer (training is native FSDP2), and the advisory's precondition torch<2.6 does
    not hold either - we pin torch 2.11.0.
  evidence: >-
    grep for 'transformers import Trainer' and 'TrainingArguments' returns 0 matches.
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2025-217
  component: transformers
  status: not_affected
  reason: X-CLIP checkpoint conversion path. We do not use X-CLIP.
  evidence: grep for 'xclip' and 'x_clip' returns 0 matches.
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2026-2290
  component: transformers
  status: not_affected
  reason: LightGlue model loading path. We do not use LightGlue.
  evidence: grep for 'lightglue' returns 0 matches.
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2025-194
  component: torch
  status: affected
  reason: >-
    Memory corruption in torch.jit.script; local attack. Fixed in 2.13.0. Upgrading
    requires re-validating the CUDA build against the cluster driver, so it is scheduled
    rather than immediate.
  mitigation: >-
    The vulnerable API is not called - grep for 'jit.script' and 'jit.trace' returns
    0 matches.
  follow_up: evaluate the torch 2.13.0 upgrade
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2026-3447
  component: setuptools
  status: not_affected
  reason: >-
    MANIFEST.in exclusion bypass on macOS APFS/HFS+ when building an sdist, letting an
    excluded file be packed and published. We publish no sdist to any index -
    distribution is the git repository and the GitHub release tarball. Our build-backend
    floor is now setuptools>=83 in both pyproject.toml files; the scanner reports this
    against the resolver's setuptools, not a declared dependency.
  evidence: >-
    setuptools does not appear in requirements.txt; pip-audit surfaces it from the
    resolution environment.
  approver: Trillion Labs research team
  expires: 2026-12-31
```
