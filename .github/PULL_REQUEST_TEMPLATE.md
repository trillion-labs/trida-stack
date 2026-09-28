## Summary

<!-- What does this PR do and why? Link any related issue: Closes #123 -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactor / maintenance
- [ ] CI / build

## Checklist

- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] CPU-safe tests pass (`pytest -q inference/test_smoke.py inference/vllm/vllm_native_diffusion/test_two_stream_cpu.py`)
- [ ] Docs updated if behavior, flags, or the model contract changed
- [ ] `COMPLIANCE.md` / `NOTICE` updated if dependencies or third-party code changed
- [ ] No PolyForm-Noncommercial kernels bundled (they stay as fetch/patch recipes — see `CONTRIBUTING.md`)
- [ ] No secrets, credentials, or large binaries added

## Notes for reviewers

<!-- Anything that needs GPU verification, upstream coordination, or special context. -->
