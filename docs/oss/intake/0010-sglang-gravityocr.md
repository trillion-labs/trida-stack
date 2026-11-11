# SGLang v0.5.10 + GravityOCR self-speculative patch

| | |
|---|---|
| 구분 | 수정포함 |
| 라이선스 | Apache-2.0 |
| 획득 경로 | https://github.com/sgl-project/sglang, tag v0.5.10 (commit `1519acf37c23f2189adb93f57ca9cd2db1bebf18`), cloned and patched by `inference/gravityocr/sglang/setup.sh` |
| 사용 위치 | `inference/gravityocr/` (its own server environment) |
| 평가 기록일 | 2026-10-30 |
| 결정 시점 | the PR that adds `inference/gravityocr/` |
| 결정 | **조건부 채택** |
| 총점 | **64 / 100** |
| 재평가 예정 | 2026-12-31, when the advisory exceptions it depends on expire; then every six months |

## Relation to 0002

[`0002-sglang.md`](0002-sglang.md) records SGLang as Zoro's diffusion serving backend.
That build is assembled by `inference/zoro-1/sglang/` from a different upstream (a
PolyForm-NC fork, see [`COMPLIANCE.md`](../../../COMPLIANCE.md) §1b) at a different pin.

This record is a separate component, because almost everything that the intake decision
rests on differs:

- the upstream is sgl-project/sglang itself, at tag v0.5.10;
- our changes are an Apache-2.0 patch to 25 upstream files and `python/pyproject.toml`,
  plus four new files, all committed in this repository (수정포함);
- it runs in its own Python 3.12 environment with its own resolved lock
  (`inference/gravityocr/sglang/requirements-lock.txt`), never installed together with
  `requirements.txt`.

0002 is unchanged by this record.

## Why we needed something here

GravityOCR decodes self-speculatively: each round drafts a block of tokens with the
block-diffusion head and verifies them with the autoregressive head of the same model.
Serving that needs an engine with paged KV cache, continuous batching and CUDA graphs,
native support for GLM-OCR's vision encoder and multimodal requests, and a
speculative-decoding worker interface that a new algorithm can be added to. Upstream
SGLang has all three: `srt/models/glm_ocr.py`, and the n-gram and EAGLE workers that our
worker is modelled on.

## Alternatives considered

| 후보 | 왜 선택하지 않았나 |
|---|---|
| Zoro's SGLang build (`inference/zoro-1/sglang/`) | a different fork at a different pin, under a noncommercial licence. Our patch does not apply to it, and building on it would put PolyForm-NC terms behind an otherwise Apache-2.0 path. |
| A newer SGLang release | rebasing is its own project: on the first later release we tried, 40 of the patch's 91 hunks did not apply. Later releases also require torch 2.11 or newer, CUDA 13 Python bindings and a newer FlashInfer than our fast path is written against. They do not remove the two CRITICAL deserialization sinks either (see D below); the `dill.loads()` call is still present in v0.5.23. |
| vLLM | serves base GLM-OCR autoregressively, but running our draft-and-verify decode would need an engine change of the same size. We did not build one. |
| Transformers in-process decoding | the GravityOCR repository keeps an in-process reference decoder, but it is not a serving engine: no batching across requests and no paged cache. Same reason as in 0002. |

## Score

| 기준 | 점수 | 근거 |
|---|---|---|
| A 기능 적합성 | 15 | covers the requirement, but we carry a 25-file patch and four new files. The fast path has been validated only on Hopper (sm_90) GPUs with a CUDA 12.x toolkit. |
| B 라이선스 적합성 | 16 | Apache-2.0, allowed. The obligations need upkeep: the patch modifies Apache-2.0 files, so the changes are stated (Apache-2.0 §4(b)) and the folder carries a NOTICE. |
| C 유지보수 활성도 | 15 | upstream is very active, but we are frozen on v0.5.10 and the patch does not carry over to later releases. Upstream activity reaches us only through a rebase. |
| D 보안 대응 | 7 | upstream publishes advisories, but five apply to v0.5.10 and none is fixed in a release we can move to (see below). This is between the 0–5 band (a known unfixed issue on a path we use) and 6–11. It is 7 rather than 5 or less because the one unfixed issue on our request path is LOW and local, and the CRITICAL sink in our runtime is behind a flag the launcher refuses. If either changes, this drops to the 0–5 band. |
| E 통합·이탈 비용 | 11 | the patch reaches into the scheduler, attention backend and CUDA-graph runner, and the fast path calls a private FlashInfer planning API at an exact version. Leaving means porting the worker to another engine, which is a project. |
| **총점** | **64** | |

**Open findings at evaluation time.** Each has an entry in
[`exceptions.md`](../../supply-chain/exceptions.md):

- **sglang 0.5.10**
  - PYSEC-2026-535 (CRITICAL, dill RCE behind `--enable-custom-logit-processor`):
    affected, mitigated.
  - PYSEC-2026-536 and -538 (CRITICAL, diffusion runtime only): not affected.
  - PYSEC-2026-3695 (LOW, multimodal cache hash): affected, mitigated.
  - PYSEC-2026-3064 (LOW, LoRA): not affected.
- **torch 2.9.1, pinned by SGLang**
  - PYSEC-2026-2286: affected, mitigated.
  - PYSEC-2025-195 and PYSEC-2026-139: not affected.
  - PYSEC-2025-194: the existing entry.
- **transformers 5.3.0, pinned by SGLang**
  - PYSEC-2026-4174, a repository-wide entry.
  - PYSEC-2026-3929 and -2290: the existing entries.
- **diskcache 5.6.3, through outlines**
  - PYSEC-2026-2447: not affected.

The lock raises 11 transitive pins above the environment the patch was developed in
(pillow, aiohttp, yarl, multidict, urllib3, filelock, setuptools, datasets, fsspec, anyio,
certifi), so that only the findings above remain.

## Decision

**Conditional.** 64 is in the conditional band, and these are the conditions:

1. **Loopback by default.**
   - `serve.sh` binds 127.0.0.1. A non-loopback `SERVE_HOST` is refused unless `API_KEY`
     is set. With a key it runs one tokenizer worker, the only mode in which SGLang v0.5.10
     checks keys.
   - Whenever the server runs one tokenizer worker, `serve.sh` also sets a random admin key
     that it never prints, which closes SGLang's management endpoints
     (`/update_weights_from_disk` among them). With the `FAST=1` default of two workers and
     no key they stay open to users of the host; `serve.sh` says so at startup.
   - `EXTRA_ARGS` accepts only a fixed list of logging and capacity options, by exact name.
     That keeps out `--enable-custom-logit-processor`, `--config`, `--host`, the keys, the
     LoRA options, other front-ends and transports (`--grpc-mode`, `--encoder-only`, DP
     attention, multi-node) and the model-loading options.
   - Clients must be trusted even with a key. The server fetches `http(s)` image URLs and
     reads local image paths that a request names. `ocr_request.py` sends images as
     `data:` URLs only.
2. **The exceptions above stay current.** They expire on 2026-12-31. Renewing them means
   re-reading each against the then-current SGLang releases.
3. **Isolated environment.**
   - The server environment's exact pins come from SGLang's own `pyproject.toml`: torch
     2.9.1, transformers 5.3.0, flashinfer-python 0.6.7.post2 and sglang-kernel 0.4.1.
   - Root and zoro-1 keep their own pins. [`0007-torch.md`](0007-torch.md) and
     [`0008-transformers.md`](0008-transformers.md) record this scope.
   - The environment is never mixed with `requirements.txt`.
4. **Conditional licences in that environment are recorded here.**
   - `pycountry` is LGPL-2.1, pulled in by `outlines` and `pydantic-extra-types`. It is
     installed unmodified from its own package and imported at run time, never vendored,
     which is the condition [`LICENSE_POLICY.md`](../LICENSE_POLICY.md) sets for LGPL.
   - `certifi`, `tqdm` and `orjson` include MPL-2.0 and are used unmodified.
   - NVIDIA's CUDA libraries, `cuda-python` and `nvidia-cutlass-dsl` come under NVIDIA's
     own licences. They are installed from NVIDIA's packages and not redistributed, like
     the CUDA libraries the root torch pin already installs.
5. **CI checks the patch weekly, not the engine.**
   - `setup.sh` checks that the clone is at the pinned commit and that the patch applies
     when someone runs it. The engine itself is checked only by a manual GPU run.
   - CI's CPU tests (`inference/gravityocr/test_gravityocr_patch.py`) check the committed
     files only: every SGLang commit quoted in the folder equals `BASE_COMMIT.txt`, the
     patch touches exactly the expected 26 files, the four new files parse and
     carry their licence headers, `setup.sh` is valid bash and installs with
     `--no-deps --no-config` from explicit indexes, and the lock names the base tag and the
     pins `setup.sh` checks. They do not fetch upstream, apply the patch or run `setup.sh`.
   - The weekly fetch-recipe job (`tools/verify_fetch.sh`, as for the Zoro recipe in 0002)
     fetches the pinned SGLang commit, applies the patch, copies the four new files in and
     checks that every patched and new source parses. It does not build the environment.

## Obligations we take on

- **Attribution.** Apache-2.0 attribution in [`NOTICE`](../../../NOTICE) and in
  `inference/gravityocr/NOTICE`.
- **Stated changes.** The modified upstream files are listed in the patch itself and
  summarised in the folder NOTICE.
- **Licence headers.**
  - The four new files carry a Trillion Labs Apache-2.0 header that names the upstream
    commit.
  - Two of them (`selfspec_diffusion_worker.py`, `selfspec_fast_commit.py`) also name
    the SGLang file they adapt.
- **Records.** Keep [`COMPLIANCE.md`](../../../COMPLIANCE.md) §1 and §2, the asset
  register row and the exception entries in step with the pin.

## Re-evaluation trigger

- **Dates.** 2026-12-31, when the exceptions expire, and every six months after that.
- **Earlier, immediately, on any of these:**
  - a new advisory against SGLang v0.5.10 on a path we run;
  - an upstream fix for PYSEC-2026-535 or -3695 in a release we could move to;
  - a licence change upstream;
  - a driver or CUDA toolkit change on the serving hosts (FlashInfer and Triton compile
    kernels at start-up against the local toolkit);
  - a decision to rebase, which also moves torch and transformers.
