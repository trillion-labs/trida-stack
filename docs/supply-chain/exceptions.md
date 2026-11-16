# Approved vulnerability exceptions

Every advisory the scanners report must either be fixed or appear here. CI fails
otherwise, and CI fails again when an entry passes its `expires` date.

This file exists instead of a permanent `continue-on-error`. A scanner that cannot
fail teaches people to ignore it; an exception with an owner and an expiry turns a
finding into a decision that comes back around.

**Rules**

- An exception lists every id the finding is known by. Scanners disagree on which is
  primary -- pip-audit leads with PYSEC, grype with GHSA -- and an exception keyed to
  one of them silently fails to cover the other. The first eight entries here were all
  approved and still blocked a release because of exactly that.
- An exception needs a reason, an approver and an expiry; `affected` entries also need
  a mitigation. "No impact" on its own is not an exception — environments change and
  the note outlives the judgement that produced it.
- `not_affected` means we checked the call sites, and the check is recorded in
  `evidence`. It is not an assumption.
- Expiry is a re-review date, not a deadline for the upstream fix. Renewing is fine;
  renewing without re-reading is not.
- The gate passes a package once any one of its ids is excepted (it groups findings by
  package, `tools/check_advisories.py`), so a new advisory on an excepted package is not
  flagged until the gate matches per finding. An id excepted once is also excepted in every
  scanned file. An entry written for one environment therefore says which
  environments its evidence covers. The entries for `inference/gravityocr/` come from
  scanning that recipe's own lock (`inference/gravityocr/sglang/requirements-lock.txt`,
  a separate Python 3.12 environment); where the same id can also reach the root or
  zoro-1 environment, the entry says so and its evidence covers both.

<!-- Machine-readable: tools/check_advisories.py parses the single YAML code fence
     below. Keep every entry inside it - an entry outside the fence is silently
     ignored by the gate. -->

```yaml
- id: PYSEC-2026-2289
  aliases: [GHSA-29pf-2h5f-8g72, CVE-2026-4372]
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
  aliases: [GHSA-xrqw-3rrv-vx5w, CVE-2026-9856]
  component: transformers
  status: affected
  reason: >-
    Path traversal via chat_template keys during save_pretrained(). Fixed upstream in
    5.10.0; blocked by the same <5 pin as PYSEC-2026-2289. The inference/gravityocr/
    server environment is also below the fix: it runs transformers==5.3.0 because SGLang
    v0.5.10 pins it exactly (python/pyproject.toml:74).
  mitigation: Download and save tokenizers only from trusted repositories.
  evidence: >-
    Root and zoro-1: same call sites as PYSEC-2026-2289. inference/gravityocr/ server:
    SGLang calls save_pretrained() on a tokenizer only on its ModelOpt export path
    (ModelOptModelLoader, model_loader/loader.py:2762), and nvidia-modelopt is not in
    inference/gravityocr/sglang/requirements-lock.txt. The only other save_pretrained in
    python/sglang is the DotsOCR config class (srt/configs/dots_ocr.py:26), a model
    GravityOCR does not use. serve.sh and ocr_request.py save nothing.
  follow_up: shares the 5.x-compatibility task above
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2026-2288
  aliases: [GHSA-69w3-r845-3855, CVE-2026-1839]
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
  aliases: [CVE-2025-14929]
  component: transformers
  status: not_affected
  reason: X-CLIP checkpoint conversion path. We do not use X-CLIP.
  evidence: grep for 'xclip' and 'x_clip' returns 0 matches.
  note: >-
    Aliases corrected on 2026-10-30. This entry used to list GHSA-x9r9-c232-4q39 and
    CVE-2026-80047, which OSV maps to PYSEC-2026-4174 (custom_generate), not to this
    advisory; its own alias is CVE-2025-14929. Until then, root's PYSEC-2026-4174 finding
    passed the gate through this entry. It now has its own entry below, written for every
    environment.
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2026-2290
  aliases: [GHSA-fgcw-684q-jj6r, CVE-2026-5241]
  component: transformers
  status: not_affected
  reason: >-
    LightGlue model loading path. We do not use LightGlue. Also reported against the
    inference/gravityocr/ server environment's transformers 5.3.0, which serves only
    GLM-OCR.
  evidence: >-
    grep for 'lightglue' returns 0 matches. A case-insensitive grep for 'lightglue' over
    python/sglang with inference/gravityocr/sglang/gravityocr_selfspec.patch applied also
    returns 0 matches.
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2025-194
  aliases: [GHSA-rrmf-rvhw-rf47, CVE-2025-3000]
  component: torch
  status: affected
  reason: >-
    Memory corruption in torch.jit.script; local attack. Fixed in 2.13.0. Upgrading
    requires re-validating the CUDA build against the cluster driver, so it is scheduled
    rather than immediate. The inference/gravityocr/ server environment runs torch 2.9.1,
    pinned exactly by SGLang v0.5.10 (python/pyproject.toml:66), so it can only move with
    an SGLang rebase.
  mitigation: >-
    Our code does not call it: grep for 'jit.script' and 'jit.trace' returns 0 matches in
    this repository outside this file. That grep is not true of the SGLang tree the
    inference/gravityocr/ recipe builds: the model registry imports every model module
    (srt/models/registry.py:93-103), so @torch.jit.script runs at import time on fixed
    upstream helpers (quick_gelu and gegelu in srt/models/deepseek_ocr.py:991 and
    srt/models/phi3_small.py:31,36). None of them compiles code taken from a request or a
    checkpoint. The fourth use, multimodal_gen/runtime/models/vaes/dac.py:21, is in the
    diffusion runtime, which serve.sh never imports (see PYSEC-2026-536).
  follow_up: >-
    evaluate the torch 2.13.0 upgrade; for inference/gravityocr/, it comes with an SGLang
    rebase
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2026-3447
  aliases: [GHSA-h35f-9h28-mq5c, CVE-2026-59890]
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
- id: PYSEC-2026-3740
  aliases: [GHSA-8mgp-746c-j5xp, CVE-2026-81726]
  component: nltk
  status: not_affected
  reason: >-
    File sandbox bypass in model import/export helpers - TransitionParser.train/parse,
    AveragedPerceptron.save/load, PerceptronTagger.save_to_json, save_maxent_params.
    Not yet patched upstream. The advisory requires the application to enable pathsec
    enforcement and let untrusted input choose model paths; we do neither.
  evidence: >-
    nltk is used only by the vendored IFEval scorer, for word_tokenize,
    RegexpTokenizer and loading the punkt tokenizer. grep for TransitionParser,
    AveragedPerceptron, PerceptronTagger, save_maxent_params and pathsec returns
    0 matches across the repository.
  note: >-
    Unrelated to this advisory, instructions_util.py loads punkt via a .pickle. That is
    NLTK's own downloaded data rather than untrusted input, but it is the kind of
    deserialization path worth revisiting if the scorer is ever fed external data.
  approver: Trillion Labs research team
  expires: 2026-12-31

# ---- inference/gravityocr/ server environment ----
# Found by scanning inference/gravityocr/sglang/requirements-lock.txt. That environment is
# upstream sgl-project/sglang v0.5.10 (commit 1519acf37c23f2189adb93f57ca9cd2db1bebf18)
# plus inference/gravityocr/sglang/gravityocr_selfspec.patch, in its own Python 3.12 venv.
# Line numbers below are in python/sglang/ at that upstream commit. The seven new files the
# patch needs, and the upstream files the patch changes where the change matters here
# (entrypoints/http_server.py, utils/auth.py), are cited by function name: the patch and the
# licence headers shift their lines.

- id: PYSEC-2026-535
  aliases: [GHSA-36m8-w8qf-g76p, CVE-2026-7304]
  component: sglang
  status: affected
  reason: >-
    Unauthenticated RCE: with --enable-custom-logit-processor set, the srt runtime
    deserializes a request's custom_logit_processor with dill.loads()
    (srt/sampling/custom_logit_processor.py:15-20). The sink is in the runtime that
    inference/gravityocr/serve.sh starts (python -m sglang.launch_server), but the flag
    defaults to False and the launcher never sets it. No upstream fix: the same
    dill.loads() call is still present in v0.5.23. OSV's affected range ends where it does
    only because of when the advisory was published.
  mitigation: >-
    The flag stays off. With it off, tokenizer_manager.py rejects any request that carries
    custom_logit_processor (ValueError at :858-865), and sampling_batch_info.py only
    deserializes when the flag is set (:120-139). serve.sh's EXTRA_ARGS accepts only a
    fixed list of logging and capacity options, exact names, no abbreviations. The list
    has neither the flag nor --config, through which a YAML file could set it, and the
    launcher's own values are checked, so none of them can carry a '--config' token
    either. serve.sh binds 127.0.0.1 unless SERVE_HOST is set, and refuses a non-loopback
    SERVE_HOST unless API_KEY is set. serve.sh runs one tokenizer worker unless
    TOKENIZER_WORKERS asks for more, and the patch makes every tokenizer worker check the keys
    (upstream v0.5.10 checks them only with one worker), so a key is enforced with that
    opt-in as well.
  evidence: >-
    server_args.py:656 defaults enable_custom_logit_processor to False. In
    inference/gravityocr/, the string 'enable-custom-logit-processor' appears only in the
    launcher test's refused inputs and the README table of refused options; serve.sh's
    allowlist does not contain it. prepare_server_args() merges a YAML file only for a
    literal '--config' token (server_args.py:6530). The patch runs custom logit processors
    only when sampling_info.has_custom_logit_processor is true: on the first token of a
    round through SGLang's own sampler, and on the verify rows in selfspec_fast_commit.py
    process_verify_logits(), which calls selfspec_sampling.py apply_custom_rows();
    sampling_batch_info.py sets that flag only when the server flag is on. Keys: upstream
    installs add_api_key_middleware only when tokenizer_worker_num == 1
    (http_server.py:1994-2022), and init_multi_tokenizer() asserts that api_key is None
    (:223-226). The patch replaces both: install_auth_middleware() (utils/auth.py) puts the
    check on the module-level app that every uvicorn worker serves, the keys reach each
    worker inside its pickled start target (entrypoints/http_worker_keys.py), and a worker
    without keys answers 503.
  follow_up: Re-check each SGLang release for a fix to the dill.loads() sink.
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2026-536
  aliases: [GHSA-gwv6-pq6m-p3rq, CVE-2026-7301]
  component: sglang
  status: not_affected
  reason: >-
    pickle.loads() on the ZMQ ROUTER socket of the multimodal_gen (diffusion image/video)
    scheduler. That runtime is never started. serve.sh runs python -m
    sglang.launch_server, which dispatches only into sglang.srt, and srt's own scheduler
    IPC uses ipc:// unix sockets. The `sglang serve` CLI can route diffusion models to
    multimodal_gen (cli/serve.py:103-108); serve.sh does not use that CLI. The
    sglang[diffusion] extras (diffusers and friends) are not in the lock.
  evidence: >-
    launch_server.py:8-47 imports only sglang.srt.* entry points. grep for
    'multimodal_gen' under python/sglang/srt with the patch applied returns 0 matches. The
    sink is multimodal_gen/runtime/managers/scheduler.py:274. srt IPC names are ipc://
    unless --enable-dp-attention is set (server_args.py:6578-6595); with it, PortArgs
    moves them to tcp:// (:6596 onward), and srt reads those sockets with recv_pyobj too.
    serve.sh's EXTRA_ARGS allowlist accepts neither --enable-dp-attention nor
    --dist-init-addr, --nnodes or --node-rank. diffusers is absent from
    inference/gravityocr/sglang/requirements-lock.txt.
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2026-538
  aliases: [GHSA-qwrp-wghp-94q2, CVE-2026-7302]
  component: sglang
  status: not_affected
  reason: >-
    Path traversal through the multipart filename on multimodal_gen's /v1/images/edits
    and /v1/videos endpoints (image_api.py:241-244, video_api.py:121-122 and
    utils.py:148-153 in multimodal_gen/runtime/entrypoints/openai/). The srt HTTP server
    that serve.sh starts does not register those routes. Its only UploadFile route,
    /v1/audio/transcriptions, reads the upload into memory and never writes a
    client-named path. Fixed upstream in v0.5.22 (sgl-project/sglang#17270).
  evidence: >-
    grep for 'images/edits' and '/v1/videos' under python/sglang/srt with the patch
    applied returns 0 matches. The only srt UploadFile use is
    srt/entrypoints/http_server.py:1458-1481 (file.read(), no path join).
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2026-3695
  aliases: [GHSA-jrcc-j37m-v8fg, CVE-2026-10775]
  component: sglang
  status: affected
  reason: >-
    LOW (local, high complexity). The multimodal cache key truncates SHA-256 to 64 bits,
    and data_hash() of a tuple raises TypeError for non-tensor list features
    (srt/managers/mm_utils.py:1175-1228). GLM-OCR is multimodal, so this hashing is on our
    request path, and a client can submit a precomputed_embedding dict
    (srt/multimodal/processors/base_processor.py:1116-1126). The impact is a failed
    request or a cache-key collision between requests. No upstream fix.
  mitigation: >-
    serve.sh binds 127.0.0.1 by default, for trusted local clients. A non-loopback
    SERVE_HOST is refused unless API_KEY is set, and the patch makes every tokenizer worker
    check the key (upstream v0.5.10 checks keys only with one worker), so only key holders
    can send requests. EXTRA_ARGS cannot switch to a front-end that skips the key check
    (--grpc-mode, --encoder-only, --use-ray are not on its allowlist).
    Do not expose the port to untrusted callers, with or without a key: the same callers
    could also make the server fetch http(s) image URLs and read local image paths
    (srt/utils/common.py:827-837). ocr_request.py sends images as data: URLs only.
  evidence: >-
    data_hash() and hash_feature() at mm_utils.py:1175-1228 are the functions named in the
    advisory (sgl-project/sglang#25462). The patch does not modify mm_utils.py.
  follow_up: Re-check each SGLang release for a fix.
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2026-3064
  aliases: [GHSA-m2jr-x4gq-5rmj, CVE-2026-10300]
  component: sglang
  status: not_affected
  reason: >-
    A reachable assert in LoRAManager when LoRA is enabled (sgl-project/sglang#23141).
    serve.sh enables no LoRA. With LoRA disabled, a request carrying lora_path is rejected
    before the scheduler, and /load_lora_adapter refuses to load one.
  evidence: >-
    tokenizer_manager.py:2233-2252 raises ValueError when enable_lora is false;
    tokenizer_communicator_mixin.py:647-650 does the same for /load_lora_adapter. LoRA is
    switched on only by --enable-lora or --lora-paths (server_args.py:6241-6244), and
    serve.sh's EXTRA_ARGS allowlist has no LoRA option and not --config. In
    inference/gravityocr/, the strings 'enable-lora' and 'lora-paths' appear only in the
    launcher test's refused inputs and the README table of refused options.
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2026-2286
  aliases: [GHSA-63cw-57p8-fm3p, CVE-2026-24747, PYSEC-2026-1856]
  component: torch
  status: affected
  reason: >-
    The weights_only unpickler can corrupt memory on a crafted .pth file. Fixed in torch
    2.10.0, so the root and zoro-1 environments (2.11.0) are not affected. The
    inference/gravityocr/ server environment cannot take the fix: SGLang v0.5.10 pins
    torch==2.9.1 (python/pyproject.toml:66), and sglang-kernel 0.4.1, flashinfer-python
    0.6.7.post2 and the +cu128 torchvision and torchaudio wheels are built against it.
  mitigation: >-
    SGLang does not unpickle a checkpoint that ships safetensors. With the default load
    format it globs *.safetensors before *.bin and stops at the first pattern that matches
    (srt/model_loader/loader.py:396-397,441-446). trillionlabs/GravityOCR ships
    model.safetensors only, and serve.sh's EXTRA_ARGS cannot set --load-format. Serve
    checkpoints only from trusted repositories, in safetensors. A request can reach the
    .bin/.pt loader too: POST /update_weights_from_disk names a model_path and load_format.
    On every start, whatever the number of tokenizer workers, serve.sh gives SGLang a fresh
    random admin key that it never prints, and the patch makes every tokenizer worker
    enforce it, so that endpoint, like every other management endpoint, refuses every
    caller, API_KEY holders included. The admin key cannot be read back: the patch blanks
    api_key and admin_api_key in /server_info and /get_server_info (upstream v0.5.10
    returns them to any caller that passes the API-key check) and keeps them out of the
    logged server arguments, and serve.sh hands both keys to SGLang through a FIFO, never
    through a file, a command line or the environment.
  evidence: >-
    The Hugging Face API lists model.safetensors and no .bin, .pt or .pth file for
    trillionlabs/GravityOCR @ bdbe9ab. The other torch.load() calls in srt are the
    .bin/.pt fallbacks (model_loader/weight_utils.py:108,684,887,894), debug dump readers,
    CI-only .bin validation, EPLB expert-location files, the EAGLE token map
    (speculative/spec_utils.py:715) and models GLM-OCR does not load. The request path:
    entrypoints/http_server.py:938-939 registers /update_weights_from_disk as
    ADMIN_OPTIONAL, which utils/auth.py decide_request_auth() allows with no key configured
    and restricts to the admin key once one is (:128-138).
    managers/io_struct.py:1224-1228 carries model_path and load_format,
    model_executor/model_runner.py:1351-1354 builds the loader from them, and the .bin/.pt
    iterator reads files with _load_pt_file() (model_loader/weight_utils.py:878-894):
    torch.load(weights_only=True), retried with weights_only=False for a legacy .tar file.
    Upstream enforces keys only when tokenizer_worker_num == 1 (http_server.py:1994-2022)
    and returns them from server_info() (:601-618); the patch's install_auth_middleware()
    (utils/auth.py) enforces them in every worker, and redact_server_args() blanks them in
    server_info(). Checked on an H100 with the patched build: every ADMIN_OPTIONAL route,
    /update_weights_from_disk included, answered 401 with no key, a wrong key and the API
    key, with one and with two tokenizer workers.
  follow_up: Lifts with an SGLang rebase onto a release that requires torch 2.10 or later.
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2025-195
  aliases: [GHSA-qfhq-4f3w-5fph, CVE-2025-3001]
  component: torch
  status: not_affected
  reason: >-
    Memory corruption in torch.lstm_cell, local only. Reported against the
    inference/gravityocr/ server environment's torch 2.9.1; OSV does not flag the root and
    zoro-1 pin (2.11.0). Nothing in this repository or in the patched SGLang calls
    lstm_cell, and GLM-OCR has no LSTM.
  evidence: >-
    grep for 'lstm_cell' returns 0 matches in this repository outside this file, and 0
    matches over python/sglang with the patch applied.
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2026-139
  aliases: [CVE-2026-4538]
  component: torch
  status: not_affected
  reason: >-
    Deserialization in torch's pt2 loading handler (torch.export and AOTInductor
    packages), local only. OSV reports it against the inference/gravityocr/ server
    environment's torch 2.9.1. Its range ends at last_affected 2.10.0, so the root and
    zoro-1 pin (2.11.0) is not flagged today, but no upstream fix is recorded. This entry
    therefore covers every environment in the repository, root and zoro-1 included:
    nothing here loads a pt2 archive.
  evidence: >-
    grep for 'torch.export', 'aoti_load_package', 'load_package(', 'aot_load' and '.pt2'
    returns 0 matches in train/, inference/ (zoro-1 and gravityocr), benchmark/, tools/,
    scripts/ and train.py. Over python/sglang with the patch applied it returns one: the
    ModelOpt export import at srt/model_loader/loader.py:2742 (from modelopt.torch.export
    import export_hf_checkpoint), on an optional export path whose nvidia-modelopt package
    is not in the lock; it writes a checkpoint and loads no pt2 archive. The
    repository's own torch.load() calls read jev_head.pt pointer-head weights that the
    trainer writes (train/train.py:231, inference/zoro-1/decide_engine.py:118,
    scripts/export_kev_checkpoint.py:32, train/tools/export_kev_checkpoint.py:28), and
    train/tools/jev_parity_trida.py:25 a test head the same tool writes; none are pt2 archives.
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2026-4174
  aliases: [GHSA-x9r9-c232-4q39, CVE-2026-80047]
  component: transformers
  status: affected
  reason: >-
    load_custom_generate() fetches a repository's custom_generate/generate.py into the
    Hugging Face modules cache before the trust_remote_code check. The code only runs
    after that check passes, but the file is written either way. Affected from 4.49.0 to
    5.8.1, with no fixed release recorded. This entry covers every environment in the
    repository, root and zoro-1 included, not only inference/gravityocr/. Root and zoro-1
    resolve transformers>=4.57,<5, where from_pretrained() of any generative model calls
    load_custom_generate() on the repository it loads, whatever trust_remote_code is
    (transformers 4.57.6 modeling_utils.py:5079,5103-5111). In the inference/gravityocr/
    server environment (transformers 5.3.0), from_pretrained() calls it only when
    trust_remote_code is true (generation/utils.py:423); serve.sh does not pass
    --trust-remote-code, and SGLang loads GLM-OCR with its own model class and never calls
    transformers' generate().
  mitigation: >-
    Load models only from trusted repositories (trillionlabs/* and vetted upstreams such
    as Qwen/*), as for PYSEC-2026-2289. The early write lands in the local modules cache,
    and executing it still needs trust_remote_code. The repository passes it when a user
    asks for it (train --trust_remote_code), and hard-codes it only in internal tools that
    load our own Trida checkpoints: benchmark/serving/serve_trida_openai.py:210-212, the
    draft-alignment tools train/tools/draftalign/longhorizon.py:35-39 and slot_diag.py:44-48
    (both through HFBlockDiffusionHybrid) and gen_selfdistill.py:21 (a tokenizer), and the
    zoro-1 serving launchers (inference/zoro-1/serve.py:173 and the vllm/ scripts), which
    pass --trust-remote-code to SGLang or vLLM. Nothing here calls
    generate(custom_generate=...).
  evidence: >-
    grep for 'custom_generate' returns 0 matches in this repository outside the docs/
    records (this file and docs/oss/intake/0008-transformers.md), and 0 matches over
    python/sglang with the patch applied. Generative from_pretrained() call sites:
    train/hf_block_diffusion.py:122 (HFBlockDiffusionHybrid inherits it, so it also covers
    train/tools/draftalign/longhorizon.py and slot_diag.py),
    benchmark/serving/serve_qwen3_ar_openai.py:58, serve_qwen3_bd_openai.py:89 and
    serve_trida_openai.py:211. inference/zoro-1/ calls from_pretrained() only for
    tokenizers and image processors, which do not reach load_custom_generate().
    trillionlabs/GravityOCR @ bdbe9ab ships no custom_generate/ directory and no .py
    file, and its config.json, processor_config.json and tokenizer_config.json have no
    auto_map.
  follow_up: >-
    Lifts when each environment reaches a fixed transformers release. For root and
    zoro-1 that is the same 5.x-compatibility task as PYSEC-2026-2289; for
    inference/gravityocr/ it is an SGLang rebase.
  approver: Trillion Labs research team
  expires: 2026-12-31

- id: PYSEC-2026-2447
  aliases: [GHSA-w8v5-vhqr-4h9v, CVE-2025-69872]
  component: diskcache
  status: not_affected
  reason: >-
    Pickle deserialization from a cache directory an attacker can write to. Not fixed in
    any release. In the inference/gravityocr/ server environment, diskcache is reached
    only through outlines (pinned ==0.1.11 by SGLang v0.5.10), which SGLang imports only
    for --grammar-backend outlines. The default backend is xgrammar, and serve.sh sets no
    grammar backend. diskcache is not in the root or zoro-1 environments.
  evidence: >-
    srt/constrained/base_grammar_backend.py:217-218 imports outlines only when the backend
    name is "outlines". server_args.py:2563-2565 defaults it to xgrammar. grep for
    'diskcache' over python/sglang with the patch applied returns 0 matches.
  note: >-
    serve.sh's EXTRA_ARGS allowlist does not accept --grammar-backend. Anyone who starts
    the server with --grammar-backend outlines by other means must keep outlines' cache
    directory unwritable by other users, and this entry has to be re-read.
  approver: Trillion Labs research team
  expires: 2026-12-31
```
