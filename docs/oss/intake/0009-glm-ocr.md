# GLM-OCR (base weights)

| | |
|---|---|
| 구분 | 외부유입 (model weights) |
| 라이선스 | MIT |
| 획득 경로 | Hugging Face, [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR) |
| 사용 위치 | base model for `trillionlabs/GravityOCR`, which `inference/gravityocr/` serves |
| 평가 기록일 | 2026-10-30 |
| 결정 시점 | chosen as the base when GravityOCR was trained, before this repository served it; recorded in the PR that adds `inference/gravityocr/` |
| 결정 | **채택** |
| 총점 | **82 / 100** |
| 재평가 예정 | 2027-10-30, or at the next base-model change |

## Why we needed something here

GravityOCR's research is the decoding method: converting a document-OCR model's text
decoder to block diffusion and decoding it self-speculatively. That needs a competent
autoregressive OCR model to start from, with a vision encoder that already reads
documents. Training such a model from scratch is not the research.

## Alternatives considered

None recorded for this repository. The base was fixed when GravityOCR was trained,
outside this repository, and this record is about bringing that checkpoint's base into
trida. Writing a comparison now would describe a choice we did not make here. Any future
base goes through the licence gate first, because a restriction on a base propagates to
every derived checkpoint (see [`0006-qwen35-4b.md`](0006-qwen35-4b.md)).

## Score

| 기준 | 점수 | 근거 |
|---|---|---|
| A 기능 적합성 | 19 | a document-OCR model with a vision encoder and a small text decoder, which is the shape the method needs; its architecture (`GlmOcrForConditionalGeneration`) is built into transformers 5.x and supported by upstream SGLang |
| B 라이선스 적합성 | 20 | MIT on the weights, with **no use restriction propagating to derived models**. This is why GravityOCR can be released under MIT. |
| C 유지보수 활성도 | 16 | an actively developed model family from an established lab, but a base model is a fixed artefact once chosen |
| D 보안 대응 | 15 | ships `model.safetensors` and no remote code (no `.py` files, no `auto_map`); the usual untrusted-checkpoint surface remains, so we load only from the trusted repository |
| E 통합·이탈 비용 | 12 | changing the base means retraining GravityOCR, as for any base model |
| **총점** | **82** | |

## Decision

Adopt. As with Qwen3.5-4B, the licence check on a base model differs in kind from one
on a library: a restriction on the base reaches every derived checkpoint. MIT carries
none, so GravityOCR's MIT release is consistent with its base.

Nothing in this repository loads `zai-org/GLM-OCR` itself. `inference/gravityocr/`
serves the derived `trillionlabs/GravityOCR`, and neither checkpoint is redistributed
here.

## Obligations we take on

Attribution of the base model wherever the derived weights are published: the
GravityOCR model card declares `base_model: zai-org/GLM-OCR`. The base and the derived
checkpoint are listed in [`COMPLIANCE.md`](../../../COMPLIANCE.md) §4b and in
[`NOTICE`](../../../NOTICE). MIT's condition, keeping the copyright and licence notice
with copies of the weights, applies to whoever redistributes them; we do not.

## Re-evaluation trigger

Annually; immediately if the licence on `zai-org/GLM-OCR` changes; and whenever a new
base is considered for GravityOCR. In that case the licence gate is the first question,
before any benchmark.
