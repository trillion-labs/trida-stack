# trida-stack

[English](README.md) | **한국어**

**자기회귀(AR) LLM을 블록 디퓨전(block diffusion) LLM으로 전환하고 서빙하기 위한 학습·추론 도구를 제공합니다.**

[![Zoro-1](https://img.shields.io/badge/%F0%9F%A4%97%20Zoro--1-diffusion%20%2B%20decisions-yellow)](https://huggingface.co/trillionlabs/Zoro-1)
[![Trida2.0-4B](https://img.shields.io/badge/%F0%9F%A4%97%20Trida2.0--4B-block%20diffusion-yellow)](https://huggingface.co/trillionlabs/Trida2.0-4B)
[![Trida-2.0-4B-1006](https://img.shields.io/badge/%F0%9F%A4%97%20Trida--2.0--4B--1006-vision--language-yellow)](https://huggingface.co/trillionlabs/Trida-2.0-4B-1006)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

학습과 추론을 위한 두 개의 독립적인 스택으로 구성되어 있습니다.

| | 스택 | 주요 기능 |
|---|---|---|
| 🏋️ | **`train/`** | **AR → 디퓨전 전환.** 별도의 디퓨전 사전학습 없이 *사전학습된 AR 체크포인트*에 블록 디퓨전 지도 미세조정(SFT)을 바로 적용합니다. AR 보조 손실을 함께 사용해 동일한 가중치로 자기회귀 생성도 수행할 수 있도록 학습합니다. 선택적으로 **유형별 의사결정(typed decision) 공동 학습**(예/아니오·선택·점수형 질문에 답하는 포인터 헤드)과 비전-언어 체크포인트의 **이미지 입력 학습**을 지원합니다. |
| ⚡ | **`inference/<model>/`** | **nano-inference: nanoGPT 스타일의 간결한 모델별 서빙 스택.** [`inference/zoro-1/`](inference/zoro-1/)은 `trillionlabs/Zoro-1`을 서빙합니다. 하나의 체크포인트로 self-speculative 디코딩과 유형별 의사결정을 지원합니다. **SGLang**, **vLLM**, **MLX** 백엔드에서 OpenAI 호환 API를 제공하며, [스피드런 리더보드](inference/zoro-1/README.md#speedrun)에서 성능 기록을 확인할 수 있습니다. |

`train/`과 `inference/` 사이에는 상호 임포트 의존성이 없습니다. `benchmark/`는 에이전트 평가를 위한 종합 평가 도구를,
`inference/`는 품질과 처리량을 간단히 측정할 수 있는 벤치마크(gsm8k / mmlu_pro / ifeval)를 제공합니다.

---

## 모델

이 저장소의 도구로 학습하고 Hugging Face에 공개한 체크포인트입니다.

| 모델 | 설명 | 서빙 방법 |
|---|---|---|
| [![Zoro-1](https://img.shields.io/badge/%F0%9F%A4%97-trillionlabs%2FZoro--1-yellow)](https://huggingface.co/trillionlabs/Zoro-1) | Qwen3.5-4B 기반의 4B 투스트림 블록 디퓨전 모델입니다. 에이전트 작업과 **의사결정 작업**을 함께 학습했습니다(`jev_head.pt` 포인터 헤드). 접근이 제한된 모델이므로 접근 권한을 요청한 뒤 `hf auth login`으로 인증해야 합니다. | [`inference/zoro-1/`](inference/zoro-1/): `--mode self-spec`(기본) / `causal` / `decide` |
| [![Trida2.0-4B](https://img.shields.io/badge/%F0%9F%A4%97-trillionlabs%2FTrida2.0--4B-yellow)](https://huggingface.co/trillionlabs/Trida2.0-4B) | Qwen3.5-4B 기반의 4B 블록 디퓨전 모델로, 하이브리드 어텐션을 사용합니다. 자기회귀 생성 기능을 유지해 무손실 self-speculative 디코딩을 지원합니다. Apache-2.0 라이선스로 배포됩니다. | [`inference/zoro-1/mlx/`](inference/zoro-1/mlx/) (Apple Silicon), [`inference/zoro-1/vllm/`](inference/zoro-1/vllm/) |
| [![Trida-2.0-4B-1006](https://img.shields.io/badge/%F0%9F%A4%97-trillionlabs%2FTrida--2.0--4B--1006-yellow)](https://huggingface.co/trillionlabs/Trida-2.0-4B-1006) | Qwen3.5 비전 타워와 프로젝터를 결합한 비전-언어 모델입니다. | [`inference/zoro-1/mlx/`](inference/zoro-1/mlx/) (이미지 입력) |

[`benchmark/`](benchmark/)의 에이전트 평가 도구는
[`trillionlabs/Trida-7B-Preview`](https://huggingface.co/trillionlabs/Trida-7B-Preview) 모델로도 실행할 수 있습니다.

---

## 지원하는 모델 계열

학습 코드는 기본적으로 특정 아키텍처에 종속되지 않습니다. `model.get_decoder()`로 디코더에 접근할 수 있는 Hugging Face(HF) 자기회귀 언어 모델을 대상으로 합니다.
현재 블록 디퓨전 마스킹과 캐시 기반 디코딩을 검증한 모델 계열은 다음 두 가지입니다.

| 계열 | 어텐션 | 비고 |
|---|---|---|
| **Qwen3** | **전체 어텐션**(full attention, dense) | 기본 구현: dense 블록 마스크와 일반 KV 캐시 사용 |
| **Qwen3.5** | **하이브리드**: gated-delta *선형 어텐션* 층과 주기적으로 배치된 *전체 어텐션* 층 | 하이브리드 구조에 맞는 마스킹과 캐시 관리가 필요합니다. 선형 어텐션 층에는 길이 축을 따라 잘라낼 수 있는 KV 캐시가 없으므로, 추측 디코딩 중 롤백할 때 conv/recurrent **상태의 스냅샷을 저장하고 복원**합니다 |

Zoro-1은 하이브리드(Qwen3.5) 구조의 **투스트림** 모델이며, `inference/zoro-1/`에서 서빙할 수 있습니다
(`--mode causal / self-spec / decide`).

---

## 변환 방식

Fast-dLLM v2 방식의 블록 디퓨전 SFT를 사용합니다. 각 학습 시퀀스는 프롬프트 뒤에 노이즈를 적용한 응답과 원본 응답을 이어 붙여 구성합니다.

```
[ S (prompt) | x_t (response, noised) | x_0 (response, clean) ]
```

각 부분의 어텐션 범위는 전용 마스크로 제어합니다.

- **S (프롬프트)**: 프롬프트 내부에는 인과적(causal) 어텐션을 적용합니다. 뒤에 오는 모든 위치에서 프롬프트를 참조할 수 있으며, 프롬프트는 생성 조건으로 사용됩니다.
- **x_t (노이즈를 적용한 응답의 k번째 블록)**: 블록 *내부에는* 양방향 어텐션을 적용하고, 프롬프트와 *이전* 응답 블록(x_0의 원본 사본)을 참조합니다.
  따라서 블록 **간에는** 자기회귀 방식으로, 블록 **내에서는** 디퓨전 방식으로 생성합니다.
- **x_0 (원본 응답)**: 블록 단위의 인과적 어텐션을 적용합니다. AR 보조 손실을 사용하면 토큰 단위의 인과적 어텐션을 적용합니다.

디퓨전 손실은 x_t의 마스킹된 위치에 대한 교차엔트로피로 계산합니다. 선택적으로 x_0의 다음 토큰 예측 교차엔트로피를 AR 보조 손실로 추가하며,
최종 손실은 `(diff + w·ar) / (1 + w)`로 계산합니다. Fast-dLLM v2의 **상보적 마스킹(complementary masking)**을 사용하므로, 서로 짝을 이루는 두 마스킹 뷰를 합치면 모든 토큰이 한 번씩 학습 대상에 포함됩니다.

주요 설정은 다음 두 가지입니다.

- **`--bd_size`**: 블록 길이(예: 32)를 지정합니다. 디코딩 시 한 번에 하나의 블록을 채웁니다.
- **`--ar_loss_weight`** (w): AR 보조 손실의 가중치입니다. `> 0`으로 설정하면 **자기회귀 생성 기능(causal head)**을 함께 학습하므로,
  동일한 체크포인트로 추측 디코딩과 일반 AR 서빙을 수행할 수 있습니다.

`--within_block_causal`을 사용하면 x_t의 블록 내부에도 토큰 단위의 인과적 어텐션을 적용합니다. x_0에도 토큰 단위의 인과적 어텐션을 적용하면
모델 전체가 *완전한 인과적 구조*를 따르므로 일반 자기회귀 서빙 엔진에서도 사용할 수 있습니다. 다만 초안(draft)의 품질은 다소 낮아집니다.

### 유형별 의사결정 (공동 학습)

[![Zoro-1](https://img.shields.io/badge/%F0%9F%A4%97-decision%20model%3A%20Zoro--1-yellow)](https://huggingface.co/trillionlabs/Zoro-1)

동일한 모델 가중치로 **의사결정 작업**도 함께 학습할 수 있습니다. *상태*(임의의 JSON과 선택적 이미지)와 응답 유형이 지정된 *질문*을 입력하면,
질문 유형에 따라 보정된 확률이나 수준의 기댓값을 반환합니다.

| 유형 | 질문 형식 | 반환값 |
|---|---|---|
| `noul` | 예/아니오 | P(예) |
| `choice` | `{키: 설명}` | 각 키에 대한 확률 분포 |
| `score` | 낮은 수준부터 순서대로 나열한 목록 | 수준의 기댓값 |

`--jev_data <kev 형식 JSONL>`을 지정하면 [kev](https://github.com/jaredpalmer/kev) 방식의 의사결정 목적 함수를
추가합니다(`train/jev.py`). 노이즈가 없는 인과적 스트림에 연결된 작은 **포인터 헤드**가 각 선택지의 `</opt>` 상태를
질문의 `<decide>` 상태와 비교해 점수를 계산합니다. 각 옵티마이저 스텝에서는 일반 SFT 마이크로배치를 처리한 뒤
`--jev_micro_per_step`개의 의사결정 마이크로배치를 처리하고, 해당 손실에 `--jev_loss_weight`를 곱해 더합니다.
각 체크포인트에는 `jev_head.pt`가 함께 저장됩니다. `train/tools/export_kev_checkpoint.py`를 사용하면 kev 디렉터리 구조로 내보낼 수 있습니다.
자세한 내용: [`train/README.md`](train/README.md#typed-decision-training-jevpy).

### 이미지 입력 (비전-언어 체크포인트)

`vision_config`가 있는 Qwen3.5 체크포인트는 이미지가 포함된 대화(`{"type": "image", "image": "imgs/0001.png"}`)를
텍스트 데이터와 함께 학습할 수 있습니다. 비전 인코더의 가중치는 고정하고, 프로젝터에는 `--projector_lr_scale` × `--lr`의 학습률을 적용합니다.
`--max_image_pixels`를 초과하는 이미지는 크기를 줄여 사용합니다. 텍스트 전용 체크포인트에는
`train/tools/compose_vl_checkpoint.py`를 사용해 기반 모델의 비전 타워를 결합할 수 있습니다.
자세한 내용: [`train/README.md`](train/README.md#images-vision-language-checkpoints).

---

## 빠른 시작

### 튜토리얼

| 노트북 | 다루는 내용 | 모델 |
|---|---|---|
| [`train/getting_started.ipynb`](train/getting_started.ipynb) | 데이터 형식과 최소 블록 디퓨전 SFT 실행(GPU 1장, 2스텝) | 사용자가 선택한 AR 기반 모델(예: Qwen3.5-4B) |
| [`inference/zoro-1/getting_started.ipynb`](inference/zoro-1/getting_started.ipynb) | 모델 서빙, API 호출, 벤치마크 실행 및 결과 해석 | [![Zoro-1](https://img.shields.io/badge/%F0%9F%A4%97-Zoro--1-yellow)](https://huggingface.co/trillionlabs/Zoro-1) |

### 학습 (AR → 디퓨전 전환)

> Qwen3.5 투스트림(하이브리드) 모델을 학습하려면 먼저 커널을 다운로드해야 합니다:
> `bash train/block_gated_delta_rule/fetch_kernels.sh` ([설치](#설치) 참고).

```bash
PYTHONPATH=. torchrun --nproc_per_node=8 train.py \
  --model_id Qwen/Qwen3-4B \
  --bd_size 32 --ar_loss_weight 0.2 \
  --max_length 32768 --pack \
  --save_dir checkpoints/my-bd-run
```

DeepSpeed 없이 PyTorch의 **FSDP2**로 실행됩니다. 패킹, 길이·응답 기준 버킷팅, 멀티턴 지도 학습, 활성값 오프로딩,
fused CE 등 전체 옵션은 `python train.py --help`에서 확인할 수 있습니다.

학습 명령에 `--jev_data <kev 형식 JSONL 파일/디렉터리>`를 추가하면 의사결정 작업도 함께 학습합니다
([유형별 의사결정](#유형별-의사결정-공동-학습) 참고). `--jev_only`를 사용하면 SFT 데이터와 디퓨전 손실 없이 의사결정 목적 함수만으로 학습합니다.

체크포인트는 **일반 HF 자기회귀 언어 모델**에 `<|mask|>` 토큰과 별도 메타데이터 파일인 `block_diffusion.json`
(`bd_size`, `mask_id`, `ar_loss_weight`)을 추가한 형태로 저장됩니다. `AutoModelForCausalLM.from_pretrained`로 바로 불러올 수 있습니다.

### 추론 (모델별 서빙 스택 `nano-inference`)

[![Zoro-1](https://img.shields.io/badge/%F0%9F%A4%97-trillionlabs%2FZoro--1-yellow)](https://huggingface.co/trillionlabs/Zoro-1)

nanoGPT처럼 작고 읽기 쉬운 서빙 스택을 모델별로 `inference/<model>/` 아래에 구성합니다.
[`inference/zoro-1/`](inference/zoro-1/)은 **[`trillionlabs/Zoro-1`](https://huggingface.co/trillionlabs/Zoro-1)**을 서빙합니다
(접근 권한을 요청한 뒤 `hf auth login`으로 인증해야 합니다).

```bash
python inference/zoro-1/serve.py                  --port 30000   # 기본값: self-spec (디퓨전 초안 -> AR 검증, b15_g8)
python inference/zoro-1/serve.py --mode causal    --port 30000   # 일반 AR
python inference/zoro-1/serve.py --mode decide    --port 30001   # 유형별 의사결정 + 채팅 (stock SGLang)
```

Self-speculative 디코딩은 동일한 샘플링 설정에서 **무손실**로 동작합니다. 확정되는 토큰은 AR 검증을 통과한 초안 토큰이거나 AR이 제시한 수정 토큰입니다.
이를 통해 AR 생성 결과의 분포를 유지하면서 **순전파 한 번에 생성하는 토큰 수**를 늘립니다. 채팅, 의사결정, 벤치마크, 체크포인트 요구사항,
**스피드런 리더보드**(H100, L40S, MLX)에 관한 자세한 내용은 [`inference/zoro-1/README.md`](inference/zoro-1/README.md)를 참고하세요.

### 의사결정 서빙 (`--mode decide`)

`--mode decide`는 **기본 SGLang 배포판(≥ 0.5.22)**에서 `serve_decide.py`를 실행합니다. 별도의 포크나 패치가 필요하지 않습니다.
단일 엔진과 동일한 모델 가중치로 `POST /v1/decide`를 통한 유형별 의사결정, 추론 후 의사결정(`"reasoning": true`),
도구 호출이 가능한 OpenAI 호환 채팅을 모두 지원합니다.

```bash
curl -s localhost:30001/v1/decide -H 'content-type: application/json' -d '{
  "request": {
    "state": {"ticket": "Order #W1 arrived broken. I want my money back."},
    "questions": {
      "intent": {"type": "choice", "instructions": "What does the customer want?",
                 "criteria": {"refund": "money back", "exchange": "a replacement", "info": "just information"}},
      "urgent": {"type": "noul", "instructions": "Should this jump the queue?"}}},
  "reasoning": false}'
# -> {"answers": {"intent": {"type": "choice", "choice": "refund", "probabilities": {...}},
#                 "urgent": {"type": "noul", "noul": 0.93}}, ...}
```
자세한 내용: [`inference/zoro-1/README.md#decisions`](inference/zoro-1/README.md#decisions).

### 백엔드

| 백엔드 | 위치 | 내용 |
|---|---|---|
| **SGLang** (패치) | [`inference/zoro-1/sglang/`](inference/zoro-1/sglang/) | 블록 내 양방향 어텐션과 가변 길이 토큰 확정을 지원하는 블록 디퓨전 및 self-speculative 디코딩 |
| **SGLang** (기본 배포판) | `inference/zoro-1/serve_decide.py` | `--mode decide`: 유형별 의사결정 + 채팅 |
| **vLLM** | [`inference/zoro-1/vllm/`](inference/zoro-1/vllm/) | 외부 플러그인(out-of-tree) 방식의 self-spec 디코딩 |
| **MLX** | [`inference/zoro-1/mlx/`](inference/zoro-1/mlx/) | Apple Silicon 기기 내 실행: 무손실 self-spec 디코딩, OpenAI 호환 서버, 이미지 입력, 경량 에이전트. q8 모델은 메모리 16–18 GB의 Mac에서 실행할 수 있습니다 |

### 스피드런

[NanoGPT 스피드런](https://github.com/KellerJordan/modded-nanogpt) 방식의 리더보드입니다. Zoro-1으로 정해진 GSM8K 표본의 요청을 하나씩 처리하면서,
정확도 87.0% 이상을 유지하고 전체 처리 시간을 최소화하는 것이 목표입니다. H100 1대, L40S 1대, Apple Silicon(MLX)의 세 트랙으로 운영합니다.
규칙과 기록: [`inference/zoro-1/README.md#speedrun`](inference/zoro-1/README.md#speedrun).

---

## 디렉터리 구조

```text
train/                         AR → 디퓨전 SFT 학습 코드 (진입점: train.py → train.train:main)
  README.md                    실행·학습 재개 방법, 모델 계열별 유의사항
  hf_block_diffusion.py        블록 마스크, 손실 함수, HF 모델 래퍼 (dense / Qwen3)
  hf_block_diffusion_hybrid.py  하이브리드 구현 (Qwen3.5: gated-delta 선형 어텐션 + 전체 어텐션)
  jev.py                       유형별 의사결정 목적 함수 (포인터 헤드, kev 방식)
  tools/                       체크포인트 내보내기·비전 타워 결합·동등성 및 FSDP 검증 도구
  getting_started.ipynb        학습 튜토리얼
  block_gated_delta_rule/      블록 단위 인과적 gated-delta Triton 커널 다운로드 절차
                               (업스트림: PolyForm-NC, 코드 미포함 — fetch_kernels.sh 실행 필요)
  data/text_sft_data.py        패킹 / 버킷팅 / 멀티턴 데이터 처리
inference/                     nano-inference: 모델별로 구성된 간결한 서빙 스택
  zoro-1/                      trillionlabs/Zoro-1
    README.md                  서빙, 의사결정, 벤치마크, 스피드런 리더보드
    getting_started.ipynb      서빙 튜토리얼
    MODEL_CONTRACT.md          서빙 코드가 요구하는 체크포인트 규격
    serve.py                   서버 실행 (--mode causal | self-spec | decide)
    serve_decide.py, decide_*.py, jev*.py
                               기본 SGLang 배포판에서 실행하는 의사결정 서버 (/v1/decide)
    chat.py                    OpenAI 호환 클라이언트
    eval.py / eval_ifeval.py    gsm8k / mmlu_pro / ifeval 품질·처리량 평가
    speedrun/                  리더보드 규칙, 하드웨어 트랙별 기록, 차트
    ifeval_lib/                저장소에 포함된 IFEval 채점 코드
    configs/                   self-spec 디코딩 설정 (b15_g8)
    sglang/                    패치된 SGLang 백엔드 설치 절차
    vllm/                      외부 vLLM 플러그인 (self-spec)
    mlx/                       Apple Silicon용 온디바이스 백엔드
                               (MLX self-spec 디코더, OpenAI 호환 서버, 에이전트)
benchmark/                     종합 에이전트 평가 도구 (bfcl_v4, tau2, functionchat, ko_agentbench,
                               swe_bench, terminal_bench) 및 serving/
data/                          데이터셋 다운로드 스크립트와 목록
```

## 설치

```bash
pip install -r requirements.txt                     # 학습 스택
pip install -r inference/zoro-1/requirements.txt    # 서빙/평가 클라이언트
```

SGLang으로 디퓨전 모델을 서빙하려면 전용 **SGLang** 백엔드가 필요합니다. 이 백엔드는 저장소에 포함되어 있지 않으므로 별도로 설치해야 합니다
([`inference/zoro-1/README.md`](inference/zoro-1/README.md) 참고). **vLLM** 백엔드는 플러그인으로 제공됩니다.
`pip install -e inference/zoro-1/vllm`으로 설치한 뒤 `VLLM_PLUGINS=trida_diffusion`을 설정하세요 ([`inference/zoro-1/vllm/README.md`](inference/zoro-1/vllm/README.md) 참고).

### 서드파티 커널 다운로드 (최초 1회 필수)

아래 세 구성요소의 업스트림 코드는 **PolyForm Noncommercial 1.0.0** 라이선스를 따르므로, Apache-2.0으로 배포되는 이 저장소에 *포함되어 있지 않습니다*.
대신 지정된 커밋의 코드를 다운로드하고 패치하는 절차를 제공합니다. 학습이나 서빙을 시작하기 **전에** 사용할 구성요소의 설치 절차를 완료하세요. 필요한 코드를 받지 않으면 임포트 오류가 발생합니다.

| 용도 | 설치 방법 | 설치되는 구성요소 |
|---|---|---|
| Qwen3.5 투스트림(하이브리드) 경로 학습 | `bash train/block_gated_delta_rule/fetch_kernels.sh` | 투스트림 Gated-DeltaNet + ShortConv Triton 커널 ([설치 안내](train/block_gated_delta_rule/README.md)) |
| **SGLang**으로 서빙 | [`inference/zoro-1/sglang/README.md`](inference/zoro-1/sglang/README.md)의 절차에 따라 설치 | 패치된 HybridDiffusion SGLang 백엔드 (clone → patch → install → shape) |
| **vLLM**으로 서빙 | [`inference/zoro-1/vllm/vllm_native_diffusion/KERNELS.md`](inference/zoro-1/vllm/vllm_native_diffusion/KERNELS.md)의 절차에 따라 설치 | 블록 끝 readout 커널 `block_causal_readout.py` |

각 설치 절차는 지정된 업스트림 커밋(`yuchen-zhu-zyc/HybridDiffusion@6ca547a`)을 사용하고 이 프로젝트의 패치를 적용합니다.
**다운로드한 코드는 비상업 라이선스를 따르며, 이 저장소의 Apache-2.0 라이선스 적용 대상이 아닙니다.** `NOTICE`와 `COMPLIANCE.md`를 참고하세요.

## 문의 및 기여

| | |
|---|---|
| 설치·사용 질문 | [Discussions · Q&A](https://github.com/trillion-labs/trida-stack/discussions/categories/q-a) |
| 재현 가능한 버그 신고 | [Issues](https://github.com/trillion-labs/trida-stack/issues/new/choose) |
| 기능 아이디어 | [Discussions · Ideas](https://github.com/trillion-labs/trida-stack/discussions/categories/ideas) |
| 모델 리비전 관련 문제 | [Zoro-1 Community](https://huggingface.co/trillionlabs/Zoro-1/discussions) |
| 기여 안내 | [CONTRIBUTING.md](CONTRIBUTING.md) |
| 보안 취약점 | [SECURITY.md](SECURITY.md)에 안내된 **비공개 신고 경로** |

채널별 이용 기준과 응답에 관한 안내는 [COMMUNITY.md](COMMUNITY.md), 행동 규범은
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)를 참고하세요. 의사결정 권한과 결정 사항의 기록 방식은
[GOVERNANCE.md](GOVERNANCE.md)에 정리되어 있습니다.

## 라이선스

라이선스는 `LICENSE`를, 서드파티 저작권 표시는 `NOTICE`와 `COMPLIANCE.md`를 참고하세요.

## Star History

<a href="https://star-history.com/#trillion-labs/trida-stack&Date">
  <img src="https://api.star-history.com/svg?repos=trillion-labs/trida-stack&type=Date" alt="Star History Chart" width="600">
</a>
