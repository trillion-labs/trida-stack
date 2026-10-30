# trida-stack

[English](README.md) | **한국어**

**자기회귀(AR) LLM을 블록 디퓨전(block-diffusion) LLM으로 바꾸고, 그 모델을 서빙하는 저장소입니다.**

[![Zoro-1](https://img.shields.io/badge/%F0%9F%A4%97%20Zoro--1-diffusion%20%2B%20decisions-yellow)](https://huggingface.co/trillionlabs/Zoro-1)
[![Trida2.0-4B](https://img.shields.io/badge/%F0%9F%A4%97%20Trida2.0--4B-block%20diffusion-yellow)](https://huggingface.co/trillionlabs/Trida2.0-4B)
[![Trida-2.0-4B-1006](https://img.shields.io/badge/%F0%9F%A4%97%20Trida--2.0--4B--1006-vision--language-yellow)](https://huggingface.co/trillionlabs/Trida-2.0-4B-1006)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

서로 독립된 두 스택으로 이루어져 있습니다.

| | 스택 | 하는 일 |
|---|---|---|
| 🏋️ | **`train/`** | **AR → 디퓨전 변환.** 디퓨전 사전학습 없이 *사전학습된 AR 체크포인트*에서 바로 블록 디퓨전 SFT를 합니다. AR 보조 손실을 함께 걸어, 같은 가중치에 쓸 수 있는 causal head가 남습니다. 필요하면 **타입 결정(typed decision) 공동 학습**(예/아니오·선택·점수 질문에 답하는 포인터 헤드)과 비전-언어 체크포인트의 **이미지 입력** 학습도 켤 수 있습니다. |
| ⚡ | **`inference/<model>/`** | **nano-inference: 모델마다 하나씩 두는 nanoGPT 스타일의 작은 서빙 스택.** [`inference/zoro-1/`](inference/zoro-1/)은 `trillionlabs/Zoro-1`을 서빙합니다. 체크포인트 하나로 self-speculative 생성과 타입 결정을 모두 제공하고, OpenAI 호환 API로 **SGLang**, **vLLM**, **MLX** 위에서 돌아가며, [스피드런 리더보드](inference/zoro-1/README.md#speedrun)가 있습니다. |

`train/`과 `inference/`는 서로를 import하지 않습니다. `benchmark/`에는 무거운 에이전트 평가 하네스가 있고,
`inference/`에는 가벼운 품질·처리량 벤치마크(gsm8k / mmlu_pro / ifeval)가 따로 들어 있습니다.

---

## 모델

이 저장소로 학습해 Hugging Face에 공개한 체크포인트입니다.

| 모델 | 설명 | 서빙 방법 |
|---|---|---|
| [![Zoro-1](https://img.shields.io/badge/%F0%9F%A4%97-trillionlabs%2FZoro--1-yellow)](https://huggingface.co/trillionlabs/Zoro-1) | 4B 투스트림 블록 디퓨전 모델(Qwen3.5-4B 기반). 에이전트 모델이자 **결정 모델**로 함께 학습했습니다(`jev_head.pt` 포인터 헤드). 게이트 모델이라 접근 요청 후 `hf auth login`이 필요합니다. | [`inference/zoro-1/`](inference/zoro-1/): `--mode self-spec`(기본) / `causal` / `decide` |
| [![Trida2.0-4B](https://img.shields.io/badge/%F0%9F%A4%97-trillionlabs%2FTrida2.0--4B-yellow)](https://huggingface.co/trillionlabs/Trida2.0-4B) | 4B 블록 디퓨전 모델(Qwen3.5-4B, 하이브리드 어텐션). causal head가 있어 무손실 self-speculative 디코딩이 됩니다. Apache-2.0. | [`inference/zoro-1/mlx/`](inference/zoro-1/mlx/) (Apple Silicon), [`inference/zoro-1/vllm/`](inference/zoro-1/vllm/) |
| [![Trida-2.0-4B-1006](https://img.shields.io/badge/%F0%9F%A4%97-trillionlabs%2FTrida--2.0--4B--1006-yellow)](https://huggingface.co/trillionlabs/Trida-2.0-4B-1006) | 비전-언어 버전(Qwen3.5 비전 타워 + 프로젝터). | [`inference/zoro-1/mlx/`](inference/zoro-1/mlx/) (이미지 입력) |

[`benchmark/`](benchmark/)의 에이전트 벤치마크 하네스는
[`trillionlabs/Trida-7B-Preview`](https://huggingface.co/trillionlabs/Trida-7B-Preview)로도 돌릴 수 있습니다.

---

## 지원하는 모델 계열

트레이너는 원칙적으로 아키텍처에 묶여 있지 않습니다(`model.get_decoder()`로 디코더에 접근할 수 있는 HF causal LM이면 됩니다).
다만 블록 디퓨전 마스킹과 캐시 디코더를 검증한 계열은 아래 두 가지입니다.

| 계열 | 어텐션 | 비고 |
|---|---|---|
| **Qwen3** | **full attention** (dense) | 기본 경로: dense 블록 마스크, 일반 KV 캐시 |
| **Qwen3.5** | **하이브리드**: gated-delta *linear* 어텐션 층 + 주기적인 *full* 어텐션 층 | 마스킹과 캐시를 하이브리드에 맞게 다뤄야 합니다. linear 층에는 잘라낼 길이 축이 없어서, speculative 롤백을 KV crop 대신 conv/recurrent **상태 스냅샷 + 복원**으로 합니다 |

하이브리드(Qwen3.5) 경로가 Zoro-1의 **투스트림** 모델이고, `inference/zoro-1/`에서 서빙합니다
(`--mode causal / self-spec / decide`).

---

## 변환 방식

Fast-dLLM v2 방식의 블록 디퓨전 SFT입니다. 각 행에는 프롬프트 뒤에 응답의 노이즈 버전과 깨끗한 버전을 나란히 둡니다.

```
[ S (prompt) | x_t (response, noised) | x_0 (response, clean) ]
```

이 셋은 별도 어텐션 마스크로 묶입니다.

- **S (프롬프트)**: 자기 안에서는 causal이고, 뒤에 오는 모든 위치가 볼 수 있습니다(조건).
- **x_t (노이즈가 섞인 k번째 블록)**: 자기 블록 *안에서는* 양방향이고, 프롬프트와 *이전* 응답 블록(깨끗한 x_0 사본)을 봅니다.
  그래서 생성은 블록 **사이에서는** 자기회귀, 블록 **안에서는** 디퓨전입니다.
- **x_0 (깨끗한 사본)**: 자기 안에서 블록 단위 causal입니다(AR 보조 손실을 켜면 토큰 단위 causal).

손실은 노이즈 위치 x_t의 마스크 교차엔트로피(디퓨전 목적)에, 선택적으로 x_0의 다음 토큰 CE(AR 보조)를 더해
`(diff + w·ar) / (1 + w)`로 합칩니다. Fast-dLLM v2의 **상보적 마스킹(complementary masking)** 덕분에 짝지은 두 뷰에서 모든 토큰이 한 번씩 학습됩니다.

가장 중요한 손잡이는 두 개입니다.

- **`--bd_size`**: 블록 길이(예: 32). 디코딩은 한 번에 블록 하나씩 채웁니다.
- **`--ar_loss_weight`** (w): AR 보조 손실의 가중치. `> 0`이면 쓸 만한 **causal head**가 남고,
  덕분에 같은 체크포인트로 speculative 디코딩(그리고 일반 AR 서빙)이 가능합니다.

`--within_block_causal`을 켜면 x_t도 블록 안에서 토큰 단위 causal이 됩니다. x_0까지 causal이면 모델 전체가
*완전한 causal*이 되어 일반 자기회귀 서빙 엔진에서도 읽을 수 있지만, 초안(draft) 품질은 조금 떨어집니다.

### 타입 결정 (공동 학습)

[![Zoro-1](https://img.shields.io/badge/%F0%9F%A4%97-decision%20model%3A%20Zoro--1-yellow)](https://huggingface.co/trillionlabs/Zoro-1)

같은 가중치를 **결정 모델**로도 학습할 수 있습니다. *상태*(아무 JSON, 이미지 선택)와 타입이 정해진 *질문*을 주면,
자유 텍스트 대신 보정된 확률을 돌려줍니다.

| 타입 | 질문 | 답 |
|---|---|---|
| `noul` | 예/아니오 | P(예) |
| `choice` | `{키: 설명}` | 키들에 대한 확률 분포 |
| `score` | 낮은 것부터 나열한 수준 목록 | 기대 수준 |

`--jev_data <kev 형식 JSONL>`을 주면 이 목적 함수가 [kev](https://github.com/jaredpalmer/kev) 방식으로
추가됩니다(`train/jev.py`). 깨끗한 causal 스트림 위의 작은 **포인터 헤드**가 각 선택지의 `</opt>` 상태를
질문의 `<decide>` 상태와 비교해 점수를 매깁니다. 옵티마이저 스텝마다 평소의 SFT 마이크로배치를 돌린 뒤
`--jev_micro_per_step`개의 결정 마이크로배치를 돌리고, 그 손실을 `--jev_loss_weight` 가중치로 더합니다.
체크포인트마다 `jev_head.pt`가 함께 저장되고, `train/tools/export_kev_checkpoint.py`로 kev 레이아웃으로 내보낼 수 있습니다.
자세한 내용: [`train/README.md`](train/README.md#typed-decision-training-jevpy).

### 이미지 (비전-언어 체크포인트)

`vision_config`가 있는 Qwen3.5 체크포인트는 이미지가 섞인 대화(`{"type": "image", "image": "imgs/0001.png"}`)를
텍스트 데이터와 함께 학습합니다. 비전 인코더는 고정하고, 프로젝터는 `--projector_lr_scale` × `--lr`로 학습하며,
`--max_image_pixels`를 넘는 이미지는 줄여서 씁니다. 텍스트 전용 체크포인트에는
`train/tools/compose_vl_checkpoint.py`로 베이스 모델의 비전 타워를 붙일 수 있습니다.
자세한 내용: [`train/README.md`](train/README.md#images-vision-language-checkpoints).

---

## 빠른 시작

### 튜토리얼

| 노트북 | 다루는 내용 | 모델 |
|---|---|---|
| [`train/getting_started.ipynb`](train/getting_started.ipynb) | 데이터 형식과 최소 블록 디퓨전 SFT 실행(GPU 1장, 2스텝) | 직접 고른 AR 베이스(예: Qwen3.5-4B) |
| [`inference/zoro-1/getting_started.ipynb`](inference/zoro-1/getting_started.ipynb) | 서빙, 호출, 벤치마크, 결과 읽는 법 | [![Zoro-1](https://img.shields.io/badge/%F0%9F%A4%97-Zoro--1-yellow)](https://huggingface.co/trillionlabs/Zoro-1) |

### 학습 (AR → 디퓨전)
> Qwen3.5 투스트림(하이브리드) 학습은 먼저 커널을 한 번 받아 와야 합니다:
> `bash train/block_gated_delta_rule/fetch_kernels.sh` ([설치](#설치) 참고).

```bash
PYTHONPATH=. torchrun --nproc_per_node=8 train.py \
  --model_id Qwen/Qwen3-4B \
  --bd_size 32 --ar_loss_weight 0.2 \
  --max_length 32768 --pack \
  --save_dir checkpoints/my-bd-run
```
DeepSpeed 없이 순수 torch **FSDP2**로 돌아갑니다. 전체 플래그(패킹, 길이/응답 버킷팅, 멀티턴 지도, 활성값 오프로드,
fused CE 등)는 `python train.py --help`로 볼 수 있습니다.

같은 실행에 `--jev_data <kev 형식 JSONL 파일/디렉터리>`를 더하면 타입 결정 목적이 함께 학습됩니다
([타입 결정](#타입-결정-공동-학습) 참고). `--jev_only`는 결정 목적만 학습합니다(SFT 데이터와 디퓨전 손실 없이).

저장된 체크포인트는 **일반 HF causal LM**에 `<|mask|>` 토큰과 `block_diffusion.json` 사이드카
(`bd_size`, `mask_id`, `ar_loss_weight`)만 더한 형태라서, `AutoModelForCausalLM.from_pretrained`로 그대로 불러올 수 있습니다.

### 추론 (서빙: `nano-inference`, 모델마다 스택 하나)

[![Zoro-1](https://img.shields.io/badge/%F0%9F%A4%97-trillionlabs%2FZoro--1-yellow)](https://huggingface.co/trillionlabs/Zoro-1)

nanoGPT처럼, 모델마다 작고 읽기 쉬운 서빙 스택을 `inference/<model>/` 아래에 하나씩 둡니다.
[`inference/zoro-1/`](inference/zoro-1/)은 **[`trillionlabs/Zoro-1`](https://huggingface.co/trillionlabs/Zoro-1)**을 서빙합니다
(게이트 모델: 접근 요청 후 `hf auth login`).
```bash
python inference/zoro-1/serve.py                  --port 30000   # 기본값: self-spec (디퓨전 초안 -> AR 검증, b15_g8)
python inference/zoro-1/serve.py --mode causal    --port 30000   # 일반 AR
python inference/zoro-1/serve.py --mode decide    --port 30001   # 타입 결정 + 채팅 (stock SGLang)
```
Self-speculative 디코딩은 같은 샘플링 설정에서 **무손실**입니다. 확정되는 토큰은 AR 토큰과 같은 초안 토큰이거나 AR의 수정 토큰,
둘 중 하나입니다. 얻는 것은 다른 답이 아니라 **forward당 토큰 수**입니다. 채팅, 결정, 벤치마크, 모델 계약,
**스피드런 리더보드**(H100, L40S, MLX)는 [`inference/zoro-1/README.md`](inference/zoro-1/README.md)에 있습니다.

### 결정 서빙 (`--mode decide`)
`--mode decide`는 **stock SGLang ≥ 0.5.22** 위에서 `serve_decide.py`를 띄웁니다(포크나 패치 불필요).
엔진 하나, 가중치 한 벌로 `POST /v1/decide`의 타입 결정, 추론 후 결정(`"reasoning": true`),
도구 호출을 지원하는 OpenAI 채팅을 모두 제공합니다.
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
| **SGLang** (패치) | [`inference/zoro-1/sglang/`](inference/zoro-1/sglang/) | 진짜 블록 디퓨전(블록 내 양방향 어텐션, 가변 길이 확정)과 self-speculative 경로 |
| **SGLang** (stock) | `inference/zoro-1/serve_decide.py` | `--mode decide`: 타입 결정 + 채팅 |
| **vLLM** | [`inference/zoro-1/vllm/`](inference/zoro-1/vllm/) | out-of-tree 플러그인, self-spec |
| **MLX** | [`inference/zoro-1/mlx/`](inference/zoro-1/mlx/) | 온디바이스(Apple Silicon): 무손실 self-spec, OpenAI 호환 서버, 이미지 입력, 작은 에이전트. q8은 16–18 GB 맥에서 돌아갑니다 |

### 스피드런
[NanoGPT 스피드런](https://github.com/KellerJordan/modded-nanogpt) 방식의 리더보드입니다. 정해진 GSM8K 표본을 Zoro-1로,
요청 하나씩, 정확도 87.0% 이상을 지키면서 가장 빨리 푸는 것이 목표입니다. 트랙은 1× H100, 1× L40S, Apple Silicon(MLX)입니다.
규칙과 기록: [`inference/zoro-1/README.md#speedrun`](inference/zoro-1/README.md#speedrun).

---

## 디렉터리 구조

```
train/                   AR -> diffusion SFT trainer    (entry: train.py -> train.train:main)
  README.md                  launching, resume, model-family notes
  hf_block_diffusion.py      block masks, loss, HF model wrapper (dense / Qwen3)
  hf_block_diffusion_hybrid.py   hybrid path (Qwen3.5: gated-delta linear + full attention)
  jev.py                     typed-decision objective (pointer head, kev's scheme)
  tools/                     export_kev_checkpoint.py, compose_vl_checkpoint.py, parity/FSDP checks
  getting_started.ipynb      training tutorial
  block_gated_delta_rule/    fetch recipe for the block-causal gated-delta Triton kernels
                             (PolyForm-NC upstream, NOT vendored -- run fetch_kernels.sh first)
  data/text_sft_data.py      packing / bucketing / multi-turn
inference/               nano-inference: one small serving stack per model
  zoro-1/                    trillionlabs/Zoro-1
    README.md                  serving, decisions, benchmarks, speedrun leaderboard
    getting_started.ipynb      serving tutorial
    MODEL_CONTRACT.md          what the serving code expects from the checkpoint
    serve.py                   --mode causal | self-spec | decide; launches the server
    serve_decide.py, decide_*.py, jev*.py   typed-decision server (/v1/decide) on stock SGLang
    chat.py                    OpenAI-compatible client
    eval.py / eval_ifeval.py   gsm8k / mmlu_pro / ifeval quality + throughput
    speedrun/                  leaderboard rules, records per hardware track, charts
    ifeval_lib/                vendored IFEval scorers
    configs/                   self-spec decode config (b15_g8)
    sglang/                    recipe for the patched SGLang backend
    vllm/                      out-of-tree vLLM plugin (self-spec)
    mlx/                       on-device backend (Apple Silicon): MLX self-spec decoder, OpenAI server, agent
benchmark/               heavier agentic eval harnesses (bfcl_v4, tau2, functionchat, ko_agentbench,
                         swe_bench, terminal_bench) + serving/
data/                    dataset download scripts + catalog
```

## 설치
```bash
pip install -r requirements.txt                     # 학습 스택
pip install -r inference/zoro-1/requirements.txt    # 서빙/평가 클라이언트
```
서빙에는 디퓨전 서빙용 **SGLang** 백엔드가 필요하며, 이 저장소에 포함돼 있지 않으니 따로 설치해야 합니다
([`inference/zoro-1/README.md`](inference/zoro-1/README.md) 참고). **vLLM** 백엔드는 플러그인입니다:
`pip install -e inference/zoro-1/vllm` 후 `VLLM_PLUGINS=trida_diffusion` ([`inference/zoro-1/vllm/README.md`](inference/zoro-1/vllm/README.md) 참고).

### 서드파티 커널: 한 번만 받아 오기 (필수)

이 스택의 세 부분은 업스트림 라이선스가 **PolyForm Noncommercial 1.0.0**이라 Apache-2.0인 이 저장소에 *포함하지 않았습니다*.
대신 각각 커밋을 고정한 받아 오기·패치 레시피가 있습니다. 학습이나 서빙 **전에** 쓰는 경로의 레시피를 실행하세요. 실행하지 않으면 import가 실패합니다.

| 하려는 일 | 실행 | 만들어지는 것 |
|---|---|---|
| Qwen3.5 투스트림(하이브리드) 경로 학습 | `bash train/block_gated_delta_rule/fetch_kernels.sh` | 투스트림 Gated-DeltaNet + ShortConv Triton 커널 ([레시피](train/block_gated_delta_rule/README.md)) |
| **SGLang**으로 서빙 | [`inference/zoro-1/sglang/README.md`](inference/zoro-1/sglang/README.md)를 따라 하기 | 패치된 HybridDiffusion SGLang 백엔드 (clone → patch → install → shape) |
| **vLLM**으로 서빙 | [`inference/zoro-1/vllm/vllm_native_diffusion/KERNELS.md`](inference/zoro-1/vllm/vllm_native_diffusion/KERNELS.md)를 따라 하기 | 블록 끝 readout 커널 `block_causal_readout.py` |

각 레시피는 업스트림 커밋(`yuchen-zhu-zyc/HybridDiffusion@6ca547a`)을 고정하고 우리 패치를 적용합니다.
**받아 온 코드는 비상업 라이선스이며 이 저장소의 Apache-2.0 라이선스 적용 대상이 아닙니다.** `NOTICE`와 `COMPLIANCE.md`를 참고하세요.

## 도움 받기, 기여, 신고

| | |
|---|---|
| 설치·사용 질문 | [Discussions · Q&A](https://github.com/trillion-labs/trida-stack/discussions/categories/q-a) |
| 재현 가능한 결함 | [Issues](https://github.com/trillion-labs/trida-stack/issues/new/choose) |
| 기능 아이디어 | [Discussions · Ideas](https://github.com/trillion-labs/trida-stack/discussions/categories/ideas) |
| 모델 revision 문제 | [Zoro-1 Community](https://huggingface.co/trillionlabs/Zoro-1/discussions) |
| 기여하려면 | [CONTRIBUTING.md](CONTRIBUTING.md) |
| 보안 취약점 | **공개 Issue가 아니라** [SECURITY.md](SECURITY.md)의 비공개 경로 |

채널별 기준과 응답 기대치는 [COMMUNITY.md](COMMUNITY.md), 행동 규범은
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), 누가 무엇을 결정하고 그 결정이 어디에
기록되는지는 [GOVERNANCE.md](GOVERNANCE.md)에 있습니다.

## 라이선스
`LICENSE`를 보세요. 서드파티 저작권 표시는 `NOTICE`와 `COMPLIANCE.md`에 있습니다.

## Star History

<a href="https://star-history.com/#trillion-labs/trida-stack&Date">
  <img src="https://api.star-history.com/svg?repos=trillion-labs/trida-stack&type=Date" alt="Star History Chart" width="600">
</a>
