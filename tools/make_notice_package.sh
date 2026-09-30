#!/usr/bin/env bash
# Assemble the 납품 고지패키지 -- the bundle a downstream consumer needs in order to
# comply with every license in a trida-stack build.
#
# Layout follows the 공급망관리 guideline §4-3.
# Run from the repository root: bash tools/make_notice_package.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/trida-stack-oss-docs"
rm -rf "$OUT"
mkdir -p "$OUT/licenses"

cp "$ROOT/LICENSE"                                          "$OUT/licenses/trida-stack-Apache-2.0.txt"
cp "$ROOT/NOTICE"                                           "$OUT/THIRD_PARTY_NOTICES.txt"
cp "$ROOT/COMPLIANCE.md"                                    "$OUT/COMPLIANCE.md"
cp "$ROOT/docs/oss/ASSET_INVENTORY.csv"                     "$OUT/ASSET_INVENTORY.csv"
cp "$ROOT/train/block_gated_delta_rule/LICENSE.HybridDiffusion" \
                                                            "$OUT/licenses/HybridDiffusion-PolyForm-NC-1.0.0.txt"
cp "$ROOT/train/block_gated_delta_rule/NOTICE.HybridDiffusion" \
                                                            "$OUT/licenses/HybridDiffusion-NOTICE.txt"

# SBOMs, when the release workflow has produced them.
for f in sbom.spdx.json sbom.cyclonedx.json; do
  [ -f "$ROOT/$f" ] && cp "$ROOT/$f" "$OUT/$f"
done

cat > "$OUT/OSS_NOTICE.txt" <<'NOTICE'
trida-stack 오픈소스 고지문
Trillion Labs

본 배포물은 다음 오픈소스를 포함하거나, 빌드 시 내려받아 사용합니다.
각 컴포넌트의 라이선스 전문은 licenses/ 폴더에 있습니다.

[저장소에 포함된 컴포넌트]
 1. trida-stack (Apache-2.0) - (c) Trillion Labs
 2. instruction_following_eval / IFEval scorer (Apache-2.0)
    - (c) The Google Research Authors
    - 위치: inference/ifeval_lib/

[빌드 시 내려받는 컴포넌트 - 본 저장소에 포함되지 않음]
 3. HybridDiffusion two-stream Gated-DeltaNet kernels
    (PolyForm Noncommercial 1.0.0) - (c) yuchen-zhu-zyc
    - 고정 커밋: 6ca547a
    - 위치: train/block_gated_delta_rule/, inference/sglang/,
            inference/vllm/vllm_native_diffusion/block_causal_readout.py

*** 중요 - 상업적 이용 제한 ***

위 3번 컴포넌트는 PolyForm Noncommercial 1.0.0 라이선스이며 상업적 이용을
금지합니다. 이 제한은 해당 커널을 사용하는 모든 경로에 적용됩니다:

  - two-stream(hybrid) 학습 경로
  - block-diffusion 서빙 경로
  - self-speculative 서빙 경로

본 저장소 자체의 Apache-2.0 라이선스는 위 제한을 해소하지 않습니다.
causal(AR) 전용 경로는 위 커널에 의존하지 않으며 Apache-2.0만 적용됩니다.

[수정 여부]
 - HybridDiffusion 커널: 수정함. 수정 내용은 각 레시피의 patch 파일에 있습니다.
   (train/block_gated_delta_rule/trillion_mods.patch,
    inference/sglang/two_stream_diffusion.patch)
 - 그 외 컴포넌트: 수정 없음.

[모델 가중치]
 본 고지문은 소프트웨어에만 적용됩니다. 모델 가중치는 별도 라이선스를 따르며
 본 배포물에 포함되지 않습니다.

문의: https://github.com/trillion-labs/trida-stack/issues
NOTICE

tar -czf "$ROOT/trida-stack-oss-docs.tar.gz" -C "$ROOT" trida-stack-oss-docs
rm -rf "$OUT"
echo "wrote trida-stack-oss-docs.tar.gz"
