#!/bin/bash
# NOTE: published as a worked example, not as a runnable entry point. This script ran on our
# SLURM cluster and sources a private driver tree ($SCRATCH/scripts/lib/) that is not part of this
# repository; paths and partition names are ours. Read it for the methodology behind the numbers in
# the docs, and adapt rather than run. See the README in this directory.
# Long draft-align run on autopilot (user 2026-09-10: "drafter direction; if you need more steps, go for it; loop engineer it").
# Chain: gen (already running, $GEN) -> dataset v2 (v1 shards + part2) -> train 1000 steps (2 nodes, round-1 speed
# defaults, IB) -> per-checkpoint evals (vLLM self-spec N=4/8 + AR guard) -> offline per-slot diagnostic on 500/1000.
# usage: autopilot_long.sh <GEN_JOB_ID>
set -euo pipefail
GEN=${1:-none}   # "none" = no upstream dependency (data already on disk)
L=$SCRATCH; S=$L/models/scratch; D=$L/scripts/lib/draftalign
NAME=${NAME:-trida-4b-draftalign-long}; STEPS=${STEPS:-1000}; SAVE_EVERY=${SAVE_EVERY:-250}
TAG=${TAG:-long}   # eval tag prefix; MUST differ per run or eval dirs collide
PCOLD=${PCOLD:-0.5}; MWARM=${MWARM:-0}; AUF_FLOOR=${AUF_FLOOR:--1.0}   # deploy-matched masks + Spec-AUF
LR=${LR:-5e-6}; LR_MIN_RATIO=${LR_MIN_RATIO:-1.0}; WARMUP=${WARMUP:-50}; BD=${BD:-8}; AR_W=${AR_W:-1.0}; DEPJ=${DEPJ:-afterok}
V2=$S/data-draftalign-v2; mkdir -p $V2
for f in shard-n0.jsonl shard-n1.jsonl shard-n0.part2.jsonl shard-n1.part2.jsonl; do ln -sfn $S/data-draftalign-v1/$f $V2/$f; done
[ -e $S/$NAME/RUN.json ] && { echo "refusing: $S/$NAME exists"; exit 2; }
DEPFLAG=$([ "$GEN" = none ] && echo "" || echo "--dependency=$DEPJ:$GEN")
T=$(sbatch -p p0 $DEPFLAG --export=ALL,DATASET=$V2,SAVE_NAME=$NAME,MASK_PATTERN=canvas,BD=$BD,PCOLD=$PCOLD,MWARM=$MWARM,AUF_FLOOR=$AUF_FLOOR,AR_W=$AR_W,LR=$LR,LR_MIN_RATIO=$LR_MIN_RATIO,WARMUP=$WARMUP,MAX_STEPS=$STEPS,SAVE_EVERY=$SAVE_EVERY $D/train_2node.sbatch | awk '{print $4}')
echo "TRAIN=$T (afterok:$GEN)"
EV=""
for st in $(seq $SAVE_EVERY $SAVE_EVERY $STEPS); do
  r=$(DEP=afterok:$T bash $D/eval_ckpt.sh $S/$NAME/step_$st $TAG-s$st); echo "$r"; EV="$EV $(echo "$r" | grep -oE '\b[0-9]{4}\b' | tr '\n' ' ')"
done
DG=$(sbatch -p p0 --dependency=afterok:$T --export=ALL,CKPTS="$SCRATCH/trida-stack-run/checkpoints/qwen35-4b-flare-v6-2n/step_18000:step18000 $S/$NAME/step_$((STEPS/2)):${TAG}_s$((STEPS/2)) $S/$NAME/step_$STEPS:${TAG}_s$STEPS" $D/slot_diag.sbatch | awk '{print $4}')
echo "SLOTDIAG=$DG"
echo "{\"gen\":$GEN,\"train\":$T,\"evals\":\"$EV\",\"slotdiag\":$DG,\"name\":\"$NAME\",\"steps\":$STEPS,\"save_every\":$SAVE_EVERY,\"submitted\":\"$(date -Is)\"}" > $S/autopilot_${NAME}.json
cat $S/autopilot_${NAME}.json
