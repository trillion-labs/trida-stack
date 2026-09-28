#!/bin/bash
# NOTE: published as a worked example, not as a runnable entry point. This script ran on our
# SLURM cluster and sources a private driver tree ($SCRATCH/scripts/lib/) that is not part of this
# repository; paths and partition names are ours. Read it for the methodology behind the numbers in
# the docs, and adapt rather than run. See the README in this directory.
# One-node Slurm job: several servers (one per GPU) of one engine, then the fixed-output GSM8K sweep against each.
# env: RUN_DIR JOB ENGINE=vllm|sglang CFGS="name mode block;name mode block;..." CONCS("1 4 8 16") LIMIT(64) MAXTOK(512)
#      MAXSEQS(16) SMOKE=1 -> LIMIT=10, CONCS="1", print 2 sample outputs
set -u
L=$SCRATCH; OUT=$RUN_DIR/$JOB; mkdir -p $OUT; cd $OUT
EVALPY=$L/env/HybridDiffusion/cache/venvs/hybrid-diffusion-eval/bin/python
export HF_HOME=$L/env/HybridDiffusion/cache/huggingface HF_DATASETS_CACHE=$L/env/HybridDiffusion/cache/huggingface/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
[ "${SMOKE:-0}" = 1 ] && { LIMIT=10; CONCS="1"; SHOW=2; } || SHOW=0
CONCS=${CONCS:-1 4 8 16}; LIMIT=${LIMIT:-64}; MAXTOK=${MAXTOK:-512}; MAXSEQS=${MAXSEQS:-16}
PB=$((33000 + (${SLURM_JOB_ID:-$$} % 400) * 10))
echo "{\"job\":\"$JOB\",\"engine\":\"$ENGINE\",\"cfgs\":\"$CFGS\",\"concs\":\"$CONCS\",\"limit\":$LIMIT,\"max_tokens\":$MAXTOK,\"max_seqs\":$MAXSEQS,\"host\":\"$(hostname)\",\"slurm_job\":\"${SLURM_JOB_ID:-}\",\"start\":\"$(date -Is)\"}" > $OUT/job.json
declare -A PID PORT; NAMES=(); i=0
IFS=";" read -r -a LIST <<< "$CFGS"
for cfg in "${LIST[@]}"; do set -- $cfg; n=$1; m=$2; b=${3:-0}; p=$((PB+i)); PORT[$n]=$p; NAMES+=($n)
  MODE=$m NAME=$n PORT=$p GPU=$i BLOCK=$b MAXSEQS=$MAXSEQS bash $L/scripts/lib/dflash/serve_$ENGINE.sh > $OUT/serve_$n.log 2>&1 &
  PID[$n]=$!; i=$((i+1)); done
healthy(){ curl -s http://localhost:$1/v1/models 2>/dev/null | grep -q "\"$2\""; }
for n in ${NAMES[@]}; do for t in $(seq 1 400); do healthy ${PORT[$n]} $n && break; kill -0 ${PID[$n]} 2>/dev/null || break; sleep 3; done
  if healthy ${PORT[$n]} $n; then echo "$n healthy"; else echo "$n NOT HEALTHY"; grep -h -i -E "error|Traceback|unrecognized|invalid" $OUT/serve_$n.log | grep -v "WARNING\|INFO" | tail -4 | cut -c1-220; fi; done
RP=(); for n in ${NAMES[@]}; do healthy ${PORT[$n]} $n || continue
  ( for C in $CONCS; do $EVALPY $L/scripts/lib/sweep_client.py --port ${PORT[$n]} --model $n --limit $LIMIT --concurrency $C --max_tokens $MAXTOK --show $SHOW --save $OUT/bench_${n}_c$C.json > $OUT/bench_${n}_c$C.log 2>&1
      echo "$n C=$C rc=$? : $(tail -1 $OUT/bench_${n}_c$C.log | cut -c1-260)"; done ) & RP+=($!); done
wait "${RP[@]}"
echo "--- acceptance / spec metrics from server logs"
for n in ${NAMES[@]}; do grep -h -i -o "SpecDecoding metrics.*\|accept[a-z_ ]*length[^,]*\|Mean acceptance length[^,]*" $OUT/serve_$n.log | tail -2 | sed "s/^/$n: /" | cut -c1-200; done
for n in ${NAMES[@]}; do kill ${PID[$n]} 2>/dev/null; done; sleep 5; for n in ${NAMES[@]}; do pkill -P ${PID[$n]} 2>/dev/null; done
echo "{\"end\":\"$(date -Is)\"}" > $OUT/job_end.json; echo "=== DFLASH BENCH $JOB DONE ==="
