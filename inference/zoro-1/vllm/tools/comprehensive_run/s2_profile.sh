#!/bin/bash
# torch-profiler trace of ONE 128-token self-spec request on the S2 build (graph config selectable via COMPILATION_CONFIG).
set -u
L=$SCRATCH; ROOT=$L/env/vllm-uv27; OUT=${RUN_DIR:-$L/runs/prof_$(date +%Y%m%d_%H%M%S)}; mkdir -p $OUT/trace
S=$L/scripts/serve_diff_cl.sh; P=$((30000 + (${SLURM_JOB_ID:-$$} % 400) * 10)); N=${N:-4}; CL=$((2*N-1))
COMPAT=$L/env/cudacompat/extracted/usr/local/cuda-13.0/compat
NVLIBS=$(find $ROOT/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | tr "\n" ":")
echo "code=$(md5sum ${CODE_DIR:-$L/code/vllm-native-dev9}/vllm_native_diffusion/qwen3_5_diffusion.py | cut -c1-8) selfspec N=$N CL=$CL cc=${COMPILATION_CONFIG:-default} ntok=128" > $OUT/meta.txt
env LD_LIBRARY_PATH=$COMPAT:$NVLIBS PATH=$ROOT/.venv/bin:$PATH CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12 PYTHONPATH=${CODE_DIR:-$L/code/vllm-native-dev9} VLLM_PLUGINS=trida_diffusion VLLM_LOGGING_LEVEL=WARNING \
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PORT=$P CL=$CL THRESH=0.90 MAXSTEPS=8 TRIDA_SELFSPEC_N=$N TRIDA_COUNT_FWD=1 PROFILER_DIR=$OUT/trace bash $S > $OUT/C_serve.log 2>&1 & PC=$!
for i in $(seq 1 360); do curl -s http://localhost:$P/v1/models 2>/dev/null | grep -q trida-bd && break; kill -0 $PC 2>/dev/null || { echo DIED | tee -a $OUT/meta.txt; exit 1; }; sleep 3; done
echo "healthy" | tee -a $OUT/meta.txt
PROMPT="Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take? Answer step by step:"
curl -s http://localhost:$P/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"trida-bd\",\"prompt\":\"$PROMPT\",\"max_tokens\":128}" -o /dev/null
sleep 2; C0=$(stat -c %s $OUT/C_serve.log)
curl -s -X POST http://localhost:$P/start_profile -o /dev/null -w "start_profile=%{http_code}\n" | tee -a $OUT/meta.txt
curl -s http://localhost:$P/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"trida-bd\",\"prompt\":\"$PROMPT\",\"max_tokens\":128}" -o $OUT/C_profiled_resp.json
curl -s -X POST http://localhost:$P/stop_profile -o /dev/null -w "stop_profile=%{http_code}\n" --max-time 900 | tee -a $OUT/meta.txt
sleep 3; C1=$(stat -c %s $OUT/C_serve.log); tail -c +$((C0+1)) $OUT/C_serve.log | head -c $((C1-C0)) > $OUT/C_profiled_slice.log
grep "\[FWD\]" $OUT/C_profiled_slice.log | tail -1 > $OUT/C_profiled_fwd.txt; : > $OUT/A_time.txt; : > $OUT/B_time.txt
kill $PC 2>/dev/null; sleep 4; pkill -P $PC 2>/dev/null
python3 $L/diag/analyze_profile.py $OUT 2>&1 | tee $OUT/analysis.txt
echo "=== PROF DONE $OUT ==="
