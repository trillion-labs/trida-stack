#!/bin/bash
# A3: self-spec GDN state fingerprint check. Run item 0 (128 tok) on a self-spec server with
# [SSTATE] dumps, then for each step re-prefill the emitted prefix (max_tokens=1) on the SAME server
# and compare the persisted ssm/conv state per layer against the fresh-prefill state.
# env: CODE_DIR, N (default 4), NTOK (128)
set -u
L=$SCRATCH; ROOT=$L/env/vllm-uv27; N=${N:-4}; CL=$((2*N-1)); NTOK=${NTOK:-128}
OUT=${RUN_DIR:-$L/runs/a3_$(date +%Y%m%d_%H%M%S)}; mkdir -p $OUT; cd $OUT
COMPAT=$L/env/cudacompat/extracted/usr/local/cuda-13.0/compat
NVLIBS=$(find $ROOT/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | tr "\n" ":")
VENV="LD_LIBRARY_PATH=$COMPAT:$NVLIBS PATH=$ROOT/.venv/bin:$PATH CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12 PYTHONPATH=${CODE_DIR:-$L/code/vllm-native-dev9} VLLM_PLUGINS=trida_diffusion VLLM_LOGGING_LEVEL=WARNING"
P=$((30000 + (${SLURM_JOB_ID:-$$} % 400) * 10)); G=${CUDA_VISIBLE_DEVICES:-0}
env $VENV CUDA_VISIBLE_DEVICES=$G PORT=$P CL=$CL THRESH=0.90 MAXSTEPS=8 TRIDA_SELFSPEC_N=$N TRIDA_DUMP_STATE=1 TRIDA_COUNT_FWD=1 TRIDA_TRACE_JSONL=$OUT/trace.jsonl bash $L/scripts/serve_diff_cl.sh > $OUT/serve.log 2>&1 & SP=$!
for i in $(seq 1 360); do curl -s http://localhost:$P/v1/models 2>/dev/null | grep -q trida-bd && break; kill -0 $SP 2>/dev/null || { echo DIED; exit 1; }; sleep 3; done
echo "server healthy on $P"
$ROOT/.venv/bin/python - $OUT $P $CL $NTOK <<"PY" 2>&1 | grep -v "Warning\|warn" | tee $OUT/results.txt
import json,urllib.request,os,time,re,sys,math,statistics as st
OUT,PORT,CL,NTOK=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
CK="$SCRATCH/trida-stack-run/checkpoints/qwen35-4b-flare-v6-2n/step_18000"
from transformers import AutoTokenizer; tok=AutoTokenizer.from_pretrained(CK,trust_remote_code=True)
rows=[json.loads(l) for l in open("$SCRATCH/diag/gsm8k_test.jsonl")]
def http(body,timeout=900):
    req=urllib.request.Request(f"http://localhost:{PORT}/v1/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}); return json.loads(urllib.request.urlopen(req,timeout=timeout).read())
q=rows[0]["question"]+"\nPlease reason step by step, and put your final answer within \\boxed{}."
s=tok.apply_chat_template([{"role":"user","content":q}],tokenize=False,add_generation_prompt=True,enable_thinking=False); ids=tok(s,add_special_tokens=False).input_ids; P=len(ids)
SS=re.compile(r"\[SSTATE\] slot=(\d+) seqlen=(\d+) k=(\d+) proj=(\[.*\])"); FW=re.compile(r"\[FWD\] req=(\d+) ")
NS={"nan":float("nan"),"inf":float("inf"),"__builtins__":{}}
def dumps_since(a0):
    time.sleep(1.5); seg=open(OUT+"/serve.log","rb").read()[a0:].decode("utf-8","replace")
    out=[]; pend=[]
    for ln in seg.splitlines():
        m=SS.search(ln)
        if m: pend.append((int(m.group(2)),int(m.group(3)),eval(m.group(4),NS))); continue
        if FW.search(ln): out.extend(pend); pend=[]
    return out+pend
# ---- Phase 1: self-spec run
a0=os.path.getsize(OUT+"/serve.log"); d=http({"model":"trida-bd","prompt":ids,"max_tokens":NTOK}); run=dumps_since(a0)
time.sleep(1.0); tr=[json.loads(l) for l in open(OUT+"/trace.jsonl")][-1]
emitted=[t for b in tr["blocks"] for t in b["ids"]]; steps=tr["spec_steps"]
print("self-spec run: P=%d emitted=%d steps=%d tok/fwd=%.3f dumps=%d"%(P,len(emitted),len(steps),len(emitted)/max(len(steps),1),len(run)))
# committed length before each step t: L_0=0 (canvas t0=seed), L_{t+1}=L_t+1+n_acc_t
Ls=[0]
for k,na in steps: Ls.append(Ls[-1]+1+na)
run_by_seq={}
for (seq,k,proj) in run: run_by_seq.setdefault(seq,proj)   # first dump at that seqlen
# ---- Phase 2: fresh prefill of prompt+emitted[:L_t] for a spread of steps
picks=[t for t in range(1,len(steps)) if t<=12 or t%5==0][:40]
def norm(v): return math.sqrt(sum(x*x for x in v))
def rd(a,b,idx): a=[a[i] for i in idx]; b=[b[i] for i in idx]; return norm([x-y for x,y in zip(a,b)])/(norm(b)+1e-12)
print("\n| step | L (committed) | k, n_acc at step | seqlen | ssm mean | ssm max (L) | conv mean | conv max (L) | flag |")
print("|---:|---:|---|---:|---:|---:|---:|---:|---|")
bad=0
for t in picks:
    Lc=Ls[t]; prefix=ids+emitted[:Lc]; seq=len(prefix)+CL
    if seq not in run_by_seq: continue
    a0=os.path.getsize(OUT+"/serve.log")
    try: http({"model":"trida-bd","prompt":prefix,"max_tokens":1})
    except Exception as e: print("| %d | %d | err %s |"%(t,Lc,str(e)[:60])); continue
    ref=[x for x in dumps_since(a0) if x[0]==seq]
    if not ref: print("| %d | %d | (no ref dump) |"%(t,Lc)); continue
    rp=ref[0][2]; dp=run_by_seq[seq]; n=min(len(rp),len(dp))
    ssm=[rd(dp[li],rp[li],range(4)) for li in range(n)]; conv=[rd(dp[li],rp[li],range(4,6)) for li in range(n)]
    ms=max(range(n),key=lambda i:ssm[i]); mc=max(range(n),key=lambda i:conv[i])
    flag="" if max(ssm)<2e-2 and max(conv)<2e-2 else "**DIFF**"; bad+=bool(flag)
    print("| %d | %d | k=%d n_acc=%d | %d | %.2e | %.2e (L%d) | %.2e | %.2e (L%d) | %s |"%(t,Lc,steps[t-1][0],steps[t-1][1],seq,st.mean(ssm),ssm[ms],ms,st.mean(conv),conv[mc],mc,flag))
print("\nsteps checked: %d, flagged: %d"%(len(picks),bad))
# identity vs AR greedy for this item
try:
    ar=json.load(open("$SCRATCH/runs/20260909_143259/vllm-causal-clean/gsm8k_details_vllm-causal-clean.json"))[0]["generation"]
    gen=d["choices"][0]["text"]; k=next((i for i in range(min(len(ar),len(gen))) if ar[i]!=gen[i]),min(len(ar),len(gen)))
    print("identity vs AR greedy (item 0): first diff at char %d of %d (self-spec len %d)"%(k,len(ar),len(gen)))
except Exception as e: print("identity check skipped:",e)
PY
kill $SP 2>/dev/null; sleep 4; pkill -P $SP 2>/dev/null; echo "=== A3 DONE $OUT ==="
