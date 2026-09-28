#!/usr/bin/env python3
"""nano-inference ifeval — reproduce IFEval strict/loose on a served diffusion LLM.

  python eval_ifeval.py --port 30000                    # full set (541 prompts)
  python eval_ifeval.py --port 30000 --num-problems 50  # quick subset

Scoring uses the Google Research IFEval evaluator, vendored under ./ifeval_lib
(Apache-2.0). Extra deps: absl-py immutabledict langdetect nltk (see requirements.txt).
Kept as a separate file so the core eval.py stays tiny.
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Heavy imports (requests, datasets, and the ifeval_lib evaluator + its deps) are
# deferred into the functions that use them, so `eval_ifeval.py --help` works before
# those packages are installed.


def strip_thinking(t):
    return re.sub(r"^.*</think>\s*", "", t or "", count=1, flags=re.DOTALL)


def score(pred, ref):
    import importlib
    import sys

    # The vendored evaluator imports itself as `instruction_following_eval`; alias it.
    sys.modules.setdefault("instruction_following_eval", importlib.import_module("ifeval_lib"))
    from ifeval_lib.evaluation_lib import (
        InputExample,
        test_instruction_following_loose as _loose,
        test_instruction_following_strict as _strict,
    )
    inp = InputExample(
        key=ref["key"],
        instruction_id_list=ref["instruction_id_list"],
        prompt=ref["prompt"],
        kwargs=[{k: v for k, v in kw.items() if v is not None} for kw in ref["kwargs"]],
    )
    p = strip_thinking(pred)
    return (_strict(inp, {inp.prompt: p}).follow_instruction_list,
            _loose(inp, {inp.prompt: p}).follow_instruction_list)


def run_one(base, ref, args):
    import requests
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": ref["prompt"]}],
        "max_tokens": args.max_tokens, "temperature": args.temperature,
        "top_p": args.top_p, "top_k": args.top_k, "presence_penalty": args.presence_penalty,
        "chat_template_kwargs": {"enable_thinking": args.think},
    }
    try:
        r = requests.post(f"{base}/v1/chat/completions", json=body, timeout=args.timeout).json()
        return ref, r["choices"][0]["message"]["content"], None
    except Exception as e:
        return ref, "", str(e)


def main():
    ap = argparse.ArgumentParser(description="Reproduce IFEval on a served diffusion LLM.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--num-problems", type=int, default=0, help="0 = full set (first N otherwise)")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--presence-penalty", type=float, default=1.5)
    ap.add_argument("--no-think", dest="think", action="store_false")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--max-workers", type=int, default=64)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("google/IFEval", split="train")
    if 0 < args.num_problems < len(ds):
        ds = ds.select(range(args.num_problems))
    refs = [{
        "key": x["key"], "prompt": x["prompt"],
        "instruction_id_list": x["instruction_id_list"],
        "kwargs": json.loads(x["kwargs"]) if isinstance(x["kwargs"], str) else x["kwargs"],
    } for x in ds]
    base = f"http://{args.host}:{args.port}"
    print(f"ifeval: {len(refs)} prompts -> {base}  (workers={args.max_workers})", flush=True)

    ps = pt = ic = it = lps = lit = lic = 0  # strict prompt/inst, loose prompt/inst
    errors = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = [pool.submit(run_one, base, r, args) for r in refs]
        for i, f in enumerate(as_completed(futs), 1):
            ref, pred, err = f.result()
            errors += err is not None
            strict, loose = score(pred, ref)          # empty pred scores as all-fail
            pt += 1; ps += all(strict); it += len(strict); ic += sum(strict)
            lps += all(loose); lit += len(loose); lic += sum(loose)
            if i % max(len(refs) // 10, 1) == 0:
                print(f"  {i}/{len(refs)}  ({time.time()-t0:.0f}s)", flush=True)

    el = time.time() - t0
    print(f"\nifeval  ({pt} prompts, wall={el:.0f}s, errors={errors}):")
    print(f"  prompt-strict {ps/pt*100:.2f}   inst-strict {ic/max(it,1)*100:.2f}")
    print(f"  prompt-loose  {lps/pt*100:.2f}   inst-loose  {lic/max(lit,1)*100:.2f}")


if __name__ == "__main__":
    main()
