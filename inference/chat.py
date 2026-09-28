#!/usr/bin/env python3
"""nano-inference chat — talk to a served diffusion LLM (OpenAI-compatible).

  python chat.py "What is 17 * 23?"     # one-shot
  python chat.py                         # interactive REPL (Ctrl-D to exit)
  python chat.py --no-think "..."        # disable the thinking trace

Assumes a model is already served (see serve.py). Sampling defaults:
temp 1.0 / top_p 0.95 / top_k 50 / no presence penalty — the same canonical setting
eval.py uses, so chat and benchmark runs decode identically.
"""
import argparse


def served_model_id(base):
    """The id the server exposes (vLLM rejects unknown model names; SGLang accepts any)."""
    import requests  # deferred, as in complete(): `chat.py --help` must work without it installed
    try:
        return requests.get(f"{base}/v1/models", timeout=10).json()["data"][0]["id"]
    except Exception:
        return "default"


def complete(base, prompt, args):
    import requests  # deferred so `chat.py --help` works before requests is installed
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "chat_template_kwargs": {"enable_thinking": args.think},
    }
    r = requests.post(f"{base}/v1/chat/completions", json=body, timeout=args.timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser(description="Chat with a served diffusion LLM.")
    ap.add_argument("prompt", nargs="?", help="one-shot prompt; omit for interactive REPL")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="disable enable_thinking")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--model", default=None, help="served model id (default: first id from /v1/models)")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    args.model = args.model or served_model_id(base)   # resolve once, not per REPL turn
    if args.prompt:
        print(complete(base, args.prompt, args))
        return

    print("nano-inference chat — Ctrl-D to exit")
    while True:
        try:
            prompt = input("\n>>> ")
        except EOFError:
            print()
            break
        if prompt.strip():
            print(complete(base, prompt, args))


if __name__ == "__main__":
    main()
