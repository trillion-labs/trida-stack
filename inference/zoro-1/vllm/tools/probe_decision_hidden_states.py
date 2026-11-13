"""Compare stock vLLM token pooling with the SGLang decision reference on a GPU.

Run each backend in a separate process against the same checkpoint, then compare
the saved artifacts. No weights, tokenizer, or decision rows are modified.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from decide_engine import DecideEngine, PointerHead, prefix_prompts  # noqa: E402
from decide_request import parse  # noqa: E402


REQUEST = {
    "state": {"invoice": {"amount": 120, "limit": 100}, "note": "literal <|box_end|> stays text"},
    "questions": {
        "over_limit": {"type": "noul", "instructions": "The amount exceeds the limit."},
        "route": {"type": "choice", "instructions": "Where should this invoice go?",
                  "criteria": {"approve": "Within the limit", "review": "Over the limit", "reject": "Invalid"}},
        "amount": {"type": "score", "instructions": "How large is the invoice?",
                   "criteria": ["Below 100", "100 to 200", "Above 200"]},
    },
}


def encode(checkpoint):
    from transformers import AutoTokenizer
    return parse(REQUEST).encode(AutoTokenizer.from_pretrained(checkpoint))


def probabilities(checkpoint, states):
    saved = torch.load(Path(checkpoint) / "jev_head.pt", map_location="cpu", weights_only=True)
    head = PointerHead(saved["head"]["q.weight"].shape[1], saved["head_dim"]).eval()
    head.load_state_dict(saved["head"])
    temperature_path = Path(checkpoint) / "temperature.json"
    temperature = json.loads(temperature_path.read_text()).get("text", 1) if temperature_path.exists() else 1
    with torch.no_grad():
        return [torch.softmax(head(h[-1:], h[:-1], torch.zeros(len(h) - 1, dtype=torch.long)) / temperature, -1)
                for h in states]


def run_vllm(args):
    from vllm import LLM, PoolingParams
    from vllm.config import PoolerConfig
    enc = encode(args.checkpoint)
    llm = LLM(model=args.checkpoint, runner="pooling", convert="embed",
              pooler_config=PoolerConfig(task="token_embed", tok_pooling_type="ALL", use_activation=False),
              dtype="bfloat16", max_model_len=8192, max_num_seqs=8,
              gpu_memory_utilization=0.5, enforce_eager=True, enable_prefix_caching=False,
              trust_remote_code=True)
    prompts = [{"prompt_token_ids": row["ids"]} for row in enc.rows]
    selected = []
    for _ in range(2):  # Repeated batch tests output ordering and state reuse.
        outs = llm.encode(prompts, pooling_task="token_embed",
                          pooling_params=PoolingParams(use_activation=False), use_tqdm=False)
        batch = []
        for row, out in zip(enc.rows, outs):
            h = torch.as_tensor(out.outputs.data).float().cpu()
            assert h.ndim == 2 and len(h) == len(row["ids"]), (h.shape, len(row["ids"]))
            assert torch.isfinite(h).all()
            batch.append(h[row["opts"] + [row["decide"]]])
        assert len(batch) == len(enc.rows)
        selected.append(batch)
    return {"version": importlib.metadata.version("vllm"), "rows": enc.rows,
            "states": selected[0], "probabilities": probabilities(args.checkpoint, selected[0]),
            "repeat_max_abs": max((a-b).abs().max().item() for a, b in zip(*selected))}


def run_sglang(args):
    engine = DecideEngine(args.checkpoint, mem_fraction_static=0.5, context_length=8192)
    try:
        enc = encode(args.checkpoint)
        prompts, spans = prefix_prompts(enc, engine.image_pad)
        outs = engine._run(engine.engine.async_generate(
            input_ids=prompts, sampling_params={"max_new_tokens": 1, "temperature": 0},
            return_hidden_states="last"))
        h = torch.tensor([out["meta_info"]["hidden_states"] for out in outs], dtype=torch.float32)
        states = [h[start:start + count] for start, count in spans]
        return {"version": importlib.metadata.version("sglang"), "rows": enc.rows,
                "states": states, "probabilities": probabilities(args.checkpoint, states)}
    finally:
        engine.shutdown()


def run_worker(args):
    from vllm_decide_engine import VllmDecideEngine, row_prompt
    engine = VllmDecideEngine(args.checkpoint, mem_fraction=0.5, context_length=8192)
    try:
        enc = parse(REQUEST).encode(engine.tok)
        messages = [{"role": "user", "content": "Say hello."}]
        before = engine.chat(messages, max_new_tokens=8, temperature=0, enable_thinking=False)

        async def capture():
            return [torch.stack([await engine._capture_prefix(row_prompt(
                {"ids": row["ids"][:pos + 1]}, engine.image_pad))
                for pos in row["opts"] + [row["decide"]]]) for row in enc.rows]

        states, repeated = engine._run(capture()), engine._run(capture())
        after = engine.chat(messages, max_new_tokens=8, temperature=0, enable_thinking=False)
        assert before == after, "ordinary generation changed after capture"
        return {"version": importlib.metadata.version("vllm"), "rows": enc.rows,
                "states": states, "probabilities": probabilities(args.checkpoint, states),
                "worker": engine._worker_identity, "ordinary_generation_unchanged": True,
                "repeat_max_abs": max((a-b).abs().max().item() for a, b in zip(states, repeated))}
    finally:
        engine.shutdown()


def compare(args):
    v = torch.load(args.output / ("worker.pt" if args.backend == "compare-worker" else "vllm.pt"), weights_only=True)
    s = torch.load(args.output / "sglang.pt", weights_only=True)
    assert v["rows"] == s["rows"], "Decision input rows changed between backends"
    assert len(v["probabilities"]) == len(s["probabilities"]) == len(REQUEST["questions"])
    differences = [(a-b).abs().max().item() for a, b in zip(v["probabilities"], s["probabilities"])]
    report = {"vllm": v["version"], "sglang": s["version"], "checkpoint": args.checkpoint,
              "row_lengths": [len(r["ids"]) for r in v["rows"]],
              "repeat_max_abs": v["repeat_max_abs"], "max_probability_difference": max(differences),
              "head_sha256": hashlib.sha256((Path(args.checkpoint) / "jev_head.pt").read_bytes()).hexdigest(),
              "min_hidden_cosine_similarity": min(torch.nn.functional.cosine_similarity(a, b).min().item()
                                                  for a, b in zip(v["states"], s["states"])),
              "same_argmax": all(a.argmax() == b.argmax() for a, b in zip(v["probabilities"], s["probabilities"])),
              "vllm_probabilities": [x.tolist() for x in v["probabilities"]],
              "sglang_probabilities": [x.tolist() for x in s["probabilities"]]}
    report["same_argmax"] = bool(report["same_argmax"])
    name = "worker-report.json" if args.backend == "compare-worker" else "report.json"
    (args.output / name).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert max(differences) < 0.01, "Probability mismatch exceeds the probe's BF16 tolerance"
    assert report["same_argmax"], "Selected options differ"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("backend", choices=("vllm", "worker", "sglang", "compare", "compare-worker"))
    ap.add_argument("checkpoint")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.backend in ("compare", "compare-worker"):
        compare(args)
    else:
        result = {"vllm": run_vllm, "worker": run_worker, "sglang": run_sglang}[args.backend](args)
        torch.save(result, args.output / f"{args.backend}.pt")
        print(f"Saved {args.backend} hidden-state probe", flush=True)


if __name__ == "__main__":
    main()
