"""Bounded H100 probe; this does not enable a production decision endpoint.

Run reference and candidate in separate processes/environments, then compare.
The candidate needs decision_readout.patch on top of the public two-stream patch.
The causal mode uses the same fork's native full-prefill hidden-state API.
It deliberately generates/discards one token to retain existing request cleanup.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from decide_engine import (  # noqa: E402
    DecideEngine, PointerHead, load_image_processor, prefix_prompts, serving_temperatures,
)
from decide_request import parse, to_answer  # noqa: E402


def score(checkpoint, encoded, states):
    saved = torch.load(Path(checkpoint) / "jev_head.pt", map_location="cpu", weights_only=True)
    head = PointerHead(saved["head"]["q.weight"].shape[1], saved["head_dim"]).eval()
    head.load_state_dict(saved["head"])
    temperatures = serving_temperatures(checkpoint)
    temperature = temperatures[1 if encoded.has_image else 0]
    with torch.no_grad():
        return [torch.softmax(head(h[-1:], h[:-1], torch.zeros(len(h) - 1, dtype=torch.long))
                              / temperature, -1).tolist() for h in states]


def run(args):
    from transformers import AutoTokenizer

    cases = json.loads(args.requests.read_text())
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    proc = load_image_processor(args.checkpoint)
    if args.mode == "reference":
        wrapper = DecideEngine(args.checkpoint, mem_fraction_static=0.5, context_length=16384)
        engine = wrapper.engine
    else:
        if not args.backend:
            raise ValueError("fork modes require --backend pointing to the isolated backend's eval directory")
        # The shared environment has a .pth that prepends another checkout.
        # Override it in this process; multiprocessing inherits this sys.path.
        sys.path.insert(0, str(args.backend.resolve()))
        import sglang as sgl
        import sglang.srt.layers.logits_processor as logits_module
        assert Path(logits_module.__file__).resolve().is_relative_to(args.backend.resolve())
        print("candidate logits source:", logits_module.__file__, flush=True)
        sys.path.insert(0, str(Path(__file__).parent))
        from shape_for_sglang import shape

        model = str(args.output.parent / "shaped")
        if not Path(model).exists():
            shape(Path(args.checkpoint), Path(model))
        algorithm = {} if args.mode == "causal" else {
            "dllm_algorithm": "HybridDiffusionSelfSpec",
            "dllm_algorithm_config": str(ROOT / "configs/zoro1_self_spec_b15_g8.yaml"),
        }
        engine = sgl.Engine(
            model_path=model, mem_fraction_static=0.5, context_length=16384,
            enable_return_hidden_states=True, disable_cuda_graph=True,
            disable_overlap_schedule=True, max_running_requests=16,
            attention_backend="flashinfer", **algorithm,
        )
    result = {"mode": args.mode, "checkpoint": args.checkpoint,
              "requests_sha256": hashlib.sha256(args.requests.read_bytes()).hexdigest(),
              "prefix_layout": args.reference_layout if args.mode == "reference" else "serial",
              "head_sha256": hashlib.sha256((Path(args.checkpoint) / "jev_head.pt").read_bytes()).hexdigest(),
              "multimodal_enabled": bool(engine.tokenizer_manager.model_config.is_multimodal),
              "cases": []}
    captured_states = []
    try:
        for repeat in range(args.repeats):
            for case in cases:
                req = parse(case["request"])
                encoded = req.encode(tok, proc)
                if encoded.has_image and not result["multimodal_enabled"]:
                    item = {"name": case["name"], "repeat": repeat,
                            "unsupported": "DLLM backend is text-only; image inputs would be ignored"}
                    result["cases"].append(item)
                    print(json.dumps(item), flush=True)
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                    continue
                prompts, spans = prefix_prompts(encoded, tok.convert_tokens_to_ids("<|image_pad|>"))
                kw = {}
                if encoded.has_image:
                    image = req.image
                    ps = proc.patch_size
                    grid = encoded.grid
                    from PIL import Image
                    image = image.resize((int(grid[0][2]) * ps, int(grid[0][1]) * ps), Image.Resampling.BICUBIC)
                    kw["image_data"] = [image] * len(prompts)
                params = {"max_new_tokens": 1, "temperature": 0}
                if args.mode == "reference" and args.reference_layout == "batch":
                    outputs = engine.generate(input_ids=prompts, sampling_params=params,
                                              return_hidden_states="last", **kw)
                else:
                    # Bounded feasibility probe: avoid inline absorption into a
                    # running decode batch. Mixed-request capture is a later step.
                    single_kw = {"image_data": kw["image_data"][0]} if kw else {}
                    outputs = [engine.generate(input_ids=ids, sampling_params=params,
                                               return_hidden_states=("last" if args.mode == "reference" else True),
                                               **single_kw) for ids in prompts]
                states = []
                for output in outputs:
                    h = torch.tensor(output["meta_info"]["hidden_states"], dtype=torch.float32)
                    h = h.reshape(-1, h.shape[-1])
                    assert h.shape[0] >= 1 and torch.isfinite(h).all(), h.shape
                    if args.mode != "causal":
                        assert h.shape[0] == 1, h.shape
                    # The fork's native causal API returns all fresh prefill
                    # states. Its final state is the same prefix endpoint.
                    states.append(h[-1])
                assert len(states) == len(prompts)
                states = torch.stack(states)
                if args.save_states:
                    captured_states.append({"name": case["name"], "repeat": repeat, "states": states})
                    torch.save(captured_states, args.output.with_suffix(".states.pt"))
                groups = [states[start:start + count] for start, count in spans]
                probabilities = score(args.checkpoint, encoded, groups)
                item = {"name": case["name"], "repeat": repeat,
                        "prompt_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                        "row_lengths": [len(r["ids"]) for r in encoded.rows],
                        "probabilities": probabilities,
                        "questions": [{"qid": q.qid, "qtype": q.qtype, "keys": q.keys, "options": q.options}
                                      for q in req.questions],
                        "answers": {q.qid: to_answer(q, p) for q, p in zip(req.questions, probabilities)}}
                result["cases"].append(item)
                print(json.dumps(item), flush=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
            if hasattr(engine, "flush_cache"):
                response = engine.flush_cache()
                result.setdefault("cache_flush", []).append(str(response))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        engine.shutdown()


def compare(args):
    sys.path.insert(0, str(ROOT / "mlx/tests"))
    from compare_decisions import differences as answer_differences

    ref = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    assert ref["head_sha256"] == candidate["head_sha256"]
    if "requests_sha256" in ref and "requests_sha256" in candidate:
        assert ref["requests_sha256"] == candidate["requests_sha256"]
    lookup = {c["name"]: c for c in ref["cases"]}
    state_files = (args.reference.with_suffix(".states.pt"), args.candidate.with_suffix(".states.pt"))
    stored_states = None
    if all(path.exists() for path in state_files):
        stored_states = [{(c["name"], c["repeat"]): c["states"]
                          for c in torch.load(path, map_location="cpu", weights_only=True)} for path in state_files]
    rows = []
    for case in candidate["cases"]:
        expected = lookup[case["name"]]
        if "unsupported" in case:
            rows.append({"name": case["name"], "repeat": case["repeat"],
                         "unsupported": case["unsupported"], "passed": False})
            continue
        assert "unsupported" not in expected, case["name"]
        assert case["prompt_sha256"] == expected["prompt_sha256"], case["name"]
        assert case["row_lengths"] == expected["row_lengths"]
        assert len(case["probabilities"]) == len(expected["probabilities"])
        differences = []
        for a, b in zip(expected["probabilities"], case["probabilities"]):
            assert len(a) == len(b)
            differences.extend(abs(x - y) for x, y in zip(a, b))
        answer_match = all(
            a.get("choice") == case["answers"][qid].get("choice")
            for qid, a in expected["answers"].items() if a["type"] == "choice"
        )
        answer_errors = answer_differences(expected["answers"], case["answers"], args.atol)
        behavior = []
        if args.profile == "self-spec-bf16":
            assert expected["questions"] == case["questions"]
            answer_errors = []
            for item in (expected, case):
                canonical = {q["qid"]: to_answer(SimpleNamespace(**q), p)
                             for q, p in zip(item["questions"], item["probabilities"])}
                if item["answers"] != canonical:
                    answer_errors.append("derived answer fields disagree with their own probabilities")
            for q, a, b in zip(expected["questions"], expected["probabilities"], case["probabilities"]):
                margin = sorted(a, reverse=True)[0] - sorted(a, reverse=True)[1] if len(a) > 1 else 1.0
                changed = a.index(max(a)) != b.index(max(b))
                derived = {key: abs(expected["answers"][q["qid"]][key] - case["answers"][q["qid"]][key])
                           for key in ("score", "confidence") if key in expected["answers"][q["qid"]]}
                behavior.append({"question": q["qid"], "type": q["qtype"], "winner_changed": changed,
                                 "reference_margin": margin,
                                 "derived_differences": derived,
                                 "threshold_crossings": [t for t in (0.5, 0.7, 0.9) if (max(a) < t) != (max(b) < t)]})
                if changed and margin > 2 * args.atol:
                    answer_errors.append(f"{q['qid']}: winner changed away from a numerical tie")
        rows.append({"name": case["name"], "repeat": case["repeat"],
                     "max_probability_difference": max(differences), "choice_match": answer_match,
                     "probabilities_passed": max(differences) <= args.atol,
                     "answer_errors": answer_errors,
                     "behavior": behavior,
                     "passed": max(differences) <= args.atol and not answer_errors})
        if stored_states is not None:
            key = (case["name"], case["repeat"])
            ref_key = key if key in stored_states[0] else (case["name"], expected["repeat"])
            a, b = stored_states[0][ref_key], stored_states[1][key]
            assert a.shape == b.shape
            delta = a - b
            rows[-1]["hidden_states"] = {
                "equal": torch.equal(a, b), "max_abs": delta.abs().max().item(),
                "rms": delta.square().mean().sqrt().item(),
                "min_cosine": torch.nn.functional.cosine_similarity(a, b).min().item(),
            }
    assert rows and set(lookup) == {r["name"] for r in rows}
    report = {"profile": args.profile, "atol": args.atol, "comparisons": rows,
              "passed": all(r["passed"] for r in rows)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(not report["passed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["reference", "candidate", "causal", "compare"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--backend", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--save-states", action="store_true")
    parser.add_argument("--reference-layout", choices=["batch", "serial"], default="batch")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--profile", choices=["strict", "self-spec-bf16"], default="strict")
    parser.add_argument("--atol", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.atol is None:
        args.atol = 0.02 if args.profile == "self-spec-bf16" else 0.005
    return compare(args) if args.mode == "compare" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
