"""A tiny Zoro-1 client: the two calls most demos need, plus a mock so the demo runs without a model.

  decide(state, questions)  POST /v1/systemone        typed answers with calibrated probabilities
  chat(messages)            POST /v1/chat/completions  text, thinking off

Copy it as is or replace it; nothing else in the repo depends on it.
"""
from __future__ import annotations

import hashlib
import json
import random

import httpx


class Zoro:
    def __init__(self, url: str | None = None, model: str = "zoro-1", mock: bool = False, timeout: float = 120):
        self.url, self.model = (url or "").rstrip("/"), model
        self.mock = mock or not self.url
        self.http = None if self.mock else httpx.Client(timeout=timeout)

    def decide(self, state: dict, questions: dict, image: str | None = None) -> dict:
        """{question id: answer}. Question types: noul (yes/no), choice (pick one), score (ordered levels)."""
        if self.mock:
            return _mock_decide(state, questions)
        body = {"model": self.model, "state": state, "questions": questions, "reasoning": False}
        if image:
            body["images"] = [image]          # one data URL / base64 image per request
        r = self.http.post(f"{self.url}/v1/systemone", json=body)
        r.raise_for_status()
        return r.json()["answers"]

    def chat(self, messages: list[dict], max_tokens: int = 200, temperature: float = 0.7) -> str:
        if self.mock:
            return "(mock) " + messages[-1]["content"][:60]
        r = self.http.post(f"{self.url}/v1/chat/completions", json={
            "model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}})
        r.raise_for_status()
        return (r.json()["choices"][0]["message"].get("content") or "").strip()


def _mock_decide(state: dict, questions: dict) -> dict:
    rng = random.Random(hashlib.sha256(json.dumps([state, questions], sort_keys=True).encode()).digest())
    out = {}
    for qid, q in questions.items():
        if q["type"] == "noul":
            out[qid] = {"type": "noul", "noul": rng.random()}
            continue
        keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
        w = [rng.random() for _ in keys]
        probs = {k: x / sum(w) for k, x in zip(keys, w)}
        if q["type"] == "choice":
            best = max(probs, key=probs.get)
            out[qid] = {"type": "choice", "choice": best, "confidence": probs[best], "probabilities": probs}
        else:
            out[qid] = {"type": "score", "score": sum(int(k) * p for k, p in probs.items()), "probabilities": probs}
    return out
