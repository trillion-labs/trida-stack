"""Zoro-1 as the guesser: one /v1/systemone call per drawing.

Both Zoro-1 servers take the same System One body: inference/zoro-1/serve_decide.py (SGLang, batches concurrent
requests; use this for a public game) and inference/zoro-1/mlx (trida_mlx.server on a Mac, one request at a time;
fine for a room or two). Each drawing is one request with two questions that share one prefill:

  guess     choice over the candidate words (the option keys are the words themselves)
  has_text  noul: is an answer written out in the drawing? (spelling the answer is the obvious cheat; symbols
            such as $ ? ! or a heart are part of a picture and don't count)
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import random
import time
from dataclasses import dataclass, field

import httpx

STATE = ("A drawing game. A player drew a picture of one word, with a pen, in a few seconds, without writing "
         "any letters. You are shown the drawing.")
GUESS = "Which word is the drawing meant to show?"
HAS_TEXT = "Did the player write out the answer (or part of it) as text in the drawing, instead of drawing it?"
HAS_TEXT_CRITERIA = {
    "true": "yes: words or letters spelling out or abbreviating an answer are written in the drawing",
    "false": "no: only pictures, possibly with symbols such as $, ?, !, arrows, hearts, music notes or a number on a "
             "clock or a sign",
}
MAX_IMAGE_PIXELS = 448 * 448  # the model's image cap; the client already sends 448x336


@dataclass
class Verdict:
    probs: dict[str, float] = field(default_factory=dict)  # candidate word -> probability (sums to ~1)
    text_p: float = 0.0                                     # P(the drawing contains writing)
    ms: int = 0
    error: str | None = None


def systemone_body(model: str, image: str, candidates: list[str], glosses: dict | None = None) -> dict:
    return {
        "model": model,
        "state": {"game": STATE},
        "images": [image],
        "questions": {
            "guess": {"type": "choice", "instructions": GUESS, "criteria": {w: (glosses or {}).get(w) or "" for w in candidates}},
            "has_text": {"type": "noul", "instructions": HAS_TEXT, "criteria": HAS_TEXT_CRITERIA},
        },
        "max_image_pixels": MAX_IMAGE_PIXELS,
        "reasoning": False,
    }


def parse_response(data: dict, candidates: list[str]) -> Verdict:
    answers = data.get("answers") or {}
    guess = answers.get("guess") or {}
    raw = guess.get("probabilities") or {}
    probs = {w: float(raw.get(w, 0.0)) for w in candidates}
    total = sum(probs.values())
    if total <= 0:
        raise ValueError(f"no probabilities in the answer: {guess!r}")
    probs = {w: p / total for w, p in probs.items()}
    text_p = float((answers.get("has_text") or {}).get("noul", 0.0))
    return Verdict(probs=probs, text_p=text_p)


class Judge:
    """Calls the model server, or fakes it with `mock=True` (for playing without a model)."""

    def __init__(self, url: str | None = None, model: str = "zoro-1", concurrency: int = 4,
                 timeout: float = 120.0, mock: bool = False):
        self.url = (url or "").rstrip("/")
        self.model = model
        self.mock = mock or not self.url
        self.sem = asyncio.Semaphore(max(1, concurrency))
        self.client = None if self.mock else httpx.AsyncClient(timeout=timeout)

    @property
    def mode(self) -> str:
        return "mock" if self.mock else self.url

    async def health(self) -> dict:
        if self.mock:
            return {"ok": True, "mode": "mock"}
        try:
            r = await self.client.get(f"{self.url}/v1/models", timeout=5)
            r.raise_for_status()
            ids = [m.get("id") for m in r.json().get("data", [])]
            return {"ok": True, "mode": self.url, "models": ids}
        except Exception as e:  # noqa: BLE001 - reported to the page as-is
            return {"ok": False, "mode": self.url, "error": str(e)}

    async def guess(self, image: str, candidates: list[str], truth: str | None = None,
                    glosses: dict | None = None) -> Verdict:
        """`truth` is only used by the mock (so mock games have a winner); the real model never sees it."""
        t0 = time.perf_counter()
        try:
            if self.mock:
                await asyncio.sleep(0.4)
                v = _mock(image, candidates, truth)
            else:
                async with self.sem:
                    r = await self.client.post(f"{self.url}/v1/systemone",
                                               json=systemone_body(self.model, image, candidates, glosses))
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                v = parse_response(r.json(), candidates)
        except Exception as e:  # noqa: BLE001 - one failed drawing must not end the round
            v = Verdict(error=str(e) or type(e).__name__)
        v.ms = int((time.perf_counter() - t0) * 1000)
        return v

    async def aclose(self):
        if self.client:
            await self.client.aclose()


def _mock(image: str, candidates: list[str], truth: str | None) -> Verdict:
    """Deterministic per drawing; more ink -> more confident in the truth."""
    rng = random.Random(hashlib.sha256((image + "|".join(candidates)).encode()).digest())
    ink = min(1.0, len(image) / 40000)
    logits = {w: rng.gauss(0, 1) + (1.5 + 2.0 * ink if w == truth else 0.0) for w in candidates}
    m = max(logits.values())
    exp = {w: math.exp(x - m) for w, x in logits.items()}
    z = sum(exp.values())
    return Verdict(probs={w: e / z for w, e in exp.items()}, text_p=rng.random() * 0.1)

