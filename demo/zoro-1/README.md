# Zoro-1 demos

Small, fun, runnable things built on **[Zoro-1](https://huggingface.co/trillionlabs/Zoro-1)**. Each folder here is
one self-contained demo with its own README, dependencies and tests. Play with them, fork them, and add yours.

| demo | what it is | Zoro-1 features used |
|---|---|---|
| [zoro-draw](zoro-draw/) | Multiplayer drawing game: everyone draws, Zoro-1 guesses, the probability on the right answer is your score. Decks include 한국어 속담 and memes. | decision head: `choice` + `noul` on an image |
| *yours* | [Add a demo](#add-a-demo) | |

## What Zoro-1 gives a demo

Zoro-1 is a 4B model (Trida-2.0 block-diffusion backbone) with two heads, served from one copy of the weights:

- **Chat** (`POST /v1/chat/completions`, OpenAI-compatible): text, tool calls, image input, optional thinking.
  It runs self-speculative decoding, which makes it fast on a laptop.
- **Decisions** (`POST /v1/systemone`): instead of generating text, the decision head answers typed questions about
  a state with **calibrated probabilities**. One state, several questions, one prefill:

| type | answer | good for |
|---|---|---|
| `noul` | P(yes) | checks and filters ("does the drawing contain letters?") |
| `choice` | a probability for every option | classifying, picking, guessing, scoring a guess |
| `score` | E[level] over ordered levels, plus each level's probability | ratings, meters, "how much" |

Decisions are fast and deterministic, and the probabilities make good game mechanics: a score, a meter, an odds
roll. Most of the fun in these demos comes from them.

```bash
curl -s localhost:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "zoro-1",
  "state": {"excuse": "my cat deleted my homework"},
  "questions": {
    "believable": {"type": "score", "instructions": "How believable is this excuse?",
                   "criteria": ["obviously fake", "hard to believe", "could be true", "believable", "definitely true"]},
    "funny": {"type": "noul", "instructions": "Is the excuse funny?"}},
  "reasoning": false}'
```

- Add an image with `"images": ["data:image/png;base64,…"]` (one per request, up to 448×448 by default).
- `choice` options are `"criteria": {"key": "description"}`; the description is optional. Short English descriptions
  help when the options are in another language.
- The request format is defined in [`inference/zoro-1/decide_request.py`](../../inference/zoro-1/decide_request.py);
  test your requests against it (the template shows how).
- For chat, send `"chat_template_kwargs": {"enable_thinking": false}` when you want a quick reply.

## Run Zoro-1

Zoro-1 is gated on Hugging Face: request access on the [model page](https://huggingface.co/trillionlabs/Zoro-1),
then `hf auth login`.

**On a Mac (Apple Silicon, 16 GB+)**, with the MLX server; chat and decisions on port 8080:

```bash
cd inference/zoro-1/mlx
uv sync
uv run hf auth login
uv run trida-mlx-convert --model trillionlabs/Zoro-1 --out ~/models/Zoro-1-mlx-q8 --bits 8   # once
uv run trida-mlx-server --model ~/models/Zoro-1-mlx-q8 --port 8080
```

It handles one request at a time: fine for a demo with a few people.

**On a GPU**, with SGLang; decisions and chat on port 30001, with concurrent requests batched:

```bash
cd inference/zoro-1
pip install "sglang>=0.5.22" fastapi uvicorn
python serve.py --mode decide --host 127.0.0.1 --port 30001
```

This server has no API key, so keep it on localhost. If the GPU is another machine, reach it through an SSH
tunnel (`ssh -N -L 30001:127.0.0.1:30001 GPU_HOST`) and point the demo at `http://127.0.0.1:30001`.

Check it's up: `curl -s localhost:8080/v1/models` (or `:30001`). See [`inference/zoro-1`](../../inference/zoro-1/)
for details.

## Add a demo

The full guide is in **[CONTRIBUTING.md](CONTRIBUTING.md)** ([한국어](CONTRIBUTING.ko.md)). In short:

1. Copy [`_template/`](_template/) to `demo/zoro-1/<your-demo>` and rename the package.
2. Keep it self-contained, talking to Zoro-1 over HTTP (`--model-url` / `ZORO_URL`), with a `--mock` mode and tests
   that pass without a model.
3. Write the README with the template's sections (description, screenshot, how it uses Zoro-1, run, files, test).
4. Add a row to the table at the top of this page and open a pull request with the
   [checklist](CONTRIBUTING.md#7-pull-request-checklist).

Ideas that suit Zoro-1: party games scored by the decision head, "rate my ___" meters, a judge for debates, a
picture-guessing bot, a sorting hat, interactive fiction where choices are scored, a classroom quiz host.
