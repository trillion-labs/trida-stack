# Excuse Judge (notebook demo template)

<!-- Copy this folder to demo/zoro-1/<your-demo>, rename the notebook, and replace this README.
     A notebook demo's README can be short: these four sections are enough. -->

Give Zoro-1 an excuse for late homework: the decision head rates how believable and how funny it is, and the chat
endpoint answers as the teacher. One notebook, one dependency (`httpx`).

## How it uses Zoro-1

One `/v1/systemone` call with a `score` question (5 levels of believability) and a `noul` (is it funny?), then one
`/v1/chat/completions` reply with thinking off.

## Run

Start Zoro-1 ([Run Zoro-1](../README.md#run-zoro-1)), then open `excuse_judge.ipynb` and run all cells. It talks to
`ZORO_URL` (default `http://127.0.0.1:8080`). Without a model, set `MOCK = True` in the setup cell.

```sh
cd demo/zoro-1/<your-demo>
uv run --no-project --with jupyter --with httpx jupyter lab excuse_judge.ipynb
```

## Test

The notebook must run top to bottom in mock mode:

```sh
ZORO_MOCK=1 uv run --no-project --with jupyter --with httpx jupyter nbconvert --to notebook --execute excuse_judge.ipynb --output /tmp/out.ipynb
```
