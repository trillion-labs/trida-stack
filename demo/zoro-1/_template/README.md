# My Demo

<!-- Copy this folder to demo/zoro-1/<your-demo>, rename the package (my_demo) and the script in pyproject.toml,
     then replace this README. Keep the section headings: every demo README has them. -->

One or two sentences: what it is and why it's fun or useful.

![screenshot](docs/screenshot.png)  <!-- a screenshot or GIF, recorded with the real model if you can -->

## How it uses Zoro-1

Which endpoints and question types, and why. Show one real request.
This template's example: a `score` question (how believable is an excuse, 5 levels) and a `noul` (is it funny?) in
one `/v1/systemone` call, then one `/v1/chat/completions` reply.

## Run

Start Zoro-1 first ([Run Zoro-1](../README.md#run-zoro-1)), then:

```sh
cd demo/zoro-1/my-demo
uv run my-demo --model-url http://127.0.0.1:8080 "my cat deleted my homework"
uv run my-demo --mock "my cat deleted my homework"        # no model needed
```

## Files

| file | what |
|---|---|
| `my_demo/zoro.py` | tiny Zoro-1 client (decide, chat) with a mock |
| `my_demo/main.py` | the demo |
| `tests/` | mock-mode test + a check against Zoro-1's request parser |

## Test

```sh
uv run --extra test pytest -q
PYTHONPATH=../../../inference/zoro-1 uv run --extra test pytest -q
```
