# Contributing a Zoro-1 demo

English · [한국어](CONTRIBUTING.ko.md)

Thanks for building something with Zoro-1. This page covers how to propose a demo, what a demo folder must
contain, and what reviewers check before merging. The short version is in the [README](README.md#add-a-demo); this
is the full one.

## 1. Before you start

- **Look at what exists.** The table in the [README](README.md) lists the demos. A new demo should do something
  the others don't, or do it for a different audience.
- **Big or unsure? Ask first.** Open a [Discussion](https://github.com/trillion-labs/trida-stack/discussions/categories/ideas)
  or an issue with one paragraph: what it does, who it's for, which Zoro-1 features it uses. Small demos can go
  straight to a pull request.
- **Read the repository's [CONTRIBUTING.md](../../CONTRIBUTING.md)** for the general rules (license, the
  noncommercial kernel caveat, pull request flow). Demos are Apache-2.0 like the rest of the repository.

What makes a good Zoro-1 demo:

- It **uses the decision head** where it can. Calibrated probabilities make great mechanics: a score, a meter, a
  ranking, a judge, odds. Chat alone is something every model does; decisions are what make Zoro-1 different.
- It's **fun or useful in under a minute**, for someone who has never seen it.
- It **runs on a laptop.** Zoro-1 on a Mac answers one request at a time, so design for a few users, not hundreds.
- It **plays to Zoro-1's strengths.** It's a 4B model: short, structured outputs work far better than long free
  text. If it struggles with something (we found Korean in-character chat and 삼행시 weak), design around it, for
  example by giving choice options short English descriptions.

## 2. Folder layout

Copy [`_template/`](_template/) to `demo/zoro-1/<your-demo>`:

```
demo/zoro-1/<your-demo>/
  README.md            required, see section 4
  pyproject.toml       required: name, Apache-2.0, dependencies, a console script, a "test" extra
  <your_package>/      your code
  tests/               required: pass in mock mode
  docs/                optional: screenshots or a GIF for the README
  uv.lock              recommended: commit it so `uv run` is reproducible
```

- **Name:** lowercase with dashes for the folder (`zoro-tarot`), underscores for the Python package (`zoro_tarot`).
  Rename the package and the script in `pyproject.toml`; leave nothing called `my_demo`.
- **Self-contained:** everything your demo needs is in its folder. Don't import from other demos or from the rest
  of the repository at runtime. (Tests may read Zoro-1's request parser, see section 5.) If you want another
  demo's code, copy it.
- **Python 3.10+, run with [uv](https://docs.astral.sh/uv/).** `uv run <script>` from the folder must work on a clean
  machine. Keep dependencies few and well-known.
- **No build step for the web UI** unless it really needs one. Plain HTML, CSS and JavaScript served by the Python
  app (as in zoro-draw) is easiest for others to read and change.

## 3. Talking to Zoro-1

- **Over HTTP only**, to a Zoro-1 server the user starts separately: `POST /v1/systemone` for decisions and
  `POST /v1/chat/completions` for chat. Don't load model weights inside a demo. This way the demo works with both
  the MLX server (Mac) and the SGLang server (GPU) without changes.
- **Take the server URL from `--model-url` or the `ZORO_URL` environment variable.** Use `zoro-1` as the default
  `model` field.
- **Mock mode is required:** a `--mock` flag that fakes Zoro-1 (deterministically, so tests are stable). Anyone
  must be able to try the demo, and CI must be able to test it, without a model or a GPU. `_template/my_demo/zoro.py`
  has a small client with a mock you can copy.
- **Fail visibly.** If the model server is down or a call fails, say so in the UI or the console (zoro-draw prints
  a warning at startup and shows an error on the round). Never fail silently with a default answer that looks
  like a real one.
- **Keep the model server private.** If your demo is a web app, the browser talks to your app and your app talks
  to Zoro-1. Don't expose the Zoro-1 port, and don't send secrets or hidden answers to the browser.
- **Concurrency:** the MLX server handles one request at a time. Add a `--concurrency` limit (default 1 is safe
  for the Mac; SGLang batches, so higher is fine there), and show a "Zoro is thinking…" state instead of freezing.
- **Decisions, practically:**
  - Put the stable part of the `state` first: the server caches state prefixes, so repeated calls get faster.
  - Ask several questions in one request when they share a state; they share one prefill.
  - Use `"reasoning": false` for anything interactive.
  - Images: one per request, as a data URL; 448×448 is the default cap. Give drawings a white background.
  - Option descriptions (`"criteria": {"key": "description"}`) are your best lever for accuracy.
- **Chat, practically:** send `"chat_template_kwargs": {"enable_thinking": false}` for quick replies, cap
  `max_tokens`, and handle a reply that's cut off or empty.

## 4. The README

Every demo README has these sections, in this order (copy them from the template):

1. **Title and one or two sentences**: what it is and why it's fun or useful.
2. **A screenshot or GIF.** Record it with the real model if you can; if it's from mock mode, say so under the image.
3. **How it uses Zoro-1**: the endpoints and question types, and one real request body.
4. **Run**: the commands for a real Zoro-1 server and for `--mock`. Link to [Run Zoro-1](README.md#run-zoro-1)
   instead of repeating how to start the model.
5. **Files**: a short table of what lives where.
6. **Test**: the test commands.

Write it in English. A Korean section or a `README.ko.md` alongside is welcome. Keep images small (a few hundred
KB at most).

## 5. Tests

- `uv run --extra test pytest -q` from your folder must pass **in mock mode**, without a model or network.
- Test your demo's logic (scoring, rules, parsing of model output), not just that it starts.
- **Check your requests against Zoro-1's own parser.** Add a test that imports
  [`inference/zoro-1/decide_request.py`](../../inference/zoro-1/decide_request.py) with `pytest.importorskip` and
  validates the bodies you send (see `_template/tests/test_demo.py`). Run it with:
  ```sh
  PYTHONPATH=../../../inference/zoro-1 uv run --extra test pytest -q
  ```
  This catches request-format mistakes before anyone runs your demo against a real server.
- Before opening the pull request, **run the demo once against a real Zoro-1 server** and say so in the PR.

## 6. Content and safety

Demos are public and often played in groups, at work or on stage.

- Anything players see (prompts, decks, characters, scenarios, example outputs) must be friendly to a general
  audience: no sexual content, no harassment, no hate, nothing targeting real private people. Edgy humor is fine;
  cruelty isn't.
- **Real people and brands:** don't impersonate real people or organizations, and don't ship pages that could be
  mistaken for a real service (logins, payments, official notices).
- **Chat personas** must be told to stay PG-13 and to deflect sexual requests, and must not claim to be human
  when sincerely asked outside the game's fiction.
- **Players' data:** don't collect more than the demo needs. If you log games (for example to improve prompts or
  build training data), make it off by default, say what's logged in the README, and never commit logs.
- **No secrets** (tokens, API keys), **no model weights or large datasets** in the folder.
- Copyrighted material: don't ship copyrighted images, lyrics or long text. Names and short references (a meme's
  name, a proverb) are fine.

## 7. Pull request checklist

Copy this into your PR description:

```markdown
- [ ] Folder `demo/zoro-1/<name>`, package and script renamed from the template
- [ ] Self-contained; runs with `uv run` from the folder
- [ ] `--mock` works without a model; `--model-url` / `ZORO_URL` for a real server
- [ ] Tests pass in mock mode; requests pass Zoro-1's parser (`PYTHONPATH=../../../inference/zoro-1`)
- [ ] Tried against a real Zoro-1 server (MLX or SGLang): ___
- [ ] README has: description, screenshot/GIF, how it uses Zoro-1, run, files, test
- [ ] Row added to the table in `demo/zoro-1/README.md`
- [ ] Content is safe for a general audience; no secrets, weights, logs or personal data
- [ ] Apache-2.0
```

## 8. Review and after merge

- A maintainer reviews within the response times in [COMMUNITY.md](../../COMMUNITY.md). Reviews check the
  checklist above, run the tests, and usually play the demo once.
- Small changes to someone else's demo (fixes, new decks or prompts) are welcome as pull requests; for bigger
  changes, ask the demo's author in an issue first.
- When Zoro-1's API changes, maintainers update the demos' requests; demos that stay broken for a long time may be
  moved out of the table until someone fixes them.
- Questions: [Discussions · Q&A](https://github.com/trillion-labs/trida-stack/discussions/categories/q-a).
