"""Simulated users for Zoro-1 Chat: an LLM plays a person using the app on this Mac, to collect more real
trajectories (through the app's own tool routing, tools, reply checks and logs).

    trida-mlx-simuser --episodes 20

The user LLM is any OpenAI-compatible server, set in the environment or a .env file (the current directory's,
then the repository root's; see .env.example): SIMUSER_BASE_URL (required, e.g. http://<host>:30000/v1),
SIMUSER_MODEL (default: the first model the server lists), SIMUSER_API_KEY (if the server needs one).

Needs the app running with Settings → Agent → "Allow automation" on (it reads the app's token from
~/Library/Application Support/Zoro-1 Chat/automation-token). Each episode: the user LLM invents a person and a
goal for one scenario (translation of pasted text, a Korean request, a file task, hard math, a pick-one
decision, small talk, ...), then writes one message at a time, reading Zoro-1's actual replies, pushing back
when the reply is wrong (wrong language, task not done, a claimed action that did not happen), and ends with
[DONE]. Approvals still show in the app as usual. The app records everything (chats, request logs); this
writes an episode log too (--out), and Server → Export Training Data… turns it all into JSONL.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TOKEN_FILE = Path.home() / "Library/Application Support/Zoro-1 Chat/automation-token"

SCENARIOS = [
    ("translate_paste", "paste a paragraph (2-6 sentences: an email, a work update, a README excerpt that mentions "
     "file names like config.yaml or pyproject.toml, a news snippet) and ask to translate it (Korean <-> English); "
     "sometimes paste first and only ask 'translate this' in the next message"),
    ("summarize_paste", "paste a longer text (an abstract, meeting notes, a product description) and ask for a "
     "summary in a specific language, often a different language than the text"),
    ("korean_chat", "write entirely in casual Korean: ask for advice, a joke, a recipe, an explanation; expect "
     "Korean answers"),
    ("identity", "ask who the assistant is, who made it, whether it is ChatGPT/Qwen/Claude, what it can do"),
    ("files", "ask about files on your Mac's Desktop (the assistant's workspace): what is there, find PDFs or "
     "images, read a text file you name after seeing the list, search inside files"),
    ("write_file", "ask the assistant to write something (a poem, a todo list, notes) and save it as a file in "
     "the workspace, then check that it exists"),
    ("terminal", "ask for things a terminal answers: disk usage of a folder, git status of a project folder, "
     "the Python version, counting lines in files"),
    ("math_easy", "ask quick calculations or unit conversions, or the current date and time"),
    ("math_hard", "pose a hard multi-step math or logic problem (competition style) and follow up if the answer "
     "is unclear or unjustified"),
    ("decide", "describe a situation and ask the assistant to pick between named options (which laptop, which "
     "plan, which candidate) or give a yes/no judgement with how sure it is"),
    ("ambiguous", "send something ambiguous: pasted text without any instruction, 'this' referring to an "
     "earlier reply, a vague request; clarify only when asked"),
    ("long_work", "work on one task over many turns (draft an email, refine it, shorten it, translate it, "
     "change the tone), referring back to earlier versions"),
]
LANGS = ["Korean", "English"]   # the simulated users write only in these

PERSONA_PROMPT = """You are designing a realistic test user for an AI chat app.
The app: "Zoro-1 Chat", a chat app on the user's Mac with an on-device AI assistant (Zoro-1). It can chat, see
attached images, and in agent mode use tools: list/read/find/search files on the Desktop, run terminal commands
(the user approves anything that changes something), tell the time, calculate, and make calibrated decisions.

Scenario: {scenario}
The user writes only in {lang}. Any text they paste is in Korean or English, nothing else.

Invent one person and one concrete goal for this scenario. Reply with JSON only:
{{"persona": "<who they are, 1-2 sentences>", "goal": "<what they want to get done, concretely>",
 "style": "<how they write: length, tone, typos, patience>", "first_message": "<their first message, exactly>"}}"""

USER_PROMPT = """You are role-playing a human using "Zoro-1 Chat", an AI chat app on their Mac. Stay in character.
Who you are: {persona}
What you want: {goal}
How you write: {style}
Language: write every message only in {lang}. Text you paste may be Korean or English; never use any other language.

Rules:
- Write ONLY your next message to the assistant, exactly as you would type it. No quotes, no labels, no notes.
- React to what the assistant actually said. If it got something wrong (answered in the wrong language, did not do
  what you asked, said it did something it did not, repeated itself, called a tool for no reason, went off topic),
  push back the way a real person would.
- Messages in brackets like [the assistant ran: ...] describe what you saw happen in the app; you did not write them.
- When your goal is reached, or you give up, write your last message (or nothing more to say) followed by [DONE].
- Never mention that you are simulated, an AI, or testing anything."""


def load_dotenv(paths) -> None:
    """KEY=VALUE lines from the first files that exist, without overriding variables already set."""
    for p in paths:
        try:
            lines = Path(p).read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            v = v.strip().strip('"').strip("'")
            if k and v:
                os.environ.setdefault(k, v)


def dotenv_paths():
    here = Path(__file__).resolve()
    return [Path.cwd() / ".env"] + ([here.parents[3] / ".env"] if len(here.parents) > 3 else [])


def http_json(url, body=None, headers=None, timeout=3600):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})},
                                 method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class UserLLM:
    def __init__(self, base, model, api_key, temperature):
        self.base, self.temperature = base.rstrip("/"), temperature
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model or http_json(self.base + "/models", headers=self.headers, timeout=30)["data"][0]["id"]

    def chat(self, messages, max_tokens=1200) -> str:
        r = http_json(self.base + "/chat/completions", {
            "model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}, self.headers, timeout=600)
        text = r["choices"][0]["message"].get("content") or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


class App:
    def __init__(self, base):
        self.base = base.rstrip("/")
        try:
            token = TOKEN_FILE.read_text().strip()
        except OSError:
            sys.exit(f"no token at {TOKEN_FILE}: turn on Settings → Agent → Allow automation in Zoro-1 Chat")
        self.headers = {"Authorization": f"Bearer {token}"}

    def status(self):
        return http_json(self.base + "/status", headers=self.headers, timeout=10)

    def new_chat(self, agent_mode):
        return http_json(self.base + "/chats", {"agent_mode": agent_mode}, self.headers, timeout=30)["id"]

    def send(self, chat, text):
        return http_json(f"{self.base}/chats/{chat}/messages", {"text": text}, self.headers)["messages"]


def visible(messages) -> str:
    """What the person sees after a turn: tool activity, then the final reply."""
    parts, final = [], ""
    for m in messages:
        if m.get("role") != "assistant" or m.get("superseded"):
            continue
        for r in m.get("tool_runs") or []:
            res = (r.get("result") or "").strip().replace("\n", " ")
            parts.append(f"[the assistant ran: {r['name']}({r['arguments'][:200]}) -> {r.get('status')}: {res[:200]}]")
        if m.get("content"):
            final = m["content"]
        if m.get("error"):
            parts.append(f"[error shown: {m['error']}]")
    return "\n".join(parts + ([final] if final else ["[no reply]"]))


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def episode(user: UserLLM, app: App, scenario, lang, max_turns, log):
    key, desc = scenario
    spec = None
    for _ in range(3):
        try:
            spec = parse_json(user.chat([{"role": "user", "content": PERSONA_PROMPT.format(scenario=desc, lang=lang)}]))
            if spec and spec.get("first_message"):
                break
        except (json.JSONDecodeError, urllib.error.URLError, KeyError):
            spec = None
    if not spec:
        print(f"  [{key}] could not make a persona; skipped", flush=True)
        return
    agent = key in ("files", "write_file", "terminal", "math_easy", "decide", "ambiguous") or random.random() < 0.5
    chat = app.new_chat(agent)
    sys_prompt = USER_PROMPT.format(persona=spec.get("persona", ""), goal=spec.get("goal", ""),
                                    style=spec.get("style", ""), lang=lang)
    convo = [{"role": "system", "content": sys_prompt}]   # roles from the user LLM's side
    msg = spec["first_message"].strip()
    rec = {"scenario": key, "lang": lang, "agent_mode": agent, "chat_id": chat, "spec": spec, "turns": []}
    print(f"  [{key}] {spec.get('goal', '')[:90]}", flush=True)
    for turn in range(max_turns):
        done = "[DONE]" in msg
        msg = msg.replace("[DONE]", "").strip()
        if not msg:
            break
        print(f"    user: {msg[:100]!r}", flush=True)
        t0 = time.time()
        try:
            added = app.send(chat, msg)
        except urllib.error.HTTPError as e:
            print(f"    app error {e.code}: {e.read()[:200]!r}", flush=True)
            break
        seen = visible(added)
        print(f"    zoro ({time.time() - t0:.0f}s): {seen[:100]!r}", flush=True)
        rec["turns"].append({"user": msg, "app_messages": added, "seconds": round(time.time() - t0, 1)})
        if done:
            break
        convo += [{"role": "assistant", "content": msg}, {"role": "user", "content": seen}]
        msg = user.chat(convo)
    log.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    load_dotenv(dotenv_paths())
    ap.add_argument("--user-base", default=os.environ.get("SIMUSER_BASE_URL"),
                    help="OpenAI-compatible server for the user LLM (default: $SIMUSER_BASE_URL)")
    ap.add_argument("--user-model", default=os.environ.get("SIMUSER_MODEL"),
                    help="default: $SIMUSER_MODEL, else the first model the server lists")
    ap.add_argument("--user-api-key", default=os.environ.get("SIMUSER_API_KEY"))
    ap.add_argument("--user-temperature", type=float, default=0.9)
    ap.add_argument("--app", default="http://127.0.0.1:8765")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--scenario", action="append", default=None, help="only these scenarios (repeatable): "
                    + ", ".join(k for k, _ in SCENARIOS))
    ap.add_argument("--out", default=str(Path.home() / "Library/Logs/Zoro-1 Chat/simuser"))
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    if not a.user_base:
        ap.error("no user LLM: set SIMUSER_BASE_URL (environment or .env, see .env.example) or pass --user-base")
    random.seed(a.seed)
    app = App(a.app)
    st = app.status()
    if not st.get("ready"):
        sys.exit("the app's model server is not ready yet")
    user = UserLLM(a.user_base, a.user_model, a.user_api_key, a.user_temperature)
    pool = [s for s in SCENARIOS if not a.scenario or s[0] in a.scenario]
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"episodes-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    print(f"user LLM: {user.model} @ {a.user_base}; app model: {st.get('model')}; log: {path}", flush=True)
    with open(path, "w", encoding="utf-8") as log:
        for i in range(a.episodes):
            print(f"episode {i + 1}/{a.episodes}", flush=True)
            episode(user, app, random.choice(pool), random.choice(LANGS), a.max_turns, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
