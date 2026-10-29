"""Output parsing for chat responses: ``<think>`` reasoning split, Qwen-style tool calls
(JSON ``<tool_call>{...}</tool_call>`` and qwen3-coder XML ``<function=..><parameter=..>``),
and a streaming splitter that routes deltas to reasoning / content / tool-call buffers."""
from __future__ import annotations

import json
import re
import uuid

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TC_OPEN, TC_CLOSE = "<tool_call>", "</tool_call>"


def _loads_balanced(s: str):
    """json.loads the first balanced {...} object in s (tolerates trailing junk like '}}}')."""
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(s[start : i + 1])
                except Exception:
                    return None
    return None


def _param_types(tools) -> dict:
    """{tool_name: {param: json-schema type}} from an OpenAI ``tools`` list."""
    out = {}
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        props = ((fn.get("parameters") or {}).get("properties") or {}) if isinstance(fn, dict) else {}
        out[fn.get("name")] = {k: (v.get("type") if isinstance(v, dict) else None) for k, v in props.items()}
    return out


def _coerce(value: str, typ):
    """qwen3-coder XML parameters are raw text: keep strings as strings, JSON-decode the rest."""
    if typ == "string" or (isinstance(typ, list) and "string" in typ and len(typ) == 1):
        return value
    try:
        return json.loads(value)
    except Exception:
        if typ in ("integer", "number"):
            try:
                return int(value) if typ == "integer" else float(value)
            except ValueError:
                return value
        if typ == "boolean" and value.lower() in ("true", "false"):
            return value.lower() == "true"
        return value


def parse_tool_calls(text: str, tools=None) -> tuple[str, list]:
    """Returns (content_without_tool_calls, [{"name", "arguments"(dict)}])."""
    types = _param_types(tools)
    calls = []
    for block in re.findall(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", text, re.DOTALL):
        obj = _loads_balanced(block)
        if isinstance(obj, dict) and "name" in obj:
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            calls.append({"name": obj["name"], "arguments": args if isinstance(args, dict) else {}})
            continue
        for fm in re.finditer(r"<function=([^>\s]+)\s*>(.*?)(?:</function>|$)", block, re.DOTALL):
            name, a = fm.group(1).strip(), {}
            ptypes = types.get(name, {})
            for k, v in re.findall(r"<parameter=([^>]+?)\s*>\n?(.*?)\n?</parameter>", fm.group(2), re.DOTALL):
                k = k.strip()
                typ = ptypes.get(k)
                a[k] = _coerce(v if typ == "string" else v.strip(), typ)
            calls.append({"name": name, "arguments": a})
    content = re.sub(r"<tool_call>.*?(?:</tool_call>|$)", "", text, flags=re.DOTALL).strip()
    return content, calls


def split_reasoning(text: str, started_in_think: bool) -> tuple[str, str]:
    """(reasoning, content). ``started_in_think``: the prompt ended inside an open <think>."""
    if not started_in_think:
        if text.lstrip().startswith(THINK_OPEN):
            text = text.lstrip()[len(THINK_OPEN):]
            started_in_think = True
        else:
            return "", text
    if THINK_CLOSE in text:
        r, c = text.split(THINK_CLOSE, 1)
        return r.strip(), c.lstrip("\n")
    return text.strip(), ""


def to_openai_tool_calls(calls: list) -> list:
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)},
        }
        for c in calls
    ]


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of text that is a proper prefix of tag."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-n:]):
            return n
    return 0


class StreamSplitter:
    """Incrementally route generated text into reasoning / content deltas; tool-call blocks
    are withheld and returned parsed by ``finish``."""

    def __init__(self, started_in_think: bool, parse_tools: bool):
        self.phase = "maybe_think" if not started_in_think else "think"
        self.parse_tools = parse_tools
        self.buf = ""
        self.tool_text = ""
        self.in_tool = False
        self.closed_tools = ""
        self.content_started = False

    def _content(self, out, text):
        if not self.content_started:
            text = text.lstrip()
            if not text:
                return
            self.content_started = True
        out.append(("content", text))

    def feed(self, delta: str) -> list[tuple[str, str]]:
        """Returns [(kind, text)] with kind in {'reasoning', 'content'}."""
        self.buf += delta
        out = []
        while True:
            if self.phase == "maybe_think":
                s = self.buf.lstrip()
                if not s:
                    return out
                if s.startswith(THINK_OPEN):
                    self.buf = s[len(THINK_OPEN):]
                    self.phase = "think"
                    continue
                if THINK_OPEN.startswith(s):
                    return out  # could still become <think>
                self.phase = "content"
                continue
            if self.phase == "think":
                i = self.buf.find(THINK_CLOSE)
                if i >= 0:
                    if self.buf[:i]:
                        out.append(("reasoning", self.buf[:i]))
                    self.buf = self.buf[i + len(THINK_CLOSE):].lstrip("\n")
                    self.phase = "content"
                    continue
                keep = _partial_suffix(self.buf, THINK_CLOSE)
                emit = self.buf[: len(self.buf) - keep]
                if emit:
                    out.append(("reasoning", emit))
                self.buf = self.buf[len(emit):]
                return out
            # content
            if self.in_tool:
                self.tool_text += self.buf
                self.buf = ""
                j = self.tool_text.find(TC_CLOSE)
                if j >= 0:
                    rest = self.tool_text[j + len(TC_CLOSE):]
                    self.tool_text = self.tool_text[: j + len(TC_CLOSE)] + "\n"
                    self.in_tool = False
                    self.buf = rest
                    self.closed_tools += self.tool_text
                    self.tool_text = ""
                    continue
                return out
            if self.parse_tools:
                i = self.buf.find(TC_OPEN)
                if i >= 0:
                    if self.buf[:i]:
                        self._content(out, self.buf[:i])
                    self.in_tool = True
                    self.tool_text = TC_OPEN
                    self.buf = self.buf[i + len(TC_OPEN):]
                    continue
                keep = _partial_suffix(self.buf, TC_OPEN)
            else:
                keep = 0
            emit = self.buf[: len(self.buf) - keep]
            if emit:
                self._content(out, emit)
            self.buf = self.buf[len(emit):]
            return out

    def finish(self, tools=None) -> tuple[list[tuple[str, str]], list]:
        out = []
        if self.phase == "think" and self.buf:
            out.append(("reasoning", self.buf))
        elif self.buf and not self.in_tool:
            self._content(out, self.buf)
        self.buf = ""
        tool_src = self.closed_tools + (self.tool_text if self.in_tool else "")
        calls = parse_tool_calls(tool_src, tools)[1] if tool_src else []
        return out, calls
