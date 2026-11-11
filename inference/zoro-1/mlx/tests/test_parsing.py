from trida_mlx.parsing import StreamSplitter, parse_tool_calls

TOOLS = [{"type": "function", "function": {"name": n, "parameters": {"type": "object", "properties": {
    "command": {"type": "string"}, "timeout": {"type": "integer"}}}}} for n in ("now", "run_shell")]

# what Zoro-1 wrote: one wrapped call, then calls without the <tool_call> opener
RAW = """<tool_call>
<function=now>
</function>
</tool_call>
<function=run_shell>
<parameter=command>
curl -s https://a.example | grep x
</parameter>
<parameter=timeout>
30
</parameter>
</function>
</tool_call>
<function=run_shell>
<parameter=command>
echo hi
</parameter>
</function>
</tool_call>"""


def test_bare_function_blocks_are_calls():
    content, calls = parse_tool_calls(RAW, TOOLS)
    assert [c["name"] for c in calls] == ["now", "run_shell", "run_shell"]
    assert calls[1]["arguments"] == {"command": "curl -s https://a.example | grep x", "timeout": 30}
    assert content == ""


def test_bare_blocks_need_a_known_tool_name():
    text = "Use it like <function=frobnicate>\n</function> in your prompt."
    content, calls = parse_tool_calls(text, TOOLS)
    assert calls == [] and "frobnicate" in content


def test_text_before_calls_is_kept():
    content, calls = parse_tool_calls("Let me check.\n" + RAW, TOOLS)
    assert content == "Let me check." and len(calls) == 3


def test_stream_splitter_catches_bare_calls():
    sp = StreamSplitter(started_in_think=False, parse_tools=True)
    out = []
    for i in range(0, len(RAW), 7):          # odd chunk size: tags split across deltas
        out += sp.feed(RAW[i:i + 7])
    rest, calls = sp.finish(TOOLS)
    out += rest
    assert "".join(t for k, t in out if k == "content").strip() == ""
    assert [c["name"] for c in calls] == ["now", "run_shell", "run_shell"]
