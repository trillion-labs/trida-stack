"""BFCL v4 model handler for Trida-7B.

Trida-7B's chat template is Qwen-style ChatML with `<tools>` / `<tool_call>\n{...}\n</tool_call>`
function-calling — identical to what `QwenFCHandler` already formats and parses. So the handler is a
thin subclass: it inherits `_format_prompt` (renders the tool system block + ChatML), the
`<tool_call>` extraction, and `decode_ast` / `decode_execute` unchanged.

Generation itself is done by our OpenAI-compatible server (`serve_trida_openai.py`) over
`/v1/completions`; this handler only formats prompts and parses text (BFCL prompting path).

Symlinked into the BFCL package as `bfcl_eval/model_handler/local_inference/trida.py` by
`register_bfcl.sh`, and registered in `model_config.py`.
"""
import json
import os
import re
import sys

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from overrides import override

# make inference.bd_generate importable from the BFCL venv (for load_tokenizer_compat)
_REPO = "$SCRATCH/trida-stack"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


class TridaFCHandler(QwenFCHandler):
    _trida_tok = None  # cached tokenizer of the ACTUAL served checkpoint (not the released Trida-7B)

    def _trida_tokenizer(self):
        if TridaFCHandler._trida_tok is None:
            from inference.bd_generate import load_tokenizer_compat
            ckpt = os.environ.get("TRIDA_CKPT") or os.environ.get("CKPT")
            if not ckpt:
                raise RuntimeError("TRIDA_CKPT/CKPT env not set; needed to load the served model's template")
            TridaFCHandler._trida_tok = load_tokenizer_compat(ckpt)
        return TridaFCHandler._trida_tok

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, dtype=dtype, **kwargs)
        # Trida uses the same tool/tokenizer format as Qwen; keep the HF name for tokenizer loading.
        self.model_name_huggingface = model_name

    @override
    def _format_prompt(self, messages, function):
        # Render with the ACTUAL served checkpoint's chat template + tools — i.e. exactly the format
        # the model trained on (build_sample -> apply_chat_template(messages, tools=...)). The stock
        # QwenFCHandler format is off-distribution for this block-diffusion checkpoint and makes the
        # decode collapse into repetition (BFCL scored 0 across the board); this fixes that. The
        # trailing "<think>\n" anchors the reasoning start (same prefill the chat route/GSM8K use).
        try:
            tok = self._trida_tokenizer()
            tools = [{"type": "function", "function": f} for f in function] if function else None
            prompt = tok.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=False)
            return prompt + "<think>\n"
        except Exception as e:
            print(f"[trida_handler] apply_chat_template failed ({e}); falling back to QwenFC format", flush=True)
            return super()._format_prompt(messages, function)

    @staticmethod
    def _loads_balanced(s):
        """json.loads tolerant of the model's common malformed-JSON artifact: a spurious extra
        trailing brace (e.g. ``{"name": ..., "arguments": {...}}}``). Falls back to extracting the
        first COMPLETE balanced ``{...}`` object (ignoring any trailing junk) when a strict parse
        fails. Returns the parsed object or ``None``."""
        s = s.strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        depth = start = 0
        in_str = esc = False
        start = None
        for i, ch in enumerate(s):
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        try:
                            return json.loads(s[start:i + 1])
                        except Exception:
                            return None
        return None

    @staticmethod
    @override
    def _extract_tool_calls(input_string):
        """Robust superset of Qwen's `<tool_call>` parser. Beyond the inherited normalization
        (missing ``arguments`` / ``arguments`` as a JSON string), this re-parses each
        ``<tool_call>`` block with a brace-balancing loader so the model's frequent extra-trailing-
        brace artifact (``...}}}``) — which a strict ``json.loads`` drops, silently zeroing the
        call — is recovered. Normalizes every call to ``{"name": str, "arguments": dict}``."""
        raw = []
        for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", input_string, re.DOTALL):
            obj = TridaFCHandler._loads_balanced(block)
            if isinstance(obj, dict):
                raw.append(obj); continue
            # qwen3_coder XML: <function=NAME><parameter=K>V</parameter>...</function>
            for fm in re.finditer(r"<function=([^>\s]+)\s*>(.*?)</function>", block, re.DOTALL):
                a = {}
                for k, v in re.findall(r"<parameter=([^>]+?)\s*>\s*(.*?)\s*</parameter>", fm.group(2), re.DOTALL):
                    v = v.strip()
                    try: v = json.loads(v)
                    except Exception: pass
                    a[k.strip()] = v
                raw.append({"name": fm.group(1).strip(), "arguments": a})
        if not raw:  # no tag-wrapped calls parsed -> defer to the inherited extractor
            raw = QwenFCHandler._extract_tool_calls(input_string)
        out = []
        for c in raw:
            if not isinstance(c, dict) or "name" not in c:
                continue
            args = c.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            out.append({"name": c["name"], "arguments": args})
        return out
