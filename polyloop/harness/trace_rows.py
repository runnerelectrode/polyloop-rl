"""Proxy traces -> on-policy self-distillation rows.

Each recorded assistant turn with a next state becomes one row: the conversation prefix up to
and including the message the assistant answered, plus a `hint` built from the next state (the
tool or test output, or the user's next message). The student re-samples the turn on-policy;
the teacher is the same weights with the hint appended to the last user message (rlcli's
HintedTeacher). The logged response is not trained on directly, so token exactness of the
capture is not load-bearing here; it matters for the GRPO path, which uses sandbox episodes.

Row format is rlcli's OPSD prompts JSONL: {"messages": [...system/user/assistant...], "hint": str}.
Tool-role messages are folded into user text because rlcli's loader keeps only
system/user/assistant.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

HINT_PREFIX = {
    "tool": "Hindsight from the tool output that followed this step:",
    "user": "Hindsight from the user's reply to this step:",
}


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("text", "input_text", "output_text"):
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return json.dumps(content) if content is not None else ""


def _fold(messages: list[dict]) -> list[dict]:
    """Map tool/function messages into user text so the prefix renders in any chat template."""
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role in ("tool", "function"):
            out.append({"role": "user", "content": f"[tool output]\n{_text(m.get('content'))}"})
        elif role == "assistant":
            content = _text(m.get("content"))
            calls = m.get("tool_calls") or []
            if calls:
                content += "\n" + "\n".join(json.dumps(c.get("function", c)) for c in calls)
            out.append({"role": "assistant", "content": content})
        elif role in ("system", "user"):
            out.append({"role": role, "content": _text(m.get("content"))})
    return out


def _hint(next_state: dict, max_chars: int) -> str | None:
    role = next_state.get("role", "user")
    text = _text(next_state.get("content")).strip()
    if not text:
        return None
    if len(text) > max_chars:
        text = text[: max_chars // 2] + "\n...\n" + text[-max_chars // 2 :]
    return f"{HINT_PREFIX.get(role, HINT_PREFIX['user'])}\n{text}"


def iter_rows(trace_files: Iterable[Path], *, max_hint_chars: int = 2000, min_prefix_turns: int = 1,
              skip_sessions: set[str] | None = None) -> Iterable[dict]:
    for path in trace_files:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if skip_sessions and rec.get("session") in skip_sessions:
                continue
            ns = rec.get("next_state")
            if not ns:
                continue
            hint = _hint(ns, max_hint_chars)
            if not hint:
                continue
            prefix = _fold(rec.get("messages") or [])
            if not any(m["role"] == "user" and m["content"].strip() for m in prefix):
                continue
            if sum(1 for m in prefix if m["role"] == "user") < min_prefix_turns:
                continue
            yield {"messages": prefix, "hint": hint, "session": rec["session"], "turn": rec["turn"]}


def write_rows(trace_files: Iterable[Path], out: Path, **kw) -> int:
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in iter_rows(trace_files, **kw):
            f.write(json.dumps(row) + "\n")
            n += 1
    return n
