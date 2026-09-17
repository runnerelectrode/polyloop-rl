import json
from pathlib import Path

from polyloop.harness.trace_rows import iter_rows


def _trace(tmp_path: Path) -> Path:
    recs = [
        {"session": "s1", "turn": 0, "messages": [{"role": "user", "content": "hi"}], "response": "a",
         "next_state": {"role": "user", "content": "no, the other one"}},
        {"session": "s1", "turn": 1, "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a"},
                                                  {"role": "user", "content": "no, the other one"}], "response": "b", "next_state": None},
        {"session": "s2", "turn": 0, "messages": [{"role": "user", "content": "yo"}], "response": "c", "next_state": None},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return p


def test_next_state_hint_only(tmp_path):
    rows = list(iter_rows([_trace(tmp_path)]))
    assert [(r["session"], r["turn"]) for r in rows] == [("s1", 0)]
    assert "Hindsight from the user's reply" in rows[0]["hint"]


def test_session_hint_applies_to_every_turn_and_last_turn(tmp_path):
    rows = list(iter_rows([_trace(tmp_path)], session_hints={"s1": "Judge: the caller's name was never confirmed."}))
    assert [(r["session"], r["turn"]) for r in rows] == [("s1", 0), ("s1", 1)]
    assert rows[0]["hint"].endswith("Judge: the caller's name was never confirmed.")
    assert rows[1]["hint"] == "Judge: the caller's name was never confirmed."


def test_excluded_sessions_are_skipped(tmp_path):
    rows = list(iter_rows([_trace(tmp_path)], session_hints={"s1": "x", "s2": "y"}, skip_sessions={"s1"}))
    assert [r["session"] for r in rows] == ["s2"]
