import json

from polyloop.harness.trace_rows import iter_rows, write_rows


def _trace(tmp_path, recs):
    p = tmp_path / "2026-09-17.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return p


def test_session_hint_appended_and_last_turn_kept(tmp_path):
    recs = [
        {"session": "simA", "turn": 0, "messages": [{"role": "user", "content": "hi, I need a cleaning"}],
         "response": "Sure. Your name?", "next_state": {"role": "user", "content": "Maria Lopez"}},
        {"session": "simA", "turn": 1, "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Sure. Your name?"},
                                                     {"role": "user", "content": "Maria Lopez"}],
         "response": "Booked for Tuesday.", "next_state": None, "session_done": True},
        {"session": "simB", "turn": 0, "messages": [{"role": "user", "content": "braces?"}], "response": "We do braces!",
         "next_state": {"role": "user", "content": "great"}},
    ]
    p = _trace(tmp_path, recs)
    hints = {"simA": "Hindsight from the evaluator: this conversation failed (score 0.00).\nNo phone number collected."}
    rows = list(iter_rows([p], session_hints=hints))
    by = {(r["session"], r["turn"]): r for r in rows}
    assert set(by) == {("simA", 0), ("simA", 1), ("simB", 0)}
    assert by[("simA", 0)]["hint"].startswith("Hindsight from the user's reply to this step:\nMaria Lopez")
    assert by[("simA", 0)]["hint"].endswith("No phone number collected.")
    # last turn: no next state, judge hint alone
    assert by[("simA", 1)]["hint"] == hints["simA"]
    # unhinted session keeps the plain next-state hint
    assert by[("simB", 0)]["hint"] == "Hindsight from the user's reply to this step:\ngreat"


def test_last_turn_without_any_hint_is_dropped(tmp_path):
    p = _trace(tmp_path, [{"session": "s", "turn": 0, "messages": [{"role": "user", "content": "x"}], "response": "y", "next_state": None}])
    assert list(iter_rows([p])) == []
    out = tmp_path / "rows.jsonl"
    assert write_rows([p], out) == 0
    assert write_rows([p], out, session_hints={"s": "verdict"}) == 1
    assert json.loads(out.read_text())["hint"] == "verdict"
