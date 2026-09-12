"""Append-only event log per cycle plus the loop's lineage (which checkpoint serves).

Layout under runs_dir/<loop>/:
  lineage.json               incumbent + promotion history (the only mutable file)
  pool_stats.json            per-task pass rates under each incumbent (filter cache)
  holdout_cache.json         per-incumbent held-out results (evaluate cache)
  cycles/<cycle_id>/
    events.jsonl             one JSON object per line, never rewritten
    state.json               stage outputs the next stage reads (rewritten atomically)
    receipt.json             promotion_receipt.v1, written by the gate
    train/ eval/ filter/     stage artifacts
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(data, indent=2, sort_keys=True))


class LoopStore:
    def __init__(self, runs_dir: Path, loop_name: str):
        self.root = runs_dir / loop_name
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "cycles").mkdir(exist_ok=True)

    # ---- lineage --------------------------------------------------------
    @property
    def lineage_path(self) -> Path:
        return self.root / "lineage.json"

    def lineage(self) -> dict:
        return read_json(self.lineage_path, {"incumbent": None, "history": []})

    def incumbent(self) -> dict | None:
        return self.lineage().get("incumbent")

    def set_incumbent(self, candidate: dict, receipt_path: str) -> None:
        lin = self.lineage()
        lin["history"].append({"ts": now_iso(), "promoted": candidate, "receipt": receipt_path})
        lin["incumbent"] = candidate
        write_json(self.lineage_path, lin)

    # ---- caches ---------------------------------------------------------
    def cache(self, name: str) -> dict:
        return read_json(self.root / f"{name}.json", {})

    def save_cache(self, name: str, data: dict) -> None:
        write_json(self.root / f"{name}.json", data)

    # ---- cycles ---------------------------------------------------------
    def new_cycle_id(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def cycles(self) -> list[str]:
        return sorted(p.name for p in (self.root / "cycles").iterdir() if p.is_dir())

    def cycle(self, cycle_id: str | None = None) -> "Cycle":
        return Cycle(self, cycle_id or self.new_cycle_id())


class Cycle:
    def __init__(self, store: LoopStore, cycle_id: str):
        self.store = store
        self.id = cycle_id
        self.dir = store.root / "cycles" / cycle_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / "events.jsonl"
        self.state_path = self.dir / "state.json"
        self._t0 = time.monotonic()

    def emit(self, kind: str, **data: Any) -> dict:
        ev = {"ts": now_iso(), "cycle": self.id, "kind": kind, **data}
        with self.events_path.open("a") as f:
            f.write(json.dumps(ev, sort_keys=True) + "\n")
        return ev

    def events(self) -> Iterator[dict]:
        if not self.events_path.exists():
            return iter(())
        return (json.loads(line) for line in self.events_path.read_text().splitlines() if line.strip())

    @property
    def state(self) -> dict:
        return read_json(self.state_path, {})

    def update(self, **kv: Any) -> dict:
        st = self.state
        st.update(kv)
        write_json(self.state_path, st)
        return st

    def stage_done(self, name: str) -> bool:
        return name in self.state.get("stages_done", [])

    def mark_done(self, name: str, **summary: Any) -> None:
        st = self.state
        done = list(st.get("stages_done", []))
        if name not in done:
            done.append(name)
        st["stages_done"] = done
        st.setdefault("stage_summary", {})[name] = summary
        write_json(self.state_path, st)
        self.emit("stage.done", stage=name, **summary)

    def subdir(self, name: str) -> Path:
        p = self.dir / name
        p.mkdir(parents=True, exist_ok=True)
        return p
