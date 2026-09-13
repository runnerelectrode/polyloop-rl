"""`polyloop ui`: a live page for one loop, served from the node.

Stdlib only. Every request rebuilds the page from the ledger, so what you see is what the
stages wrote. Open it through an SSH port-forward:

    ssh -L 8080:127.0.0.1:8080 ubuntu@<node>   ->   http://localhost:8080
"""
from __future__ import annotations

import html
import json
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from polyloop.events import LoopStore
from polyloop.report import build as build_report

LIVE_CSS = """
.live{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:12px 0 24px}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
.stages{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.stages span{padding:2px 10px;border:1px solid var(--line);border-radius:4px;font-size:12px;color:var(--mut)}
.stages .done{color:var(--good);border-color:var(--good)}.stages .now{color:var(--acc);border-color:var(--acc);font-weight:600}
.ok{color:var(--good)}.zero{color:var(--mut)}.neg{color:var(--bad)}.hdr{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px}
"""

STAGES = ["snapshot", "preflight", "filter", "train", "evaluate", "gate", "promote", "observe"]


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def _gpu() -> list[dict]:
    out = _sh(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
    rows = []
    for line in out.splitlines():
        try:
            i, u, m, t = [x.strip() for x in line.split(",")]
            rows.append({"gpu": i, "util": int(u), "used": int(m), "total": int(t)})
        except ValueError:
            pass
    return rows


def _reward_cls(v: float) -> str:
    return "ok" if v > 0 else ("neg" if v < 0 else "zero")


def _tail_metrics(train_dir: Path, n: int = 12) -> list[dict]:
    p = train_dir / "metrics.jsonl"
    if not p.exists():
        return []
    lines = p.read_text().splitlines()[-n:]
    out = []
    for l in lines:
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    return out


def live_section(store: LoopStore) -> str:
    cycles = store.cycles()
    if not cycles:
        return "<h2>Live</h2><p class='sub'>No cycle has started.</p>"
    c = store.cycle(cycles[-1])
    st = c.state
    events = list(c.events())
    started = {e["stage"] for e in events if e["kind"] == "stage.start"}
    done = set(st.get("stages_done", []))
    current = next((s for s in STAGES if s in started and s not in done), None)
    status = st.get("status", "running" if current else "idle")
    chips = "".join(
        f"<span class='{'done' if s in done else ('now' if s == current else '')}'>{s}</span>" for s in STAGES)
    tasks = [e for e in events if e["kind"] == "rollout.task"][-40:]
    task_rows = "".join(
        f"<tr><td class='mono'>{html.escape(e['ts'][11:19])}</td><td>{e['stage']}</td><td class='mono'>{html.escape(e['task'][:60])}</td>"
        f"<td class='mono'>{' '.join(f'<span class={_reward_cls(r)}>{r:+.1f}</span>' for r in e['rewards']) or html.escape(e.get('error') or '')[:80]}</td>"
        f"<td class='mono'>{e.get('seconds', '')}</td></tr>" for e in reversed(tasks))
    n_done = {s: sum(1 for e in events if e["kind"] == "rollout.task" and e["stage"] == s) for s in ("filter", "evaluate.incumbent", "evaluate.candidate")}
    metrics = _tail_metrics(c.dir / "train")
    mrows = ""
    if metrics:
        keys = [k for k in ("step", "env/all/reward/mean", "reward/mean", "optim/rollout_logprobs_abs_diff_mean", "optim/kl_sample_train_v1", "time/total") if any(k in m for m in metrics)]
        mrows = "<tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in keys) + "</tr>" + "".join(
            "<tr>" + "".join(f"<td class='mono'>{html.escape(str(round(m[k], 4)) if isinstance(m.get(k), float) else str(m.get(k, '')))}</td>" for k in keys) + "</tr>" for m in metrics)
    gpus = "".join(
        f"<div class='kv'><span>GPU {g['gpu']} · {g['util']}% util</span><div class='bar'><i style='width:{100*g['used']/max(1,g['total']):.0f}%'></i></div><span>{g['used']/1024:.1f} / {g['total']/1024:.0f} GB</span></div>" for g in _gpu())
    containers = _sh(["docker", "ps", "-q"]).count("\n") + (1 if _sh(["docker", "ps", "-q"]) else 0)
    gpu_h = sum(st.get("gpu_seconds", {}).values()) / 3600
    return f"""
<h2 class='hdr'><span>Live · cycle <span class='mono'>{c.id}</span> <span class='pill {status}'>{status}</span></span><span class='sub'>refreshed {time.strftime('%H:%M:%S')} UTC</span></h2>
<div class='stages'>{chips}</div>
<div class='live'>
<div class='kv'><span>sandboxes running</span><b>{containers}</b></div>
<div class='kv'><span>filter episodes</span><b>{n_done['filter']}</b></div>
<div class='kv'><span>eval episodes (inc / cand)</span><b>{n_done['evaluate.incumbent']} / {n_done['evaluate.candidate']}</b></div>
<div class='kv'><span>train tasks · steps</span><b>{len(st.get('train_task_names', []))} · {st.get('train_steps', '–')}</b></div>
<div class='kv'><span>GPU·h this cycle</span><b>{gpu_h:.2f}</b></div>
{gpus}
</div>
{'<h2>Training metrics (last steps)</h2><div class="wrap"><table>' + mrows + '</table></div>' if mrows else ''}
<h2>Latest episodes</h2><div class='wrap'><table><tr><th>time</th><th>stage</th><th>task</th><th>rewards</th><th>s</th></tr>{task_rows or "<tr><td colspan=5 class='sub'>none yet</td></tr>"}</table></div>
"""


def page(store: LoopStore, cfg, refresh: int) -> str:
    report = build_report(store, cfg)
    head, _, rest = report.partition("<h1>")
    return (f"{head}<meta http-equiv='refresh' content='{refresh}'><style>{LIVE_CSS}</style><h1>{rest}"
            .replace("<h2>Held-out score per cycle</h2>", live_section(store) + "<h2>Held-out score per cycle</h2>", 1))


def serve(cfg, store: LoopStore, host: str, port: int, refresh: int = 15) -> None:
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = page(store, cfg, refresh).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # quiet
            pass

    print(f"polyloop ui: http://{host}:{port}  (ssh -L {port}:127.0.0.1:{port} <node> then open http://localhost:{port})", flush=True)
    ThreadingHTTPServer((host, port), H).serve_forever()
