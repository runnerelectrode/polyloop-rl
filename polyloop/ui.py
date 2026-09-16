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

from polyloop.events import LoopStore, read_json
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


def _sessions(store: LoopStore, n: int = 12) -> tuple[list[dict], list[dict]]:
    """Recent proxy sessions (from traces/*.jsonl) and a few hinted steps."""
    traces = sorted((store.root / "traces").glob("*.jsonl"))
    recs: list[dict] = []
    for p in traces[-3:]:
        for line in p.read_text().splitlines()[-400:]:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    outcomes = {}
    sess_log = store.root.parent / "sessions" / "sessions.jsonl"
    if sess_log.exists():
        for line in sess_log.read_text().splitlines():
            try:
                r = json.loads(line)
                outcomes[r["session"]] = r
            except (json.JSONDecodeError, KeyError):
                pass
    by: dict[str, dict] = {}
    for r in recs:
        b = by.setdefault(r["session"], {"session": r["session"], "turns": 0, "last": r["ts"], "hinted": 0, "done": False})
        b["turns"] = max(b["turns"], r.get("turn", 0) + 1)
        b["last"] = max(b["last"], r["ts"])
        b["hinted"] += 1 if r.get("next_state") else 0
        b["done"] = b["done"] or bool(r.get("session_done"))
    sessions = sorted(by.values(), key=lambda b: b["last"], reverse=True)[:n]
    for b in sessions:
        o = outcomes.get(b["session"], {})
        b["passed"] = o.get("passed_strict")
        b["task"] = o.get("task", "")
    hints = [r for r in recs if r.get("next_state") and "tool output" in (r["next_state"].get("content") or "")][-3:]
    return sessions, hints


def _live_banner(store: LoopStore) -> str:
    live = read_json(store.root / "live.json", {})
    if not live.get("sampler_path"):
        return "<div class='kv'><span>live adapter</span><b style='font-size:15px'>base model</b><span>no promotion yet</span></div>"
    return (f"<div class='kv'><span>live adapter</span><b style='font-size:15px' class='mono'>{html.escape(str(live.get('candidate')))}</b>"
            f"<span>since {html.escape(str(live.get('promoted_at','')))[:16]} · receipt {html.escape(str(live.get('receipt') or ''))[-40:]}</span></div>")


def sessions_section(store: LoopStore) -> str:
    sessions, hints = _sessions(store)
    rows = "".join(
        f"<tr><td class='mono'>{html.escape(b['session'][:34])}</td><td class='mono'>{b['turns']}</td><td class='mono'>{b['hinted']}</td>"
        f"<td>{'yes' if b['done'] else '…'}</td><td>{'' if b['passed'] is None else ('<span class=ok>pass</span>' if b['passed'] else '<span class=neg>fail</span>')}</td>"
        f"<td class='mono'>{html.escape(b['last'][11:19])}</td></tr>" for b in sessions)
    hint_html = "".join(
        f"<details><summary class='mono'>{html.escape(r['session'][:30])} · turn {r['turn']}</summary>"
        f"<p><b>agent said</b></p><pre>{html.escape((r.get('response') or '')[:600])}</pre>"
        f"<p><b>next state, becomes the teacher's hint</b></p><pre>{html.escape((r['next_state'].get('content') or '')[:600])}</pre></details>"
        for r in hints)
    return f"""
<h2>Sessions through the proxy</h2>
<div class='wrap'><table><tr><th>session</th><th>turns</th><th>with next state</th><th>closed</th><th>strict verify</th><th>last</th></tr>{rows or "<tr><td colspan=6 class='sub'>none yet: point an agent at the proxy</td></tr>"}</table></div>
<h2>Hindsight hints (latest)</h2>{hint_html or "<p class='sub'>none yet</p>"}
"""


def live_section(store: LoopStore) -> str:
    cycles = store.cycles()
    if not cycles:
        return f"<h2>Live</h2><div class='live'>{_live_banner(store)}</div><p class='sub'>No cycle has started.</p>"
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
{_live_banner(store)}
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


def log_tail(path: Path | None, n: int = 60) -> str:
    if not path or not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 64_000))
            lines = f.read().decode(errors="replace").splitlines()[-n:]
    except OSError:
        return ""
    return "\n".join(l for l in lines if "unauthenticated requests" not in l and "HF_TOKEN" not in l)


JS = """
<script>
(function(){
  var every=%d*1000;
  async function tick(){
    try{
      var r=await fetch('/fragment',{cache:'no-store'}); var h=await r.text();
      var d=document.createElement('div'); d.innerHTML=h;
      var live=document.getElementById('live'); if(live) live.innerHTML=d.querySelector('#live').innerHTML;
      var lg=document.getElementById('log'); var nl=d.querySelector('#log');
      if(lg&&nl){ var atBottom=lg.scrollHeight-lg.scrollTop-lg.clientHeight<40; lg.textContent=nl.textContent; if(atBottom) lg.scrollTop=lg.scrollHeight; }
    }catch(e){}
    setTimeout(tick,every);
  }
  setTimeout(tick,every);
  var lg=document.getElementById('log'); if(lg) lg.scrollTop=lg.scrollHeight;
})();
</script>
"""


def fragment(store: LoopStore, log_path: Path | None) -> str:
    return (f"<div id='live'>{live_section(store)}{sessions_section(store)}</div>"
            f"<h2>Log</h2><pre id='log' style='max-height:360px;overflow:auto'>{html.escape(log_tail(log_path))}</pre>")


def page(store: LoopStore, cfg, refresh: int, log_path: Path | None) -> str:
    report = build_report(store, cfg)
    head, _, rest = report.partition("<h1>")
    body = f"{head}<style>{LIVE_CSS}</style><h1>{rest}"
    body = body.replace("<h2>Held-out score per cycle</h2>", fragment(store, log_path) + "<h2>Held-out score per cycle</h2>", 1)
    return body + (JS % max(2, refresh))


def serve(cfg, store: LoopStore, host: str, port: int, refresh: int = 5, log_path: Path | None = None) -> None:
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/fragment"):
                body = fragment(store, log_path).encode()
            else:
                body = page(store, cfg, refresh, log_path).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # quiet
            pass

    print(f"polyloop ui: http://{host}:{port}  (ssh -L {port}:127.0.0.1:{port} <node> then open http://localhost:{port})", flush=True)
    ThreadingHTTPServer((host, port), H).serve_forever()
