"""`polyloop report`: one self-contained HTML page from a loop's runs directory.

Reads only what the stages wrote (state.json, events.jsonl, receipt.json, results.jsonl)
so the page is a faithful view of the ledger, not a second source of truth."""
from __future__ import annotations

import html
import json
from pathlib import Path

from polyloop.events import LoopStore, read_json

CSS = """
:root{--bg:#f6f4ee;--fg:#1d1b16;--mut:#6b675c;--line:#d9d4c7;--card:#fffdf8;--acc:#0b6e7a;--good:#2f7d32;--bad:#b3261e;--warn:#9a6b00}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15140f;--fg:#ece8dc;--mut:#a39d8e;--line:#2e2c24;--card:#1d1b15;--acc:#4fb3c0;--good:#7fc47f;--bad:#f28b82;--warn:#e0b34a}}
:root[data-theme="dark"]{--bg:#15140f;--fg:#ece8dc;--mut:#a39d8e;--line:#2e2c24;--card:#1d1b15;--acc:#4fb3c0;--good:#7fc47f;--bad:#f28b82;--warn:#e0b34a}
body{background:var(--bg);color:var(--fg);font:15px/1.5 "IBM Plex Sans",system-ui,sans-serif;margin:0;padding-block:32px;padding-inline:clamp(16px,4vw,48px);max-width:1100px;margin-inline:auto}
h1{font:600 28px/1.2 "IBM Plex Sans",system-ui,sans-serif;margin:0 0 4px}h2{font-size:18px;margin:36px 0 12px;text-wrap:balance}
.sub{color:var(--mut)}.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--mut);font-weight:500;letter-spacing:.02em;text-transform:uppercase;font-size:12px}
.wrap{overflow-x:auto}.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;border:1px solid var(--line)}
.promoted{color:var(--good);border-color:var(--good)}.rejected,.aborted,.failed{color:var(--bad);border-color:var(--bad)}.awaiting_approval{color:var(--warn);border-color:var(--warn)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.kv{background:var(--card);border:1px solid var(--line);padding:12px 14px}.kv b{display:block;font-size:22px;font-weight:600}.kv span{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.03em}
svg text{fill:var(--fg);font-size:11px}.axis{stroke:var(--line)}pre{background:var(--card);border:1px solid var(--line);padding:12px;overflow-x:auto;font-size:12px}
"""


def _fmt(x, nd=3):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _curve_svg(points: list[dict]) -> str:
    """points: [{cycle, incumbent, candidate, lo, hi, decision}]"""
    if not points:
        return "<p class='sub'>No evaluated cycles yet.</p>"
    W, H, L, B = 720, 260, 48, 36
    n = len(points)
    xs = [L + (W - L - 16) * (i / max(1, n - 1)) for i in range(n)] if n > 1 else [L + (W - L) / 2]
    ys = lambda v: H - B - (H - B - 16) * max(0.0, min(1.0, v))
    parts = [f"<svg viewBox='0 0 {W} {H}' width='100%' role='img' aria-label='held-out score per cycle'>"]
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f"<line class='axis' x1='{L}' x2='{W-16}' y1='{ys(t):.1f}' y2='{ys(t):.1f}' stroke-width='1'/><text x='{L-8}' y='{ys(t)+4:.1f}' text-anchor='end'>{t:.2f}</text>")
    inc = " ".join(f"{x:.1f},{ys(p['incumbent']):.1f}" for x, p in zip(xs, points) if p.get("incumbent") is not None)
    cand = " ".join(f"{x:.1f},{ys(p['candidate']):.1f}" for x, p in zip(xs, points) if p.get("candidate") is not None)
    if inc:
        parts.append(f"<polyline points='{inc}' fill='none' stroke='var(--mut)' stroke-width='2' stroke-dasharray='4 3'/>")
    if cand:
        parts.append(f"<polyline points='{cand}' fill='none' stroke='var(--acc)' stroke-width='2'/>")
    for x, p in zip(xs, points):
        if p.get("candidate") is None:
            continue
        col = "var(--good)" if p.get("decision") == "promote" else "var(--bad)"
        parts.append(f"<circle cx='{x:.1f}' cy='{ys(p['candidate']):.1f}' r='5' fill='{col}'/>")
        parts.append(f"<text x='{x:.1f}' y='{H-B+16}' text-anchor='middle'>{html.escape(p['cycle'][-7:])}</text>")
    parts.append(f"<text x='{L}' y='14'>held-out mean reward · solid = candidate, dashed = incumbent · green dot = promoted</text></svg>")
    return "".join(parts)


def build(store: LoopStore, cfg) -> str:
    lin = store.lineage()
    rows, points = [], []
    stage_secs_all = []
    for cid in store.cycles():
        c = store.cycle(cid)
        st = c.state
        secs = {e["stage"]: e["seconds"] for e in c.events() if e["kind"] == "stage.seconds"}
        stage_secs_all.append(secs)
        p = st.get("paired") or {}
        status = st.get("status", "running")
        rows.append(
            f"<tr><td class='mono'>{cid}</td><td><span class='pill {status}'>{status}</span></td>"
            f"<td>{len(st.get('stages_done', []))}/8</td><td class='mono'>{len(st.get('train_task_names', []))}</td>"
            f"<td class='mono'>{_fmt(p.get('incumbent_mean'))} → {_fmt(p.get('candidate_mean'))}</td>"
            f"<td class='mono'>{_fmt(p.get('mean_delta'))} [{_fmt((p.get('delta_ci95') or [None, None])[0], 2)}, {_fmt((p.get('delta_ci95') or [None, None])[1], 2)}]</td>"
            f"<td class='mono'>{p.get('wins', '–')}/{p.get('losses', '–')}/{p.get('ties', '–')}</td>"
            f"<td class='mono'>{_fmt(st.get('logprob_abs_diff_max'), 4)}</td>"
            f"<td class='mono'>{sum(st.get('gpu_seconds', {}).values())/3600:.2f}</td></tr>")
        if p:
            points.append({"cycle": cid, "incumbent": p.get("incumbent_mean"), "candidate": p.get("candidate_mean"),
                           "decision": st.get("decision")})
    n_cycles = len(rows)
    promoted = len(lin.get("history", []))
    last_receipt = None
    for cid in reversed(store.cycles()):
        r = store.cycle(cid).dir / "receipt.json"
        if r.exists():
            last_receipt = read_json(r, None)
            break
    stage_rows = ""
    if stage_secs_all:
        names = ["snapshot", "preflight", "filter", "train", "evaluate", "gate", "promote", "observe"]
        stage_rows = "".join(
            f"<tr><td>{n}</td><td class='mono'>{_fmt(sum(s.get(n, 0) for s in stage_secs_all)/max(1, sum(1 for s in stage_secs_all if n in s))/60, 1)} min</td></tr>"
            for n in names)
    pool = store.cache("pool_stats")
    inc_id = (lin.get("incumbent") or {}).get("id", "base")
    under = pool.get(inc_id, {})
    contested = sum(1 for v in under.values() if 0 < v["pass_rate"] < 1)
    solved = sum(1 for v in under.values() if v["pass_rate"] >= 1)
    doc = f"""<title>{html.escape(cfg.name)} loop</title><style>{CSS}</style>
<h1>{html.escape(cfg.name)}</h1>
<p class='sub'>{html.escape(cfg.model)} · LoRA rank {cfg.stages[0].lora_rank} · holdout {html.escape(cfg.gate.holdout)} (n={cfg.gate.holdout_limit}, K={cfg.gate.repeats}) · promote: {cfg.promote.mode}</p>
<div class='grid'>
<div class='kv'><span>cycles</span><b>{n_cycles}</b></div>
<div class='kv'><span>promoted</span><b>{promoted}</b></div>
<div class='kv'><span>incumbent</span><b style='font-size:15px' class='mono'>{html.escape(inc_id)}</b></div>
<div class='kv'><span>pool measured / contested / solved</span><b>{len(under)} / {contested} / {solved}</b></div>
</div>
<h2>Held-out score per cycle</h2>{_curve_svg(points)}
<h2>Cycles</h2><div class='wrap'><table><tr><th>cycle</th><th>status</th><th>stages</th><th>train tasks</th><th>incumbent → candidate</th><th>paired Δ [95% CI]</th><th>W/L/T</th><th>logprob |Δ| max</th><th>GPU·h</th></tr>{''.join(rows) or "<tr><td colspan=9 class='sub'>none yet</td></tr>"}</table></div>
<h2>Mean stage duration</h2><div class='wrap'><table><tr><th>stage</th><th>mean</th></tr>{stage_rows or "<tr><td colspan=2 class='sub'>none yet</td></tr>"}</table></div>
<h2>Latest receipt</h2><pre>{html.escape(json.dumps(last_receipt, indent=1, sort_keys=True)) if last_receipt else '<span class="sub">none yet</span>'}</pre>
<p class='sub'>Generated by <span class='mono'>polyloop report</span> from {html.escape(str(store.root))}.</p>
"""
    return doc
