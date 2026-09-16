"""Render docs/architecture.svg (stdlib only). Run: python scripts/arch_diagram.py"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

W, H = 1240, 640
F = "font-family='Helvetica, Arial, sans-serif'"
INK, MUTE, LINE = "#1f2933", "#52606d", "#9aa5b1"
FILL = {"agent": "#e3f1ff", "proxy": "#fff4d6", "server": "#e9f7ef", "loop": "#f3eefc", "store": "#f5f7fa", "task": "#fde8e8"}
out: list[str] = []


def text(x, y, s, size=12, weight="400", fill=MUTE, anchor="start"):
    out.append(f"<text x='{x}' y='{y}' {F} font-size='{size}' font-weight='{weight}' fill='{fill}' text-anchor='{anchor}'>{escape(s)}</text>")


def box(x, y, w, h, title, lines=(), kind="store"):
    out.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='10' fill='{FILL[kind]}' stroke='{LINE}' stroke-width='1.2'/>")
    text(x + 12, y + 22, title, 14, "700", INK)
    for i, t in enumerate(lines):
        text(x + 12, y + 42 + 16 * i, t)


def route(points, label="", dash=False, head=True, lx=None, ly=None, anchor="middle"):
    d = " stroke-dasharray='5 4'" if dash else ""
    m = " marker-end='url(#a)'" if head else ""
    pts = " ".join(f"{x},{y}" for x, y in points)
    out.append(f"<polyline points='{pts}' fill='none' stroke='{INK}' stroke-width='1.4'{d}{m}/>")
    if label:
        text(lx, ly, label, 11, anchor=anchor)


out.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>")
out.append("<defs><marker id='a' markerWidth='10' markerHeight='10' refX='9' refY='5' orient='auto'>"
           f"<path d='M0,0 L10,5 L0,10 z' fill='{INK}'/></marker></defs>")
out.append(f"<rect width='{W}' height='{H}' fill='white'/>")
text(24, 34, "polyloop-rl: daytime traffic in, gated adapter out", 20, "700", INK)

# row 1: daytime path
box(24, 90, 250, 96, "coding agent", ["mini-swe-agent, Claude Code, any", "OpenAI- or Anthropic-compatible client", "X-Session-Id · X-Turn-Type · X-Session-Done"], "agent")
box(470, 90, 270, 96, "polyloop proxy  :8787", ["tinker-cookbook capture proxy", "pairs turn t with the next state", "serves the adapter named in live.json"], "proxy")
box(800, 90, 240, 96, "SkyRL Tinker server", ["trainer GPU + sampler GPU (vLLM)", "LoRA, multi-adapter, Megatron", "rlcli serve start"], "server")
route([(274, 138), (470, 138)], "chat completions", lx=372, ly=132)
route([(740, 138), (800, 138)], "sample", lx=770, ly=132)

# row 2: what the daytime path leaves behind, and the task pool
box(24, 222, 250, 66, "task pool", ["Harbor tasks + verifier, train / holdout split", "polyloop tasks pydantic-v2 | swesmith | split"], "task")
box(470, 222, 270, 66, "traces/<date>.jsonl", ["one line per assistant turn:", "messages, response, tool calls, next_state"], "store")
route([(605, 186), (605, 222)], "records", lx=640, ly=210)
route([(149, 288), (149, 350)], "pin pool + holdout", lx=160, ly=306, anchor="start")
route([(640, 288), (640, 350)], "rows + hindsight hint → OPSD", lx=652, ly=304, anchor="start")

# server bus into the cycle (dashed): filter, train and evaluate all call the same Tinker API
route([(920, 186), (920, 318), (427, 318)], "", dash=True, head=False)
for tx in (427, 560, 755):
    route([(tx, 318), (tx, 350)], dash=True)
text(925, 260, "train + sample over the Tinker API", 11)

# row 3: the cycle
y = 350
text(24, y - 12, "one cycle  (polyloop run --loop loop.yaml)", 13, "700", INK)
stages = [("snapshot", ["pin pool, holdout,", "incumbent, trace files"]), ("preflight", ["docker, server,", "disk, staleness"]),
          ("filter", ["pass rate under the", "incumbent, keep 0<p<1"]), ("train", ["rl: cookbook Harbor RL", "opsd: hinted self-distill"]),
          ("evaluate", ["candidate vs incumbent,", "same holdout, K repeats"]), ("gate", ["paired bootstrap CI,", "regressions, logprobs"]),
          ("promote", ["approve or auto;", "writes live.json"])]
xs = [24 + 164 * i for i in range(len(stages))]
for x, (name, lines) in zip(xs, stages):
    box(x, y, 150, 74, name, lines, "loop")
for i in range(len(xs) - 1):
    route([(xs[i] + 150, y + 37), (xs[i + 1], y + 37)])

# row 4: artifacts
y2 = 490
box(24, y2, 456, 120, "polyloop ui  /  polyloop report", ["stage, sandboxes, rewards as they land, last train steps, GPU load,", "sessions through the proxy, the live adapter,", "held-out curve and the receipt of every cycle"], "store")
box(520, y2, 380, 120, "runs/<loop>/cycles/<id>/", ["events.jsonl   append-only stage events (resume, audit)", "state.json     where to resume", "receipt.json   promotion_receipt.v1: paired delta, CI95,", "               W/L/T, checks, gpu-seconds"], "store")
box(930, y2, 286, 120, "runs/<loop>/live.json", ["sampler_path of the promoted adapter", "lineage.json: every promotion", "POST /admin/promote reloads the proxy"], "store")
route([(919, y + 74), (919, 456), (710, 456), (710, y2)], "receipt", lx=800, ly=450)
route([(1083, y + 74), (1083, y2)], "promoted", lx=1095, ly=462, anchor="start")
route([(520, 550), (480, 550)])
route([(1180, y2), (1180, 62), (605, 62), (605, 90)], "the proxy serves the promoted adapter, the agent changes nothing", dash=True, lx=892, ly=56)
out.append("</svg>")
Path(__file__).resolve().parents[1].joinpath("docs", "architecture.svg").write_text("\n".join(out))
print("wrote docs/architecture.svg")
