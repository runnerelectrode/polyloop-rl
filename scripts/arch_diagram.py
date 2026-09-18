"""Render docs/architecture.svg (stdlib only). Run: python scripts/arch_diagram.py"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

W, H = 1240, 760
F = "font-family='Helvetica, Arial, sans-serif'"
INK, MUTE, LINE, LANE = "#1f2933", "#52606d", "#9aa5b1", "#8a94a6"
FILL = {"agent": "#e3f1ff", "proxy": "#fff4d6", "server": "#e9f7ef", "loop": "#f3eefc", "store": "#f5f7fa", "env": "#fde8e8", "iface": "#ffffff"}
out: list[str] = []


def text(x, y, s, size=12, weight="400", fill=MUTE, anchor="start"):
    out.append(f"<text x='{x}' y='{y}' {F} font-size='{size}' font-weight='{weight}' fill='{fill}' text-anchor='{anchor}'>{escape(s)}</text>")


def box(x, y, w, h, title, lines=(), kind="store", dashed=False):
    d = " stroke-dasharray='6 4'" if dashed else ""
    out.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='10' fill='{FILL[kind]}' stroke='{LINE}' stroke-width='1.2'{d}/>")
    text(x + 12, y + 22, title, 14, "700", INK)
    for i, t in enumerate(lines):
        text(x + 12, y + 42 + 16 * i, t)


def route(points, label="", dash=False, head=True, lx=None, ly=None, anchor="middle"):
    d = " stroke-dasharray='5 4'" if dash else ""
    m = " marker-end='url(#a)'" if head else ""
    out.append(f"<polyline points='{' '.join(f'{x},{y}' for x, y in points)}' fill='none' stroke='{INK}' stroke-width='1.4'{d}{m}/>")
    if label:
        text(lx, ly, label, 11, anchor=anchor)


def lane(y, label):
    out.append(f"<line x1='24' y1='{y}' x2='{W - 24}' y2='{y}' stroke='{LANE}' stroke-width='0.8' stroke-dasharray='2 6'/>")
    text(24, y - 6, label, 11, "700", LANE)


out.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>")
out.append("<defs><marker id='a' markerWidth='10' markerHeight='10' refX='9' refY='5' orient='auto'>"
           f"<path d='M0,0 L10,5 L0,10 z' fill='{INK}'/></marker></defs>")
out.append(f"<rect width='{W}' height='{H}' fill='white'/>")
text(24, 34, "polyloop-rl: traffic in, gated adapter out", 20, "700", INK)
text(24, 54, "the controller owns the split, the thresholds, the budget and the lineage; the environment owns how an episode is run and scored", 12)

# ---- lane 1: traffic
lane(84, "DAYTIME TRAFFIC")
box(24, 96, 250, 96, "agent", ["mini-swe-agent, Claude Code, a voice", "receptionist behind Coval's caller…", "any OpenAI / Anthropic-compatible client"], "agent")
box(470, 96, 270, 96, "polyloop proxy", ["renders chat → tokens, samples, records", "session id · main/side turn · next state", "serves the adapter in live.json (or an override)"], "proxy")
box(800, 96, 240, 96, "Tinker-API server", ["SkyRL: trainer GPU + sampler GPU", "LoRA, multi-adapter, Megatron", "rlcli serve start"], "server")
route([(274, 144), (470, 144)], "chat completions", lx=372, ly=138)
route([(740, 144), (800, 144)], "sample", lx=770, ly=138)
box(470, 214, 270, 60, "traces/<date>.jsonl", ["one line per assistant turn, token-exact:", "messages, response, tool calls, next_state"], "store")
route([(605, 192), (605, 214)])

# ---- lane 2: the cycle
lane(316, "ONE CYCLE   polyloop run --loop loop.yaml")
y = 330
stages = [("snapshot", ["pin pool, holdout,", "incumbent, traces"]), ("preflight", ["env checks, server,", "disk, staleness"]),
          ("filter", ["pass rate under the", "incumbent, keep 0<p<1"]), ("train", ["rl on contested tasks,", "then opsd on traces"]),
          ("evaluate", ["candidate vs incumbent,", "same holdout, K repeats"]), ("gate", ["paired bootstrap CI,", "regressions, logprobs"]),
          ("promote", ["approve or auto;", "writes live.json"])]
xs = [24 + 164 * i for i in range(len(stages))]
for x, (name, lines) in zip(xs, stages):
    box(x, y, 150, 74, name, lines, "loop")
for i in range(len(xs) - 1):
    route([(xs[i] + 150, y + 37), (xs[i + 1], y + 37)])
route([(640, 274), (640, y)], "rows + hindsight hint → opsd", lx=628, ly=310, anchor="end")
route([(920, 192), (920, 290), (560, 290), (560, y)], "train + sample over the Tinker API", dash=True, lx=780, ly=284)

# ---- lane 3: environments (the plugin seam)
lane(446, "ENVIRONMENT   loop.yaml: environment.kind")
ey = 460
box(24, ey, 796, 34, "Environment protocol:  load_tasks · run_rollouts · preflight · session_hints · excluded_sessions", [], "iface", dashed=True)
route([(427, y + 74), (427, ey)], "tasks + episodes", lx=440, ly=438, anchor="start")
route([(755, y + 74), (755, ey)], "tasks + episodes", lx=768, ly=438, anchor="start")
route([(591, y + 74), (591, ey)], "hints, exclusions", lx=603, ly=438, anchor="start")
ey2 = 512
box(24, ey2, 250, 84, "harbor-docker  (built in)", ["Harbor tasks + verifier,", "cookbook bash loop, Docker", "sandboxes, reward = tests"], "env")
box(296, ey2, 250, 84, "coval  (polyvoice)", ["Coval's simulated caller", "phones the proxy; judges", "score + explain (hints)"], "env")
box(570, ey2, 250, 84, "yours", ["a browser sim, a router, a", "game… four methods, one", "entry point, runner unchanged"], "env", dashed=True)
for cx in (149, 421, 695):
    route([(cx, ey + 34), (cx, ey2)])

# ---- lane 4: artifacts
lane(632, "ARTIFACTS")
ay = 644
box(24, ay, 456, 96, "runs/<loop>/cycles/<id>/  (every stage writes here)", ["events.jsonl   append-only stage events (resume, audit)", "receipt.json   promotion_receipt.v1: paired delta, CI95, W/L/T,", "               checks, gpu-seconds, holdout id"], "store")
box(510, ay, 300, 96, "polyloop ui / report", ["stage, sandboxes, rewards as they land,", "train steps, GPU load, proxy sessions,", "held-out curve + receipt of every cycle"], "store")
box(836, ay, 380, 96, "live.json + lineage.json", ["sampler_path of the promoted adapter;", "every promotion with its receipt;", "POST /admin/promote reloads the proxy"], "store")
route([(919, y + 74), (919, 620), (700, 620), (700, ay)], "receipt", lx=880, ly=614)
route([(1083, y + 74), (1083, ay)], "promoted", lx=1095, ly=530, anchor="start")
route([(1216, ay + 48), (1230, ay + 48), (1230, 70), (605, 70), (605, 96)], "the proxy serves the promoted adapter; the agent changes nothing", dash=True, lx=900, ly=64)
out.append("</svg>")
Path(__file__).resolve().parents[1].joinpath("docs", "architecture.svg").write_text("\n".join(out))
print("wrote docs/architecture.svg")
