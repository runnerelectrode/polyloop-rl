"""Render docs/stack.svg: the three layers (polyvoice, polyloop-rl, rlcli) and what sits between them. Stdlib only."""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

W, H = 1100, 520
F = "font-family='Helvetica, Arial, sans-serif'"
INK, MUTE, LINE = "#1f2933", "#52606d", "#9aa5b1"
out: list[str] = []


def text(x, y, s, size=12, weight="400", fill=MUTE, anchor="start"):
    out.append(f"<text x='{x}' y='{y}' {F} font-size='{size}' font-weight='{weight}' fill='{fill}' text-anchor='{anchor}'>{escape(s)}</text>")


def layer(y, h, fill, name, role, lines, repo):
    out.append(f"<rect x='24' y='{y}' width='1052' height='{h}' rx='12' fill='{fill}' stroke='{LINE}' stroke-width='1.2'/>")
    text(40, y + 28, name, 18, "700", INK)
    text(40, y + 48, role, 12, "700", MUTE)
    for i, t in enumerate(lines):
        text(300, y + 26 + 18 * i, t, 12.5)
    text(1060, y + h - 10, repo, 11, anchor="end")


def seam(y, label, detail):
    out.append(f"<line x1='24' y1='{y}' x2='1076' y2='{y}' stroke='{INK}' stroke-width='1.2' stroke-dasharray='6 5'/>")
    out.append(f"<rect x='200' y='{y - 13}' width='700' height='26' rx='13' fill='white' stroke='{INK}' stroke-width='1.2'/>")
    text(550, y + 4, label, 12, "700", INK, anchor="middle")
    text(1060, y + 4, detail, 11, anchor="end")


out.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>")
out.append(f"<rect width='{W}' height='{H}' fill='white'/>")
text(24, 34, "rlcli · polyloop-rl · polyvoice: each layer works without the one above it", 18, "700", INK)

layer(56, 96, "#fde8e8", "polyvoice", "environment + recipe",
      ["the world: Coval's simulated caller phones the proxy; the reward: Coval's judges",
       "the dental receptionist scenarios (pool 16, held-out 8) and system prompt",
       "the ledger: every call → session id, scenario, policy, reward, judge explanation",
       "no training code; the same shape as the built-in Docker Harbor environment"], "github.com/polygramme/polyvoice")
seam(174, "Environment protocol: load_tasks · run_rollouts · preflight · session_hints · excluded_sessions", "")
layer(196, 110, "#f3eefc", "polyloop-rl", "controller",
      ["cycles: snapshot → preflight → filter → train (rl, opsd) → evaluate → gate → promote",
       "capture proxy: token-exact traces, session + next-state pairing, serves the live adapter",
       "receipt: paired delta, bootstrap CI, regressions, logprob agreement; lineage; budget; UI",
       "owns what is measured (held-out split, thresholds); never how an episode is run",
       "reaches the model only over the Tinker API, the world only through an environment"], "github.com/runnerelectrode/polyloop-rl")
seam(328, "Tinker API over HTTP  +  rlcli and tinker-cookbook imported as libraries", "")
layer(350, 96, "#e9f7ef", "rlcli", "training primitives",
      ["rlcli serve: a Tinker-API server on SkyRL, trainer GPU + sampler GPU, LoRA, multi-adapter",
       "token-in/token-out bridge across turns, Docker sandboxes, hinted OPSD teacher",
       "trainer-vs-sampler logprob guard, trace import and pass-through capture",
       "built on tinker-cookbook (renderers, RL + distillation loops) and SkyRL (Megatron, vLLM)"], "github.com/polygramme/rlcli")
out.append(f"<rect x='24' y='466' width='1052' height='34' rx='10' fill='#f5f7fa' stroke='{LINE}' stroke-width='1'/>")
text(550, 488, "a GPU node with two GPUs (Lambda, a Modal container, your own box) plus Coval (voice) or Docker (code) as the world", 12, anchor="middle")
out.append("</svg>")
Path(__file__).resolve().parents[1].joinpath("docs", "stack.svg").write_text("\n".join(out))
print("wrote docs/stack.svg")
