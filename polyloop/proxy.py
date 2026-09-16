"""`polyloop proxy`: the OpenAI-compatible endpoint a coding agent talks to.

Wraps tinker-cookbook's capture proxy (renderer + Tinker sampling, OpenAI and Anthropic
routes) and adds what the loop needs:

- session and turn headers (`X-Session-Id`, `X-Turn-Type: main|side`, `X-Session-Done`), the
  contract OpenClaw-RL's agent extension emits; a missing session id falls back to a hash of
  the first user message so plain agents still get grouped;
- next-state pairing: assistant turn t's next state is `messages[-1]` of the session's next
  main request (a user reply or a tool result). On session done the last turn is written
  without a next state;
- a live adapter: the sampler path is read from `<runs>/<loop>/live.json` at startup and
  after every `/admin/promote`, so a promoted cycle changes what the agent samples from
  without touching the agent.

Records land as one JSON line per assistant turn under `<runs>/<loop>/traces/<date>.jsonl`.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import web

from polyloop.events import now_iso, read_json, write_json

MAIN, SIDE = "main", "side"


@dataclass
class _Pending:
    session: str
    turn: int
    request: dict
    response_text: str
    tool_calls: list[dict]
    usage: dict
    started: str
    seconds: float


@dataclass
class TraceStore:
    root: Path
    pending: dict[str, _Pending] = field(default_factory=dict)
    turns: dict[str, int] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _path(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / (time.strftime("%Y-%m-%d") + ".jsonl")

    def _write(self, p: _Pending, next_state: dict | None, done: bool) -> None:
        rec = {
            "ts": now_iso(), "session": p.session, "turn": p.turn, "started": p.started, "seconds": round(p.seconds, 2),
            "messages": p.request.get("messages", []), "tools": p.request.get("tools"),
            "response": p.response_text, "tool_calls": p.tool_calls, "usage": p.usage,
            "next_state": next_state, "session_done": done,
        }
        with self._path().open("a") as f:
            f.write(json.dumps(rec) + "\n")

    async def record(self, session: str, turn_type: str, request: dict, response_text: str,
                     tool_calls: list[dict], usage: dict, started: str, seconds: float, done: bool) -> None:
        async with self.lock:
            msgs = request.get("messages") or []
            if turn_type == MAIN and session in self.pending and msgs:
                prev = self.pending.pop(session)
                self._write(prev, msgs[-1], False)
            if turn_type != MAIN:
                return
            n = self.turns.get(session, 0)
            self.turns[session] = n + 1
            p = _Pending(session, n, request, response_text, tool_calls, usage, started, seconds)
            if done:
                self._write(p, None, True)
                self.turns.pop(session, None)
            else:
                self.pending[session] = p

    async def finish(self, session: str) -> None:
        async with self.lock:
            p = self.pending.pop(session, None)
            if p:
                self._write(p, None, True)
            self.turns.pop(session, None)


def _session_id(request: web.Request, body: dict) -> str:
    sid = request.headers.get("X-Session-Id") or body.get("session_id") or body.get("user")
    if sid:
        return str(sid)
    for m in body.get("messages") or []:
        if m.get("role") == "user":
            c = m.get("content")
            text = c if isinstance(c, str) else json.dumps(c)
            return "anon-" + hashlib.sha256(text.encode()).hexdigest()[:12]
    return "anon-" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def _extract_openai(resp: dict) -> tuple[str, list[dict], dict]:
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return "", [], resp.get("usage") or {}
    return msg.get("content") or "", msg.get("tool_calls") or [], resp.get("usage") or {}


def _extract_anthropic(resp: dict) -> tuple[str, list[dict], dict]:
    text, calls = [], []
    for block in resp.get("content") or []:
        if block.get("type") == "text":
            text.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            calls.append(block)
    return "".join(text), calls, resp.get("usage") or {}


def make_app(*, base_url: str, model: str, renderer_name: str | None, runs_dir: Path, loop_name: str,
             default_max_tokens: int = 4096) -> web.Application:
    import tinker
    from tinker_cookbook.capture.proxy.app import ProxyDeps, make_app as cookbook_app
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from polyloop.harness.rollout import prepare_env, warm

    prepare_env(base_url)
    loop_dir = runs_dir / loop_name
    # Hold a training session for the proxy's lifetime so the sampler engines stay resident.
    warm(base_url, model, log=print)
    keepalive = warm.last_client  # noqa: F841  (referenced below via app state)
    live_path = loop_dir / "live.json"
    service = tinker.ServiceClient(base_url=base_url)
    rname = renderer_name or get_recommended_renderer_name(model)
    renderer = get_renderer(rname, get_tokenizer(model), model_name=model)

    def sampling_client():
        live = read_json(live_path, {})
        path = live.get("sampler_path")
        if path:
            return service.create_sampling_client(model_path=path), path
        return service.create_sampling_client(base_model=model), model

    sc, label = sampling_client()
    deps = ProxyDeps(renderer=renderer, sampling_client=sc, model_label=label, default_max_tokens=default_max_tokens)
    inner = cookbook_app(deps)
    store = TraceStore(loop_dir / "traces")

    async def reload_live() -> str:
        sc2, label2 = sampling_client()
        deps.sampling_client = sc2
        deps.model_label = label2
        return label2

    @web.middleware
    async def capture(request: web.Request, handler):
        path = request.path
        is_openai = path.endswith("/v1/chat/completions")
        is_anthropic = path.endswith("/v1/messages")
        if not (is_openai or is_anthropic):
            return await handler(request)
        raw = await request.read()
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return await handler(request)
        if body.get("stream"):
            body["stream"] = False  # record the full turn; agents we target accept non-streaming
            request._read_bytes = json.dumps(body).encode()  # noqa: SLF001
        session = _session_id(request, body)
        turn_type = (request.headers.get("X-Turn-Type") or MAIN).lower()
        done = _truthy(request.headers.get("X-Session-Done"))
        started = now_iso()
        t0 = time.monotonic()
        resp = await handler(request)
        seconds = time.monotonic() - t0
        if resp.status == 200 and resp.body:
            try:
                data = json.loads(resp.body)
            except Exception:
                data = None
            if isinstance(data, dict):
                text, calls, usage = (_extract_openai if is_openai else _extract_anthropic)(data)
                await store.record(session, turn_type, body, text, calls, usage, started, seconds, done)
        return resp

    inner.middlewares.append(capture)

    async def promote(request: web.Request) -> web.Response:
        body = await request.json()
        if "sampler_path" in body:
            write_json(live_path, {"sampler_path": body["sampler_path"], "candidate": body.get("candidate"),
                                   "promoted_at": now_iso(), "receipt": body.get("receipt")})
        label2 = await reload_live()
        return web.json_response({"live": label2})

    async def session_done(request: web.Request) -> web.Response:
        sid = request.match_info["session"]
        await store.finish(sid)
        return web.json_response({"session": sid, "done": True})

    async def status(request: web.Request) -> web.Response:
        return web.json_response({"live": deps.model_label, "pending_sessions": len(store.pending),
                                  "traces_dir": str(store.root)})

    inner["polyloop_keepalive"] = keepalive
    inner.router.add_post("/admin/promote", promote)
    inner.router.add_post("/admin/session/{session}/done", session_done)
    inner.router.add_get("/admin/status", status)
    return inner


def serve(host: str, port: int, **kw) -> None:
    app = make_app(**kw)
    print(f"polyloop proxy on http://{host}:{port}  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, admin: /admin/status)", flush=True)
    web.run_app(app, host=host, port=port, print=None)
