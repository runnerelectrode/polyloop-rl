import asyncio

from polyloop.proxy import PolicyRouter, _POLICY, policy_from_path, with_system_prompt


def test_policy_from_path():
    assert policy_from_path("/v1/chat/completions") is None
    assert policy_from_path("/r/policy/cand-1/v1/chat/completions") == "cand-1"
    assert policy_from_path("/r/run/x/policy/base/v1/messages") == "base"
    assert policy_from_path("/r/run/x/v1/chat/completions") is None
    assert policy_from_path("/r/odd/v1/chat/completions") is None
    assert policy_from_path("/admin/status") is None


def test_with_system_prompt():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    out = with_system_prompt(body, "You are a receptionist.")
    assert out["messages"][0] == {"role": "system", "content": "You are a receptionist."}
    assert out["messages"][1]["content"] == "hi"
    assert body["messages"][0]["role"] == "user"  # input untouched
    already = {"messages": [{"role": "system", "content": "mine"}, {"role": "user", "content": "hi"}]}
    assert with_system_prompt(already, "other") is already
    assert with_system_prompt(body, None) is body


class FakeClient:
    def __init__(self, name):
        self.name = name
        self.calls = 0

    async def sample_async(self, *a, **kw):
        self.calls += 1
        await asyncio.sleep(0)
        return self.name


def test_router_dispatches_by_contextvar_and_persists(tmp_path):
    made = {}

    def make(path):
        made[path] = FakeClient(path or "base")
        return made[path]

    live = FakeClient("live")
    r = PolicyRouter(make, live, "live-label", tmp_path / "policies.json")
    r.register("cand", "tinker://cand")
    r.register("base", None)

    async def call(policy):
        _POLICY.set(policy)
        return await r.sample_async()

    async def main():
        # concurrent requests in separate tasks see their own policy
        return await asyncio.gather(call(None), call("cand"), call("base"), call("cand"))

    assert asyncio.run(main()) == ["live", "tinker://cand", "base", "tinker://cand"]
    assert r.public()["cand"]["sampler_path"] == "tinker://cand"
    # a restarted proxy re-registers from disk
    r2 = PolicyRouter(make, FakeClient("live2"), "l2", tmp_path / "policies.json")
    assert set(r2.policies) == {"cand", "base"}
    assert r2.policies["cand"]["sampler_path"] == "tinker://cand"
    # unknown attributes fall through to the live client
    assert r2.name == "live2"
