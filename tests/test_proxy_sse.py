import json

from polyloop.proxy import to_sse


def test_to_sse_one_chunk_stream():
    completion = {"id": "c1", "created": 1, "model": "m", "choices": [{"index": 0, "finish_reason": "stop",
                  "message": {"role": "assistant", "content": "Hello there."}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    events = [l[6:] for l in to_sse(completion).decode().split("\n\n") if l.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": "Hello there."}
    assert chunks[1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[2]["usage"]["completion_tokens"] == 2


def test_to_sse_tool_calls_get_indexes():
    completion = {"choices": [{"message": {"role": "assistant", "content": None,
                  "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                  "finish_reason": "tool_calls"}]}
    chunks = [json.loads(l[6:]) for l in to_sse(completion).decode().split("\n\n") if l.startswith("data: {")]
    assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert "content" not in chunks[0]["choices"][0]["delta"]
    assert chunks[1]["choices"][0]["finish_reason"] == "tool_calls"
