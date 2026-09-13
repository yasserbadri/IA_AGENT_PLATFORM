import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agent.providers.openai_compatible import OpenAICompatibleProvider


def fake_message(content, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def fake_tool_call(call_id, name, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def fake_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def provider():
    with patch("app.agent.providers.openai_compatible.AsyncOpenAI"):
        return OpenAICompatibleProvider("openai", "fake-key", "gpt-4o-mini")


@pytest.mark.asyncio
async def test_complete_returns_text_when_no_tool_calls(provider):
    response = fake_response(fake_message("Réponse directe."))
    with patch.object(provider._client.chat.completions, "create", new=AsyncMock(return_value=response)):
        turn = await provider.complete([{"role": "user", "content": "salut"}], [])

    assert turn.content == "Réponse directe."
    assert turn.tool_calls == []


@pytest.mark.asyncio
async def test_complete_normalizes_tool_calls(provider):
    tc = fake_tool_call("call_1", "sql_query", {"query": "SELECT 1"})
    response = fake_response(fake_message(None, [tc]))
    with patch.object(provider._client.chat.completions, "create", new=AsyncMock(return_value=response)):
        turn = await provider.complete([{"role": "user", "content": "test"}], [])

    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "call_1"
    assert turn.tool_calls[0].name == "sql_query"
    assert turn.tool_calls[0].arguments == {"query": "SELECT 1"}


@pytest.mark.asyncio
async def test_complete_content_none_becomes_empty_string(provider):
    response = fake_response(fake_message(None, [fake_tool_call("call_1", "web_search", {"query": "x"})]))
    with patch.object(provider._client.chat.completions, "create", new=AsyncMock(return_value=response)):
        turn = await provider.complete([{"role": "user", "content": "test"}], [])

    assert turn.content == ""


def test_to_assistant_history_entry_includes_openai_shaped_tool_calls(provider):
    tc = fake_tool_call("call_1", "web_search", {"query": "x"})
    raw = fake_message("", [tc])

    entry = provider.to_assistant_history_entry(raw)

    assert entry["role"] == "assistant"
    assert entry["tool_calls"][0]["id"] == "call_1"
    assert entry["tool_calls"][0]["type"] == "function"
    assert entry["tool_calls"][0]["function"]["name"] == "web_search"


def test_to_tool_result_entry_shape(provider):
    from app.agent.providers.base import NormalizedToolCall

    tool_call = NormalizedToolCall(id="call_1", name="web_search", arguments={})
    entry = provider.to_tool_result_entry(tool_call, "18°C")

    assert entry == {"role": "tool", "tool_call_id": "call_1", "content": "18°C"}


class FakeModel:
    def __init__(self, model_id: str):
        self.id = model_id


class FakeModelPage:
    def __init__(self, data: list[FakeModel]):
        self.data = data


@pytest.mark.asyncio
async def test_list_models_filters_out_non_chat_models(provider):
    page = FakeModelPage([
        FakeModel("gpt-4o"),
        FakeModel("gpt-4o-mini"),
        FakeModel("text-embedding-3-small"),
        FakeModel("whisper-1"),
        FakeModel("dall-e-3"),
        FakeModel("tts-1"),
    ])
    with patch.object(provider._client.models, "list", new=AsyncMock(return_value=page)):
        models = await provider.list_models()

    assert models == ["gpt-4o", "gpt-4o-mini"]


@pytest.mark.asyncio
async def test_list_models_returns_sorted_ids(provider):
    page = FakeModelPage([FakeModel("grok-4-0709"), FakeModel("grok-2-mini")])
    with patch.object(provider._client.models, "list", new=AsyncMock(return_value=page)):
        models = await provider.list_models()

    assert models == ["grok-2-mini", "grok-4-0709"]
