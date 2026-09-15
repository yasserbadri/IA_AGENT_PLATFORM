"""Tests du provider Gemini natif (SDK google-genai).

Contrairement aux autres providers du projet, Gemini n'utilise pas la couche
de compatibilité OpenAI : ces tests vérifient donc surtout les conversions
aller-retour entre le format de messages de la boucle agent et le format
natif Gemini (types.Content / types.Part).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

from app.agent.providers.gemini_provider import (
    GeminiProvider,
    _to_gemini_content,
    _tool_schemas_to_gemini,
)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Recherche web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


@pytest.fixture
def provider():
    """Provider avec un client SDK mocké : aucun appel réseau réel."""
    with patch("app.agent.providers.gemini_provider.genai.Client"):
        p = GeminiProvider(model="gemini-test")
    p._client = MagicMock()
    return p


def _fake_response(parts):
    content = types.Content(role="model", parts=parts)
    return MagicMock(candidates=[MagicMock(content=content)])


class FakeAsyncPager:
    """Imite l'AsyncPager renvoyé par client.aio.models.list()."""

    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        self._idx = 0
        return self

    async def __anext__(self):
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


def test_tool_schemas_are_unwrapped_from_openai_envelope():
    tools = _tool_schemas_to_gemini(TOOL_SCHEMAS)
    declarations = tools[0].function_declarations
    assert len(declarations) == 1
    assert declarations[0].name == "web_search"
    assert declarations[0].parameters.properties["query"].type == "STRING"


def test_tool_schemas_empty_returns_none():
    # types.GenerateContentConfig(tools=None) = pas d'outils, alors qu'une
    # liste vide est refusée par le SDK.
    assert _tool_schemas_to_gemini([]) is None


def test_initial_user_message_becomes_text_part():
    content = _to_gemini_content({"role": "user", "content": "Fais une mission"})
    assert content.role == "user"
    assert content.parts[0].text == "Fais une mission"


async def test_complete_returns_text_when_no_tool_calls(provider):
    provider._client.aio.models.generate_content = AsyncMock(
        return_value=_fake_response([types.Part(text="Mission terminée.")])
    )
    turn = await provider.complete([{"role": "user", "content": "salut"}], TOOL_SCHEMAS)
    assert turn.content == "Mission terminée."
    assert turn.tool_calls == []


async def test_complete_normalizes_tool_calls(provider):
    part = types.Part(
        function_call=types.FunctionCall(id="call_1", name="web_search", args={"query": "météo"})
    )
    provider._client.aio.models.generate_content = AsyncMock(return_value=_fake_response([part]))

    turn = await provider.complete([{"role": "user", "content": "météo ?"}], TOOL_SCHEMAS)

    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "web_search"
    # Chez Gemini, `args` est déjà un dict : pas de json.loads() à faire.
    assert turn.tool_calls[0].arguments == {"query": "météo"}


async def test_complete_falls_back_to_name_when_call_has_no_id(provider):
    part = types.Part(function_call=types.FunctionCall(name="web_search", args={}))
    provider._client.aio.models.generate_content = AsyncMock(return_value=_fake_response([part]))

    turn = await provider.complete([{"role": "user", "content": "?"}], TOOL_SCHEMAS)

    assert turn.tool_calls[0].id == "web_search"


async def test_tool_result_round_trip_rebuilds_native_function_response(provider):
    """Le scénario complet de app/agent/loop.py : le modèle demande un outil,
    on ajoute sa réponse puis notre résultat à l'historique, et l'historique
    doit rester reconvertible en types.Content valides."""
    part = types.Part(
        function_call=types.FunctionCall(id="c1", name="web_search", args={"query": "x"})
    )
    provider._client.aio.models.generate_content = AsyncMock(return_value=_fake_response([part]))

    messages = [{"role": "user", "content": "cherche x"}]
    turn = await provider.complete(messages, TOOL_SCHEMAS)

    messages.append(provider.to_assistant_history_entry(turn.raw_message))
    messages.append(provider.to_tool_result_entry(turn.tool_calls[0], "résultat trouvé"))

    contents = [_to_gemini_content(m) for m in messages]

    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].function_call.name == "web_search"
    function_response = contents[2].parts[0].function_response
    assert function_response.name == "web_search"
    assert function_response.response == {"result": "résultat trouvé"}


async def test_list_models_keeps_only_generate_content_models(provider):
    provider._client.aio.models.list = AsyncMock(
        return_value=FakeAsyncPager(
            [
                types.Model(name="models/gemini-2.5-flash", supported_actions=["generateContent"]),
                types.Model(name="models/gemini-2.5-pro", supported_actions=["generateContent"]),
                types.Model(name="models/text-embedding-004", supported_actions=["embedContent"]),
            ]
        )
    )
    assert await provider.list_models() == ["gemini-2.5-flash", "gemini-2.5-pro"]


async def test_list_models_strips_prefix_and_dedupes(provider):
    provider._client.aio.models.list = AsyncMock(
        return_value=FakeAsyncPager(
            [
                types.Model(name="models/gemini-2.5-flash", supported_actions=["generateContent"]),
                types.Model(name="models/gemini-2.5-flash", supported_actions=["generateContent"]),
            ]
        )
    )
    assert await provider.list_models() == ["gemini-2.5-flash"]
