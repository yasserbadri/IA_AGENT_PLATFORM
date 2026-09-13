import json
from unittest.mock import AsyncMock, patch

import pytest

from app.agent.providers.mistral_provider import MistralProvider


class FakeFunctionCall:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = json.dumps(arguments)


class FakeToolCall:
    def __init__(self, name: str, arguments: dict, call_id: str = "call_123"):
        self.id = call_id
        self.function = FakeFunctionCall(name, arguments)


class FakeMessage:
    def __init__(self, content, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeChoice:
    def __init__(self, message: FakeMessage):
        self.message = message


class FakeResponse:
    def __init__(self, message: FakeMessage):
        self.choices = [FakeChoice(message)]


@pytest.fixture
def provider():
    with patch("app.agent.providers.mistral_provider.Mistral"):
        return MistralProvider()


@pytest.mark.asyncio
async def test_complete_returns_text_when_no_tool_calls(provider):
    fake_response = FakeResponse(FakeMessage("Réponse directe."))
    with patch.object(provider._client.chat, "complete_async", new=AsyncMock(return_value=fake_response)):
        turn = await provider.complete([{"role": "user", "content": "salut"}], [])

    assert turn.content == "Réponse directe."
    assert turn.tool_calls == []


@pytest.mark.asyncio
async def test_complete_normalizes_tool_calls(provider):
    fake_response = FakeResponse(
        FakeMessage(None, [FakeToolCall("web_search", {"query": "météo Tunis"})])
    )
    with patch.object(provider._client.chat, "complete_async", new=AsyncMock(return_value=fake_response)):
        turn = await provider.complete([{"role": "user", "content": "météo ?"}], [])

    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "web_search"
    assert turn.tool_calls[0].arguments == {"query": "météo Tunis"}


@pytest.mark.asyncio
async def test_complete_flattens_list_content_with_citations(provider):
    """Mistral renvoie parfois message.content comme une liste de chunks
    (texte + références citées) plutôt qu'une simple chaîne, notamment quand
    il cite les sources d'un outil comme web_search. On doit toujours obtenir
    du texte brut en retour, sinon l'insertion en base plante (colonne texte)."""

    class FakeTextChunk:
        def __init__(self, text: str):
            self.text = text

    class FakeReferenceChunk:
        def __init__(self, reference_ids: list[int]):
            self.reference_ids = reference_ids  # pas d'attribut .text, comme les vrais chunks de référence

    list_content = [
        FakeTextChunk("Il fait beau à Tunis"),
        FakeReferenceChunk([1, 2]),
        FakeTextChunk(", environ 29°C."),
    ]
    fake_response = FakeResponse(FakeMessage(list_content))

    with patch.object(provider._client.chat, "complete_async", new=AsyncMock(return_value=fake_response)):
        turn = await provider.complete([{"role": "user", "content": "météo ?"}], [])

    assert turn.content == "Il fait beau à Tunis, environ 29°C."
    assert isinstance(turn.content, str)


class FakeCapabilities:
    def __init__(self, completion_chat: bool = True, function_calling: bool = True):
        self.completion_chat = completion_chat
        self.function_calling = function_calling


class FakeModelCard:
    def __init__(self, model_id: str, capabilities: FakeCapabilities):
        self.id = model_id
        self.capabilities = capabilities


class FakeModelList:
    def __init__(self, data: list[FakeModelCard]):
        self.data = data


@pytest.mark.asyncio
async def test_list_models_keeps_only_chat_and_function_calling_models(provider):
    fake_models = FakeModelList([
        FakeModelCard("mistral-large-latest", FakeCapabilities(True, True)),
        FakeModelCard("mistral-embed", FakeCapabilities(False, False)),  # embeddings : pas du chat
        FakeModelCard("codestral-latest", FakeCapabilities(True, False)),  # chat mais pas de function calling
    ])
    with patch.object(provider._client.models, "list_async", new=AsyncMock(return_value=fake_models)):
        models = await provider.list_models()

    assert models == ["mistral-large-latest"]


@pytest.mark.asyncio
async def test_list_models_returns_empty_list_when_no_models(provider):
    with patch.object(provider._client.models, "list_async", new=AsyncMock(return_value=FakeModelList([]))):
        models = await provider.list_models()

    assert models == []
