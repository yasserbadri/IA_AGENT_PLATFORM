import json
from unittest.mock import AsyncMock, patch

import pytest

from app.agent.loop import run_agent_loop


class FakeFunctionCall:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = json.dumps(arguments)


class FakeToolCall:
    def __init__(self, name: str, arguments: dict, call_id: str = "call_123"):
        self.id = call_id
        self.function = FakeFunctionCall(name, arguments)


class FakeMessage:
    def __init__(self, content: str | None, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeChoice:
    def __init__(self, message: FakeMessage):
        self.message = message


class FakeResponse:
    def __init__(self, message: FakeMessage):
        self.choices = [FakeChoice(message)]


@pytest.mark.asyncio
async def test_agent_loop_returns_direct_text_without_tool_use():
    """Si le LLM répond directement (pas de tool_calls), la boucle s'arrête au 1er tour."""
    fake_response = FakeResponse(FakeMessage("Réponse directe, sans outil."))

    with patch("app.agent.loop.client.chat.complete_async", new=AsyncMock(return_value=fake_response)):
        result = await run_agent_loop("Une question simple")

    assert result == "Réponse directe, sans outil."


@pytest.mark.asyncio
async def test_agent_loop_executes_tool_then_returns_final_text():
    """Le LLM demande un outil au tour 1, puis répond en texte au tour 2."""
    tool_call_response = FakeResponse(
        FakeMessage(None, [FakeToolCall("web_search", {"query": "météo Tunis"})])
    )
    final_response = FakeResponse(FakeMessage("Il fait beau à Tunis."))

    mock_complete = AsyncMock(side_effect=[tool_call_response, final_response])
    logged_calls = []

    async def fake_logger(name, tool_input, output):
        logged_calls.append((name, tool_input, output))

    with patch("app.agent.loop.client.chat.complete_async", new=mock_complete), \
         patch("app.agent.loop.execute_tool", new=AsyncMock(return_value="18°C, ensoleillé")):
        result = await run_agent_loop("Quelle météo à Tunis ?", tool_call_logger=fake_logger)

    assert result == "Il fait beau à Tunis."
    assert mock_complete.call_count == 2
    assert logged_calls == [("web_search", {"query": "météo Tunis"}, "18°C, ensoleillé")]


@pytest.mark.asyncio
async def test_agent_loop_flattens_list_content_with_citations():
    """Mistral renvoie parfois message.content comme une liste de chunks
    (texte + références citées) plutôt qu'une simple chaîne, notamment quand
    il cite les sources d'un outil comme web_search. On doit toujours obtenir
    du texte brut en retour, sinon l'insertion en base plante (colonne texte)."""

    class FakeTextChunk:
        def __init__(self, text: str):
            self.text = text

    class FakeReferenceChunk:
        def __init__(self, reference_ids: list[int]):
            self.reference_ids = reference_ids
            # pas d'attribut .text, comme les vrais chunks de référence Mistral

    list_content = [
        FakeTextChunk("Il fait beau à Tunis"),
        FakeReferenceChunk([1, 2]),
        FakeTextChunk(", environ 29°C."),
    ]
    fake_response = FakeResponse(FakeMessage(list_content))

    with patch("app.agent.loop.client.chat.complete_async", new=AsyncMock(return_value=fake_response)):
        result = await run_agent_loop("Quelle météo à Tunis ?")

    assert result == "Il fait beau à Tunis, environ 29°C."
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_agent_loop_stops_after_max_turns():
    """Si le LLM demande un outil à l'infini, la boucle s'arrête à MAX_TURNS (pas de boucle infinie)."""
    infinite_tool_response = FakeResponse(
        FakeMessage(None, [FakeToolCall("web_search", {"query": "x"})])
    )
    mock_complete = AsyncMock(return_value=infinite_tool_response)

    with patch("app.agent.loop.client.chat.complete_async", new=mock_complete), \
         patch("app.agent.loop.execute_tool", new=AsyncMock(return_value="résultat")):
        result = await run_agent_loop("Mission impossible à conclure")

    assert "n'a pas pu être terminée" in result
