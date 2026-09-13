from unittest.mock import AsyncMock, patch

import pytest

from app.agent.loop import run_agent_loop
from app.agent.providers.base import NormalizedToolCall, ProviderTurn


class FakeProvider:
    """Provider factice utilisé pour tester loop.py indépendamment de tout
    SDK réel — un seul provider testé ici suffit puisque loop.py ne parle
    qu'au travers de l'interface commune (ProviderTurn / NormalizedToolCall)."""

    name = "fake"

    def __init__(self, turns: list[ProviderTurn]):
        self._turns = iter(turns)
        self.call_count = 0

    async def complete(self, messages, tools):
        self.call_count += 1
        return next(self._turns)

    def to_assistant_history_entry(self, raw_message) -> dict:
        return {"role": "assistant", "content": "…", "raw": raw_message}

    def to_tool_result_entry(self, tool_call: NormalizedToolCall, output: str) -> dict:
        return {"role": "tool", "tool_call_id": tool_call.id, "content": output}


@pytest.mark.asyncio
async def test_agent_loop_returns_direct_text_without_tool_use():
    """Si le LLM répond directement (pas de tool_calls), la boucle s'arrête au 1er tour."""
    provider = FakeProvider([ProviderTurn(content="Réponse directe, sans outil.", tool_calls=[], raw_message=None)])

    with patch("app.agent.loop.get_provider", return_value=provider):
        result = await run_agent_loop("Une question simple")

    assert result == "Réponse directe, sans outil."
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_agent_loop_executes_tool_then_returns_final_text():
    """Le LLM demande un outil au tour 1, puis répond en texte au tour 2."""
    tool_call = NormalizedToolCall(id="call_123", name="web_search", arguments={"query": "météo Tunis"})
    provider = FakeProvider(
        [
            ProviderTurn(content="", tool_calls=[tool_call], raw_message="raw-1"),
            ProviderTurn(content="Il fait beau à Tunis.", tool_calls=[], raw_message="raw-2"),
        ]
    )

    logged_calls = []

    async def fake_logger(name, tool_input, output):
        logged_calls.append((name, tool_input, output))

    with patch("app.agent.loop.get_provider", return_value=provider), \
         patch("app.agent.loop.execute_tool", new=AsyncMock(return_value="18°C, ensoleillé")):
        result = await run_agent_loop("Quelle météo à Tunis ?", tool_call_logger=fake_logger)

    assert result == "Il fait beau à Tunis."
    assert provider.call_count == 2
    assert logged_calls == [("web_search", {"query": "météo Tunis"}, "18°C, ensoleillé")]


@pytest.mark.asyncio
async def test_agent_loop_stops_after_max_turns():
    """Si le LLM demande un outil à l'infini, la boucle s'arrête à MAX_TURNS (pas de boucle infinie)."""

    class InfiniteToolProvider(FakeProvider):
        def __init__(self):
            self.call_count = 0

        async def complete(self, messages, tools):
            self.call_count += 1
            tool_call = NormalizedToolCall(id="call_x", name="web_search", arguments={"query": "x"})
            return ProviderTurn(content="", tool_calls=[tool_call], raw_message="raw")

    provider = InfiniteToolProvider()

    with patch("app.agent.loop.get_provider", return_value=provider), \
         patch("app.agent.loop.execute_tool", new=AsyncMock(return_value="résultat")):
        result = await run_agent_loop("Mission impossible à conclure")

    assert "n'a pas pu être terminée" in result
    assert provider.call_count == 8  # MAX_TURNS
