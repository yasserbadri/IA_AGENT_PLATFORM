from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.tools import web_search


@pytest.mark.asyncio
async def test_web_search_returns_answer_and_results():
    fake_json = {
        "answer": "Il fait 29°C et ensoleillé à Tunis.",
        "results": [
            {"title": "Météo Tunis", "content": "Prévisions détaillées...", "url": "https://example.com/meteo"},
        ],
    }
    fake_response = MagicMock()
    fake_response.json.return_value = fake_json
    fake_response.raise_for_status = MagicMock()

    with patch("app.core.config.settings.TAVILY_API_KEY", "fake-key"), \
         patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake_response)):
        result = await web_search.run("météo à Tunis")

    assert "29°C" in result
    assert "Météo Tunis" in result


@pytest.mark.asyncio
async def test_web_search_without_api_key_returns_clear_error():
    with patch("app.core.config.settings.TAVILY_API_KEY", ""):
        result = await web_search.run("peu importe")

    assert "TAVILY_API_KEY" in result


@pytest.mark.asyncio
async def test_web_search_no_results_returns_explicit_message():
    fake_response = MagicMock()
    fake_response.json.return_value = {"answer": None, "results": []}
    fake_response.raise_for_status = MagicMock()

    with patch("app.core.config.settings.TAVILY_API_KEY", "fake-key"), \
         patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake_response)):
        result = await web_search.run("xyzzy_inexistant")

    assert "Aucun résultat" in result
