from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.tools import call_api


@pytest.mark.asyncio
async def test_call_api_success():
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.text = '{"ok": true}'

    with patch("app.agent.tools.call_api._is_private_target", return_value=False), \
         patch("httpx.AsyncClient.request", new=AsyncMock(return_value=fake_response)):
        result = await call_api.run("https://api.example.com/status")

    assert "200" in result
    assert "ok" in result


@pytest.mark.asyncio
async def test_call_api_rejects_bad_scheme():
    result = await call_api.run("ftp://example.com/file")
    assert "http/https" in result


@pytest.mark.asyncio
async def test_call_api_rejects_bad_method():
    result = await call_api.run("https://example.com", method="DELETE")
    assert "non autorisée" in result


@pytest.mark.asyncio
async def test_call_api_blocks_private_target():
    result = await call_api.run("http://127.0.0.1:8000/secret")
    assert "interne" in result or "privée" in result
