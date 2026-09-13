from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_list_providers_returns_all_four_with_default_models():
    response = client.get("/providers/")
    assert response.status_code == 200
    body = response.json()
    names = {p["name"] for p in body}
    assert names == {"mistral", "openai", "grok", "gemini"}
    for provider in body:
        assert provider["label"]
        assert provider["default_model"]


def test_list_models_for_provider_returns_models():
    with patch(
        "app.api.providers.list_available_models", new=AsyncMock(return_value=["gpt-4o", "gpt-4o-mini"])
    ):
        response = client.get("/providers/openai/models")

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "openai"
    assert body["models"] == ["gpt-4o", "gpt-4o-mini"]
    assert body["default_model"]


def test_list_models_for_unknown_provider_returns_404():
    response = client.get("/providers/claude/models")
    assert response.status_code == 404


def test_list_models_returns_502_when_provider_call_fails():
    with patch(
        "app.api.providers.list_available_models",
        new=AsyncMock(side_effect=RuntimeError("clé API invalide")),
    ):
        response = client.get("/providers/openai/models")

    assert response.status_code == 502
    assert "openai" in response.json()["detail"]


def test_status_reports_ok_and_failed_providers_independently():
    """Une clé cassée sur un provider ne doit pas empêcher de connaître le
    statut des trois autres — chaque vérification est indépendante."""

    async def fake_list_available_models(name: str, **kwargs):
        if name == "openai":
            raise RuntimeError("401 Unauthorized")
        return ["fake-model-1", "fake-model-2"]

    with patch("app.api.providers.list_available_models", new=fake_list_available_models):
        response = client.get("/providers/status")

    assert response.status_code == 200
    body = {s["name"]: s for s in response.json()}

    assert body["openai"]["ok"] is False
    assert "401" in body["openai"]["error"]
    assert body["openai"]["model_count"] is None

    for name in ("mistral", "grok", "gemini"):
        assert body[name]["ok"] is True
        assert body[name]["model_count"] == 2
        assert body[name]["error"] is None


def test_status_checks_all_providers_concurrently_not_sequentially():
    """asyncio.gather doit lancer les 4 vérifications en parallèle : le temps
    total doit rester proche de la plus longue vérification individuelle, pas
    de leur somme."""
    import asyncio
    import time

    async def slow_list_available_models(name: str, **kwargs):
        await asyncio.sleep(0.1)
        return ["m1"]

    with patch("app.api.providers.list_available_models", new=slow_list_available_models):
        started = time.monotonic()
        response = client.get("/providers/status")
        elapsed = time.monotonic() - started

    assert response.status_code == 200
    assert elapsed < 0.3  # largement < 4 * 0.1s si ça avait été séquentiel
