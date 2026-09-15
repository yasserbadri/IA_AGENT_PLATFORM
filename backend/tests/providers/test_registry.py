from unittest.mock import AsyncMock, patch

import pytest

from app.agent import providers as providers_module
from app.agent.providers import AVAILABLE_PROVIDERS, get_provider
from app.agent.providers.gemini_provider import GeminiProvider
from app.agent.providers.mistral_provider import MistralProvider
from app.agent.providers.openai_compatible import OpenAICompatibleProvider


@pytest.fixture(autouse=True)
def clear_provider_cache():
    """Le registre met en cache les instances de provider — on le vide entre
    chaque test pour ne pas laisser fuiter un mock d'un test à l'autre."""
    providers_module._PROVIDERS.clear()
    yield
    providers_module._PROVIDERS.clear()


def test_available_providers_lists_all_four():
    assert set(AVAILABLE_PROVIDERS) == {"mistral", "openai", "grok", "gemini"}


def test_get_provider_mistral_returns_mistral_provider():
    with patch("app.agent.providers.mistral_provider.Mistral"):
        provider = get_provider("mistral")
    assert isinstance(provider, MistralProvider)
    assert provider.name == "mistral"


def test_get_provider_openai_returns_openai_compatible():
    with patch("app.agent.providers.openai_compatible.AsyncOpenAI"):
        provider = get_provider("openai")
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.name == "openai"


def test_get_provider_grok_uses_xai_base_url():
    with patch("app.agent.providers.openai_compatible.AsyncOpenAI") as mock_client:
        get_provider("grok")
    _, kwargs = mock_client.call_args
    assert kwargs["base_url"] == "https://api.x.ai/v1"


def test_get_provider_gemini_uses_native_sdk_with_configured_key():
    # Gemini n'utilise plus la couche de compatibilité OpenAI (voir
    # gemini_provider.py : les clés "AQ." générées par défaut depuis 2026 ne
    # sont pas acceptées par l'endpoint OpenAI-compatible) mais le SDK natif
    # google-genai, qui gère les deux formats de clé de façon transparente.
    with patch("app.agent.providers.gemini_provider.genai.Client") as mock_client:
        provider = get_provider("gemini")
    assert isinstance(provider, GeminiProvider)
    _, kwargs = mock_client.call_args
    assert kwargs["api_key"] == providers_module.settings.GEMINI_API_KEY


def test_get_provider_unknown_name_raises():
    with pytest.raises(ValueError):
        get_provider("dall-e-mais-pas-du-tout")


def test_get_provider_caches_instances():
    with patch("app.agent.providers.mistral_provider.Mistral"):
        first = get_provider("mistral")
        second = get_provider("mistral")
    assert first is second


def test_get_provider_defaults_to_settings_default(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.DEFAULT_LLM_PROVIDER", "mistral")
    with patch("app.agent.providers.mistral_provider.Mistral"):
        provider = get_provider(None)
    assert provider.name == "mistral"


def test_get_provider_with_explicit_model_overrides_default():
    with patch("app.agent.providers.openai_compatible.AsyncOpenAI"):
        provider = get_provider("openai", "gpt-4o")
    assert provider._model == "gpt-4o"


def test_get_provider_without_model_falls_back_to_default():
    with patch("app.agent.providers.openai_compatible.AsyncOpenAI"):
        provider = get_provider("openai")
    from app.core.config import settings

    assert provider._model == settings.OPENAI_MODEL


def test_get_provider_caches_instances_per_model():
    """Deux modèles différents pour le même provider -> deux instances distinctes."""
    with patch("app.agent.providers.openai_compatible.AsyncOpenAI"):
        default = get_provider("openai")
        overridden = get_provider("openai", "gpt-4o")
        same_override_again = get_provider("openai", "gpt-4o")
    assert default is not overridden
    assert overridden is same_override_again


@pytest.mark.asyncio
async def test_list_available_models_delegates_to_provider(monkeypatch):
    from app.agent import providers as providers_module

    providers_module._MODELS_CACHE.clear()

    with patch("app.agent.providers.openai_compatible.AsyncOpenAI"):
        provider = get_provider("openai")
    provider.list_models = AsyncMock(return_value=["gpt-4o", "gpt-4o-mini"])

    models = await providers_module.list_available_models("openai")

    assert models == ["gpt-4o", "gpt-4o-mini"]
    provider.list_models.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_available_models_uses_cache_on_second_call():
    from app.agent import providers as providers_module

    providers_module._MODELS_CACHE.clear()

    with patch("app.agent.providers.openai_compatible.AsyncOpenAI"):
        provider = get_provider("openai")
    provider.list_models = AsyncMock(return_value=["gpt-4o"])

    await providers_module.list_available_models("openai")
    await providers_module.list_available_models("openai")

    provider.list_models.assert_awaited_once()  # deuxième appel servi depuis le cache


@pytest.mark.asyncio
async def test_list_available_models_propagates_sdk_errors():
    from app.agent import providers as providers_module

    providers_module._MODELS_CACHE.clear()

    with patch("app.agent.providers.openai_compatible.AsyncOpenAI"):
        provider = get_provider("openai")
    provider.list_models = AsyncMock(side_effect=RuntimeError("clé API invalide"))

    with pytest.raises(RuntimeError):
        await providers_module.list_available_models("openai")
