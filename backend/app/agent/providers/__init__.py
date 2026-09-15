"""
Registre central des providers LLM. app/agent/loop.py appelle get_provider(name, model)
et n'a jamais besoin de savoir quel SDK ou quel base_url se cache derrière.

Les clients sont créés une seule fois par couple (provider, modèle) et réutilisés,
comme pour l'ancien client Mistral module-level — pas de coût de connexion par mission.
Chaque provider a un modèle par défaut (settings.*_MODEL), mais une mission peut
choisir n'importe quel autre modèle disponible pour la clé API de ce provider
(ex: passer de "gpt-4o-mini" à "gpt-4o" sur la même clé OpenAI) — voir
list_available_models(), qui interroge l'API du provider plutôt que de
maintenir une liste de noms de modèles à la main.
"""

import asyncio
import time

from app.agent.providers.base import LLMProvider
from app.agent.providers.gemini_provider import GeminiProvider
from app.agent.providers.mistral_provider import MistralProvider
from app.agent.providers.openai_compatible import OpenAICompatibleProvider
from app.core.config import settings

XAI_BASE_URL = "https://api.x.ai/v1"

_PROVIDERS: dict[tuple[str, str | None], LLMProvider] = {}

AVAILABLE_PROVIDERS = ("mistral", "openai", "grok", "gemini")

# Modèle par défaut pour chaque provider, tel que configuré via les variables
# d'environnement — utilisé comme valeur de repli quand une mission ne précise
# pas de modèle, et exposé tel quel par GET /providers pour préremplir l'UI.
DEFAULT_MODELS: dict[str, str] = {
    "mistral": settings.MISTRAL_MODEL,
    "openai": settings.OPENAI_MODEL,
    "grok": settings.XAI_MODEL,
    "gemini": settings.GEMINI_MODEL,
}

# Cache de la liste des modèles par provider (name -> (timestamp, models)).
# Appeler /v1/models à chaque frappe clavier dans l'UI serait inutile et
# consommerait du quota API pour rien ; la liste des modèles disponibles
# pour une clé donnée ne change quasiment jamais en cours de session.
_MODELS_CACHE: dict[str, tuple[float, list[str]]] = {}
_MODELS_CACHE_TTL_SECONDS = 300

# Variable d'environnement portant la clé API de chaque provider, utilisée
# pour produire un message d'erreur explicite quand elle est absente plutôt
# que de laisser le SDK échouer avec une erreur d'authentification obscure.
_API_KEY_ENV_VARS: dict[str, str] = {
    "mistral": "MISTRAL_API_KEY",
    "openai": "OPENAI_API_KEY",
    "grok": "XAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class MissingAPIKeyError(RuntimeError):
    """Levée quand la clé API d'un provider n'est pas configurée du tout.

    Distinguer ce cas d'une clé présente mais refusée change le diagnostic
    côté UI : ici il n'y a rien à débugger côté réseau ou quota, il manque
    simplement une ligne dans le .env.
    """


def _api_key_for(name: str) -> str:
    return {
        "mistral": settings.MISTRAL_API_KEY,
        "openai": settings.OPENAI_API_KEY,
        "grok": settings.XAI_API_KEY,
        "gemini": settings.GEMINI_API_KEY,
    }.get(name, "")


def ensure_api_key_configured(name: str) -> None:
    if not _api_key_for(name).strip():
        raise MissingAPIKeyError(
            f"Clé API absente : la variable {_API_KEY_ENV_VARS[name]} n'est pas renseignée "
            f"dans le fichier .env. Ajoute-la puis redémarre le serveur."
        )


def default_model_for(name: str) -> str:
    try:
        return DEFAULT_MODELS[name]
    except KeyError:
        raise ValueError(
            f"Provider LLM inconnu: '{name}'. Providers disponibles: {', '.join(AVAILABLE_PROVIDERS)}."
        ) from None


def _build_provider(name: str, model: str | None) -> LLMProvider:
    if name == "mistral":
        return MistralProvider(model=model)
    if name == "openai":
        return OpenAICompatibleProvider("openai", settings.OPENAI_API_KEY, model or settings.OPENAI_MODEL)
    if name == "grok":
        return OpenAICompatibleProvider(
            "grok", settings.XAI_API_KEY, model or settings.XAI_MODEL, base_url=XAI_BASE_URL
        )
    if name == "gemini":
        return GeminiProvider(model=model)
    raise ValueError(
        f"Provider LLM inconnu: '{name}'. Providers disponibles: {', '.join(AVAILABLE_PROVIDERS)}."
    )


def get_provider(name: str | None = None, model: str | None = None) -> LLMProvider:
    name = name or settings.DEFAULT_LLM_PROVIDER
    cache_key = (name, model)
    if cache_key not in _PROVIDERS:
        _PROVIDERS[cache_key] = _build_provider(name, model)
    return _PROVIDERS[cache_key]


async def list_available_models(name: str, *, use_cache: bool = True) -> list[str]:
    """Modèles réellement utilisables par la clé API configurée pour ce
    provider, récupérés en direct via GET /models (ou l'équivalent SDK) —
    plutôt qu'une liste statique qui deviendrait vite fausse. Lève
    MissingAPIKeyError si la clé n'est pas configurée, sinon l'erreur du SDK
    sous-jacent (clé invalide, réseau...) telle quelle ; à l'appelant de la
    traduire en réponse HTTP propre."""
    ensure_api_key_configured(name)

    if use_cache:
        cached = _MODELS_CACHE.get(name)
        if cached is not None and (time.monotonic() - cached[0]) < _MODELS_CACHE_TTL_SECONDS:
            return cached[1]

    provider = get_provider(name)
    list_models = getattr(provider, "list_models", None)
    if list_models is None:
        raise NotImplementedError(f"list_models() non supporté pour le provider '{name}'.")

    models = await list_models()
    _MODELS_CACHE[name] = (time.monotonic(), models)
    return models


# Nombre d'appels de complétion réels lancés en parallèle lors d'un test de
# modèles. Volontairement bas : on ne veut pas déclencher un rate limit chez
# le provider juste pour peupler l'UI, et ces appels consomment un (petit)
# peu de quota réel sur la clé de l'utilisateur.
_MODEL_TEST_CONCURRENCY = 3
_MODEL_TEST_MESSAGES = [{"role": "user", "content": "Réponds juste 'ok'."}]


async def _test_one_model(name: str, model: str, semaphore: asyncio.Semaphore) -> "ModelTestResult":
    from app.api.schemas import ModelTestResult  # import tardif : évite un cycle schemas -> providers

    async with semaphore:
        try:
            provider = get_provider(name, model)
            turn = await provider.complete(_MODEL_TEST_MESSAGES, tools=[])
            preview = (turn.content or "").strip().replace("\n", " ")[:80]
            return ModelTestResult(model=model, ok=True, detail=preview or None)
        except Exception as exc:  # noqa: BLE001 — erreur SDK/réseau/modèle : renvoyée telle quelle à l'appelant
            return ModelTestResult(model=model, ok=False, detail=str(exc)[:300])


async def test_models(name: str, models: list[str] | None = None) -> list["ModelTestResult"]:
    """Teste RÉELLEMENT chaque modèle de `models` (ou, si omis, la liste
    complète renvoyée par list_available_models) avec un vrai appel de
    complétion minimal — contrairement à list_available_models(), qui ne fait
    que refléter ce que GET /models annonce sans garantir qu'un appel de
    complétion va réellement aboutir (modèle déprécié, accès restreint sur ce
    compte, etc). Lancés avec une concurrence bornée pour éviter le rate
    limit côté provider."""
    ensure_api_key_configured(name)

    if models is None:
        models = await list_available_models(name)

    semaphore = asyncio.Semaphore(_MODEL_TEST_CONCURRENCY)
    return await asyncio.gather(*(_test_one_model(name, model, semaphore) for model in models))
