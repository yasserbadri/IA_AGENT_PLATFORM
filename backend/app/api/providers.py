import asyncio

from fastapi import APIRouter, HTTPException

from app.agent.providers import AVAILABLE_PROVIDERS, DEFAULT_MODELS, list_available_models, test_models
from app.api.schemas import ProviderInfo, ProviderModelsRead, ProviderModelsTestRead, ProviderStatus

router = APIRouter(prefix="/providers", tags=["providers"])

# Libellé humain affiché dans l'UI pour chaque provider.
PROVIDER_LABELS: dict[str, str] = {
    "mistral": "Mistral",
    "openai": "OpenAI",
    "grok": "Grok (xAI)",
    "gemini": "Gemini",
}


@router.get("/", response_model=list[ProviderInfo])
async def list_providers() -> list[ProviderInfo]:
    """Liste des providers LLM configurés, avec le modèle par défaut de chacun
    (settings.*_MODEL). Sert à peupler le sélecteur provider + modèle côté UI,
    sans dupliquer cette liste en dur dans le JS."""
    return [
        ProviderInfo(name=name, label=PROVIDER_LABELS[name], default_model=DEFAULT_MODELS[name])
        for name in AVAILABLE_PROVIDERS
    ]


@router.get("/status", response_model=list[ProviderStatus])
async def check_providers_status() -> list[ProviderStatus]:
    """Vérifie, pour les quatre providers EN MÊME TEMPS (asyncio.gather, pas
    l'un après l'autre), si leur clé API configurée fonctionne réellement —
    en tentant d'appeler leur endpoint /models. Sert à afficher d'un coup
    d'oeil quelles clés sont valides avant de lancer une mission, plutôt que
    de le découvrir après coup quand une mission échoue."""
    results = await asyncio.gather(
        *(list_available_models(name) for name in AVAILABLE_PROVIDERS),
        return_exceptions=True,
    )

    statuses: list[ProviderStatus] = []
    for name, result in zip(AVAILABLE_PROVIDERS, results):
        if isinstance(result, Exception):
            statuses.append(
                ProviderStatus(name=name, label=PROVIDER_LABELS[name], ok=False, error=str(result))
            )
        else:
            statuses.append(
                ProviderStatus(name=name, label=PROVIDER_LABELS[name], ok=True, model_count=len(result))
            )
    return statuses


@router.get("/{name}/models", response_model=ProviderModelsRead)
async def list_models_for_provider(name: str) -> ProviderModelsRead:
    """Modèles réellement disponibles pour la clé API configurée de ce
    provider, récupérés en direct auprès du provider (GET /models ou
    équivalent SDK) — pas une liste statique maintenue à la main, qui
    deviendrait vite fausse. Renvoie 502 si la clé API est absente, invalide,
    ou si le provider est injoignable ; le client peut alors retomber sur le
    modèle par défaut plutôt que bloquer la création de mission."""
    if name not in AVAILABLE_PROVIDERS:
        raise HTTPException(status_code=404, detail=f"Provider inconnu: '{name}'.")
    try:
        models = await list_available_models(name)
    except Exception as exc:  # noqa: BLE001 — erreur SDK/réseau/clé API : on la renvoie proprement au client
        raise HTTPException(
            status_code=502,
            detail=f"Impossible de récupérer les modèles disponibles pour '{name}' : {exc}",
        ) from exc
    return ProviderModelsRead(provider=name, models=models, default_model=DEFAULT_MODELS[name])


@router.get("/{name}/models/test", response_model=ProviderModelsTestRead)
async def test_models_for_provider(name: str, limit: int | None = None) -> ProviderModelsTestRead:
    """Teste RÉELLEMENT, avec un vrai appel de complétion, chaque modèle
    renvoyé par /providers/{name}/models — plutôt que de se fier uniquement
    à la liste théorique de GET /models, qui peut annoncer des modèles
    dépréciés ou restreints sur ce compte précis. `limit` permet de ne
    tester que les N premiers modèles (utile pour un aperçu rapide sans
    consommer trop de quota si la liste est longue)."""
    if name not in AVAILABLE_PROVIDERS:
        raise HTTPException(status_code=404, detail=f"Provider inconnu: '{name}'.")
    try:
        models = await list_available_models(name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Impossible de récupérer les modèles disponibles pour '{name}' : {exc}",
        ) from exc

    if limit is not None:
        models = models[:limit]

    results = await test_models(name, models)
    return ProviderModelsTestRead(provider=name, results=results)
