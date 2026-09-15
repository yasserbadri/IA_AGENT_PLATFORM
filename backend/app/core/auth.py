"""
Authentification basique par clé API, appliquée aux routes qui coûtent de
l'argent ou exposent des données (missions, agent, providers) — pas à `/` ni
`/health`, qui doivent rester accessibles pour le monitoring/orchestrateur.

Mécanisme volontairement simple (une seule dépendance FastAPI, pas de JWT, pas
de table "users") : le client envoie sa clé dans le header `X-API-Key`, on la
compare à la liste `settings.API_KEYS`. Suffisant pour un usage interne/petite
équipe ; si des comptes utilisateurs distincts avec des permissions différentes
deviennent nécessaires, ce fichier est l'endroit à remplacer par un vrai
système d'auth (OAuth2/JWT).

Comme les clés des providers LLM, l'authentification est opt-in : si
`API_KEYS` est vide, elle est désactivée (pratique en dev local, cohérent avec
le reste du projet) — mais un warning est loggé pour ne pas l'oublier en prod.
"""

import logging

from fastapi import Header, HTTPException, status

from app.core.config import settings

logger = logging.getLogger(__name__)

_warned_no_auth = False


async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    global _warned_no_auth

    if not settings.API_KEYS:
        if not _warned_no_auth:
            logger.warning(
                "AUTH DESACTIVEE : API_KEYS est vide, toutes les routes protégées sont "
                "accessibles sans clé. Configure API_KEYS dans .env avant un déploiement réel."
            )
            _warned_no_auth = True
        return

    valid_keys = {key.strip() for key in settings.API_KEYS.split(",") if key.strip()}
    if x_api_key is None or x_api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API manquante ou invalide (header 'X-API-Key').",
        )
