"""
Rate limiting basé sur `slowapi` (wrapper FastAPI de la lib `limits`).

Ciblé sur les deux endpoints qui coûtent vraiment de l'argent ou des
ressources : créer une mission et lancer l'agent (chaque run peut déclencher
plusieurs appels LLM + outils). Les routes de lecture (GET /missions/...) ne
sont pas limitées ici — un excès de lecture n'a pas le même impact qu'un excès
d'appels LLM.

Stockage en mémoire (`memory://`) : suffisant tant que l'API tourne en un seul
process (cas actuel du docker-compose). Si un jour plusieurs instances de
l'API tournent derrière un load balancer, remplacer par
`storage_uri=settings.REDIS_URL` (déjà dispo dans le projet) pour partager les
compteurs entre instances.

Clé de limitation : la clé API du client si elle est fournie (header
`X-API-Key`), sinon son adresse IP. Ça permet de limiter par client plutôt que
de mélanger tout le monde derrière une seule IP (utile derrière un proxy) —
et ça marche même quand l'authentification est désactivée (API_KEYS vide).
"""

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def rate_limit_key(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    return api_key or get_remote_address(request)


limiter = Limiter(
    key_func=rate_limit_key,
    # Par défaut slowapi scope les limites par URL EXACTE ("url"), donc
    # POST /missions/<id-1>/run et POST /missions/<id-2>/run compteraient
    # comme deux endpoints différents avec des compteurs séparés — ce qui
    # viderait la limite de tout son sens sur une route avec un paramètre
    # dans le chemin. "endpoint" scope par fonction de vue à la place.
    key_style="endpoint",
)
