"""
Outil call_api : permet à l'agent d'appeler une API HTTP externe (GET/POST),
pour des missions comme "vérifie le statut de ce endpoint" ou "récupère ces
données JSON publiques".

Sécurité :
- Seuls http/https sont acceptés, GET/POST uniquement.
- Blocage basique des cibles internes/privées (protection SSRF) : loopback,
  réseaux privés (10.x, 172.16-31.x, 192.168.x), link-local. Ce n'est PAS une
  protection exhaustive de niveau production (pas de vérif DNS-rebinding par
  ex.) — pour un vrai déploiement, préférer un allowlist explicite de domaines.
- Timeout court, réponse tronquée pour ne pas noyer le contexte du LLM.
"""

import ipaddress
import socket
from urllib.parse import urlparse

import httpx

MAX_CHARS = 3000
ALLOWED_METHODS = {"GET", "POST"}

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "call_api",
        "description": (
            "Appelle une API HTTP externe (GET ou POST) et renvoie la réponse. "
            "Utile pour récupérer des données publiques depuis un service tiers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL complète de l'API à appeler"},
                "method": {"type": "string", "enum": ["GET", "POST"], "description": "Méthode HTTP"},
                "body": {"type": "string", "description": "Corps JSON optionnel (pour POST)"},
            },
            "required": ["url"],
        },
    },
}


def _is_private_target(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True  # ne résout pas -> on refuse par prudence

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return True
    return False


async def run(url: str, method: str = "GET", body: str | None = None) -> str:
    method = method.upper()
    if method not in ALLOWED_METHODS:
        return f"Erreur: méthode '{method}' non autorisée (GET ou POST uniquement)."

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "Erreur: seules les URLs http/https sont autorisées."
    if not parsed.hostname:
        return "Erreur: URL invalide."
    if _is_private_target(parsed.hostname):
        return "Erreur: appel vers une cible interne/privée refusé."

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.request(method, url, content=body)
    except httpx.HTTPError as exc:
        return f"Erreur réseau: {exc}"

    truncated = response.text[:MAX_CHARS]
    suffix = "\n... (réponse tronquée)" if len(response.text) > MAX_CHARS else ""
    return f"Statut: {response.status_code}\n{truncated}{suffix}"
