"""
Outil web_search : recherche web via l'API Tavily, conçue pour être consommée
par des agents LLM (résumé déjà généré + contenu de page extrait), bien plus
fiable que l'API "Instant Answer" de DuckDuckGo sur les questions d'actualité.
Tier gratuit : 1000 requêtes/mois, clé API nécessaire (TAVILY_API_KEY).
"""

import httpx

from app.core.config import settings

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Recherche des informations sur le web. Utilise cet outil quand tu as "
            "besoin d'informations récentes ou que tu ne connais pas."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Les mots-clés de recherche"}
            },
            "required": ["query"],
        },
    },
}


async def run(query: str) -> str:
    if not settings.TAVILY_API_KEY:
        return (
            "Erreur de configuration: TAVILY_API_KEY n'est pas définie. "
            "Impossible d'effectuer la recherche web."
        )

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "max_results": 5,
            },
        )
        response.raise_for_status()
        data = response.json()

    answer = data.get("answer")
    results = data.get("results", [])

    if not answer and not results:
        return f"Aucun résultat clair trouvé pour '{query}'."

    parts = []
    if answer:
        parts.append(f"Résumé: {answer}")
    if results:
        formatted = "\n".join(
            f"- {r.get('title', 'Sans titre')}: {r.get('content', '')[:300]} ({r.get('url', '')})"
            for r in results
        )
        parts.append(f"Résultats:\n{formatted}")

    return "\n\n".join(parts)
