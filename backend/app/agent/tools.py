"""
Chaque outil a deux parties :
1. Un schéma JSON (TOOL_SCHEMAS) décrivant à l'API Anthropic ce que fait l'outil
   et quels paramètres il attend — c'est ce que le LLM "voit" pour décider de l'appeler.
2. Une fonction Python réelle qui l'exécute, appelée depuis execute_tool().

On commence avec UN SEUL outil (recherche web) pour bien comprendre le mécanisme
avant d'en ajouter d'autres (SQL, fichier, API externe — étape suivante du plan).

NOTE: web_search utilise l'API Tavily (https://tavily.com), conçue spécifiquement
pour être consommée par des agents LLM plutôt que par un humain dans un navigateur :
résultats déjà résumés (`answer`), contenu de page extrait (pas juste un lien),
et bien plus fiable que l'API "Instant Answer" de DuckDuckGo sur les questions
d'actualité ou non-encyclopédiques. Tier gratuit : 1000 requêtes/mois, une clé
API est nécessaire (voir TAVILY_API_KEY dans la config).
"""

import httpx

from app.core.config import settings

TOOL_SCHEMAS = [
    {
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
]


async def _web_search(query: str) -> str:
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


async def execute_tool(name: str, tool_input: dict) -> str:
    """Route un appel d'outil demandé par le LLM vers son implémentation réelle."""
    if name == "web_search":
        return await _web_search(tool_input["query"])

    raise ValueError(f"Outil inconnu: {name}")
