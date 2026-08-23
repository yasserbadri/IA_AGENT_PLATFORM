"""
Outil sql_query : permet à l'agent d'interroger la base Postgres de la
plateforme elle-même, en lecture seule, pour répondre à des questions comme
"combien de missions ont échoué cette semaine ?" ou "quel est le taux de
succès des runs ce mois-ci ?".

Sécurité (c'est un agent autonome qui écrit ses propres requêtes, donc plus
strict qu'un simple filtre anti-injection classique) :
- Seules les requêtes commençant par SELECT sont acceptées.
- Un mot-clé de modification (INSERT/UPDATE/DELETE/DROP/ALTER/...) n'importe
  où dans la requête la rejette (bloque les CTE avec effets de bord type
  `WITH x AS (DELETE ... RETURNING *) SELECT * FROM x`).
- La transaction est systématiquement annulée (ROLLBACK), même en cas de
  succès : double sécurité, cet outil ne doit jamais pouvoir committer.
- Nombre de lignes renvoyées plafonné pour ne pas noyer le contexte du LLM.
"""

import re

from sqlalchemy import text

from app.db.session import AsyncSessionLocal

MAX_ROWS = 50
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|grant|revoke|create)\b", re.IGNORECASE
)

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "sql_query",
        "description": (
            "Exécute une requête SQL en lecture seule (SELECT uniquement) sur la base de "
            "la plateforme. Tables disponibles : missions (id, prompt, status, created_at, "
            "updated_at), agent_runs (id, mission_id, status, started_at, finished_at), "
            "tool_calls (id, run_id, tool_name, input_payload, output_payload, error, "
            "created_at), reports (id, run_id, content, created_at)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "La requête SQL SELECT à exécuter"}
            },
            "required": ["query"],
        },
    },
}


async def run(query: str) -> str:
    stripped = query.strip().rstrip(";")

    if not stripped.lower().startswith("select"):
        return "Erreur: seules les requêtes SELECT sont autorisées."

    if _FORBIDDEN_KEYWORDS.search(stripped):
        return "Erreur: cette requête contient un mot-clé de modification non autorisé."

    async with AsyncSessionLocal() as session:
        try:
            result = await session.execute(text(stripped))
            rows = result.fetchmany(MAX_ROWS)
            columns = list(result.keys())
        except Exception as exc:  # noqa: BLE001 — on renvoie l'erreur SQL au LLM, pas planter
            return f"Erreur SQL: {exc}"
        finally:
            await session.rollback()  # jamais de commit possible depuis cet outil

    if not rows:
        return "Aucune ligne retournée."

    lines = [", ".join(columns)]
    lines.extend(", ".join(str(value) for value in row) for row in rows)

    suffix = f"\n... (résultat tronqué à {MAX_ROWS} lignes)" if len(rows) == MAX_ROWS else ""
    return "\n".join(lines) + suffix
