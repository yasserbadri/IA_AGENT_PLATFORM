"""
Registre central des outils de l'agent. Chaque outil vit dans son propre
module, testable indépendamment, et expose deux choses :
- TOOL_SCHEMA (dict) : ce que le LLM "voit" pour décider de l'appeler.
- run(...) (coroutine) : l'implémentation réelle.

Ce fichier se contente d'agréger tout ça pour app/agent/loop.py, qui n'a pas
besoin de savoir comment chaque outil est implémenté.
"""

from app.agent.tools import call_api, read_file, sql_query, web_search

TOOL_SCHEMAS = [
    web_search.TOOL_SCHEMA,
    sql_query.TOOL_SCHEMA,
    read_file.TOOL_SCHEMA,
    call_api.TOOL_SCHEMA,
]

_HANDLERS = {
    "web_search": web_search.run,
    "sql_query": sql_query.run,
    "read_file": read_file.run,
    "call_api": call_api.run,
}


async def execute_tool(name: str, tool_input: dict, provider=None) -> str:
    """Route un appel d'outil demandé par le LLM vers son implémentation réelle.

    `provider` (l'instance LLMProvider de la mission en cours) n'est transmis
    qu'à read_file, seul outil qui en a besoin : pour décrire une image via
    le modèle de vision du provider courant (voir app/agent/tools/read_file.py).
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Outil inconnu: {name}")
    if name == "read_file":
        return await handler(**tool_input, provider=provider)
    return await handler(**tool_input)
