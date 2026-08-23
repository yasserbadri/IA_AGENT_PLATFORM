"""
Le coeur du projet : la boucle agent, écrite à la main (sans LangGraph) pour
bien comprendre le mécanisme avant d'ajouter un framework d'orchestration.

Le cycle (identique dans l'idée à n'importe quel provider LLM, seul le format
des messages change d'un provider à l'autre) :
1. On envoie la mission (+ historique) au LLM avec la liste des outils disponibles.
2. Si le LLM répond avec des `tool_calls`, il veut qu'on exécute un ou plusieurs
   outils. On les exécute, on remet leur résultat dans la conversation (rôle "tool").
3. On rappelle le LLM avec ce nouveau contexte. On répète jusqu'à ce qu'il
   réponde avec du texte pur (pas de tool_calls) : la mission est finie.

Un `MAX_TURNS` évite une boucle infinie si le LLM n'arrive jamais à conclure.
"""

import json
from collections.abc import Awaitable, Callable

from mistralai import Mistral

from app.agent.tools import TOOL_SCHEMAS, execute_tool
from app.core.config import settings

MAX_TURNS = 8

# Callback optionnel appelé après chaque exécution d'outil, pour logger en base.
ToolCallLogger = Callable[[str, dict, str], Awaitable[None]]

client = Mistral(api_key=settings.MISTRAL_API_KEY)


def _extract_text(content) -> str:
    """Le SDK Mistral renvoie parfois `message.content` comme une simple chaîne,
    et parfois comme une liste de "chunks" (texte + références citées), notamment
    quand le modèle cite les sources renvoyées par un outil comme web_search.
    On aplati toujours ça en texte brut avant de le stocker ou de le renvoyer,
    sinon SQLAlchemy tente d'insérer une liste d'objets dans une colonne texte."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for chunk in content:
            text = getattr(chunk, "text", None)
            if text is None and isinstance(chunk, dict):
                text = chunk.get("text")
            if text:
                parts.append(text)
        return "".join(parts)

    return "" if content is None else str(content)


async def run_agent_loop(mission_prompt: str, tool_call_logger: ToolCallLogger | None = None) -> str:
    messages = [{"role": "user", "content": mission_prompt}]

    for _turn in range(MAX_TURNS):
        response = await client.chat.complete_async(
            model=settings.MISTRAL_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=1024,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            # Le LLM a répondu directement en texte : la mission est terminée.
            return _extract_text(message.content)

        # Le LLM veut utiliser un ou plusieurs outils.
        messages.append(
            {"role": "assistant", "content": message.content, "tool_calls": message.tool_calls}
        )

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            tool_input = json.loads(tool_call.function.arguments)

            try:
                output = await execute_tool(name, tool_input)
            except Exception as exc:  # noqa: BLE001 — on veut renvoyer l'erreur au LLM, pas planter
                output = f"Erreur lors de l'exécution de l'outil: {exc}"

            if tool_call_logger is not None:
                await tool_call_logger(name, tool_input, output)

            messages.append(
                {"role": "tool", "name": name, "content": output, "tool_call_id": tool_call.id}
            )

    return "La mission n'a pas pu être terminée dans la limite de tours autorisés."
