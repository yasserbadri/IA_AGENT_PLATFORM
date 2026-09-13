"""
Le coeur du projet : la boucle agent, écrite à la main (sans LangGraph) pour
bien comprendre le mécanisme avant d'ajouter un framework d'orchestration.

Le cycle est le même quel que soit le provider LLM choisi (Mistral, OpenAI,
xAI/Grok, Gemini — voir app/agent/providers/) :
1. On envoie la mission (+ historique) au LLM avec la liste des outils disponibles.
2. Si le LLM répond avec des `tool_calls`, il veut qu'on exécute un ou plusieurs
   outils. On les exécute, on remet leur résultat dans la conversation (rôle "tool").
3. On rappelle le LLM avec ce nouveau contexte. On répète jusqu'à ce qu'il
   réponde avec du texte pur (pas de tool_calls) : la mission est finie.

Un `MAX_TURNS` évite une boucle infinie si le LLM n'arrive jamais à conclure.
"""

from collections.abc import Awaitable, Callable

from app.agent.providers import get_provider
from app.agent.tools import TOOL_SCHEMAS, execute_tool

MAX_TURNS = 8

# Callback optionnel appelé après chaque exécution d'outil, pour logger en base.
ToolCallLogger = Callable[[str, dict, str], Awaitable[None]]


async def run_agent_loop(
    mission_prompt: str,
    tool_call_logger: ToolCallLogger | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    llm = get_provider(provider, model)
    messages = [{"role": "user", "content": mission_prompt}]

    for _turn in range(MAX_TURNS):
        turn = await llm.complete(messages, TOOL_SCHEMAS)

        if not turn.tool_calls:
            # Le LLM a répondu directement en texte : la mission est terminée.
            return turn.content

        # Le LLM veut utiliser un ou plusieurs outils.
        messages.append(llm.to_assistant_history_entry(turn.raw_message))

        for tool_call in turn.tool_calls:
            try:
                output = await execute_tool(tool_call.name, tool_call.arguments)
            except Exception as exc:  # noqa: BLE001 — on veut renvoyer l'erreur au LLM, pas planter
                output = f"Erreur lors de l'exécution de l'outil: {exc}"

            if tool_call_logger is not None:
                await tool_call_logger(tool_call.name, tool_call.arguments, output)

            messages.append(llm.to_tool_result_entry(tool_call, output))

    return "La mission n'a pas pu être terminée dans la limite de tours autorisés."
