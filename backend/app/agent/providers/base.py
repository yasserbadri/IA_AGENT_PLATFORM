"""
Types partagés par tous les providers LLM (Mistral, OpenAI, xAI/Grok, Gemini).

Chaque provider parle un format légèrement différent (structure des
tool_calls, forme de `message.content`...), mais app/agent/loop.py n'a pas
besoin de le savoir : il manipule uniquement ProviderTurn/NormalizedToolCall,
et délègue au provider lui-même la construction des messages d'historique
dans SON format natif (to_assistant_history_entry / to_tool_result_entry),
puisque ces messages ne seront de toute façon jamais relus par un autre
provider — une mission entière utilise un seul provider du début à la fin.
"""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class NormalizedToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ProviderTurn:
    content: str
    tool_calls: list[NormalizedToolCall]
    raw_message: Any  # l'objet message natif du SDK, réutilisé pour reconstruire l'historique


class LLMProvider(Protocol):
    name: str

    async def complete(self, messages: list[dict], tools: list[dict]) -> ProviderTurn: ...

    def to_assistant_history_entry(self, raw_message: Any) -> dict: ...

    def to_tool_result_entry(self, tool_call: NormalizedToolCall, output: str) -> dict: ...
