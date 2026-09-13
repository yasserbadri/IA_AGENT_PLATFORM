"""
Provider générique pour tout backend compatible avec le SDK OpenAI : OpenAI
lui-même, mais aussi xAI/Grok (docs.x.ai) et Google Gemini (couche de
compatibilité officielle sur generativelanguage.googleapis.com), qui exposent
tous les deux une API "chat completions" dans le même format qu'OpenAI —
il suffit de changer `base_url` et la clé API.

Une seule implémentation, trois instances différentes (voir __init__.py).
"""

import json

from openai import AsyncOpenAI

from app.agent.providers.base import NormalizedToolCall, ProviderTurn


class OpenAICompatibleProvider:
    def __init__(self, name: str, api_key: str, model: str, base_url: str | None = None) -> None:
        self.name = name
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def complete(self, messages: list[dict], tools: list[dict]) -> ProviderTurn:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1024,
        )
        message = response.choices[0].message

        tool_calls = [
            NormalizedToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=_safe_json_loads(tc.function.arguments),
            )
            for tc in (message.tool_calls or [])
        ]

        return ProviderTurn(content=message.content or "", tool_calls=tool_calls, raw_message=message)

    def to_assistant_history_entry(self, raw_message) -> dict:
        entry: dict = {"role": "assistant", "content": raw_message.content}
        if raw_message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in raw_message.tool_calls
            ]
        return entry

    def to_tool_result_entry(self, tool_call: NormalizedToolCall, output: str) -> dict:
        return {"role": "tool", "tool_call_id": tool_call.id, "content": output}

    async def list_models(self) -> list[str]:
        """Interroge GET /models avec la clé API configurée pour ce provider :
        la liste reflète exactement ce que cette clé peut utiliser (au lieu
        d'une liste de noms de modèles maintenue à la main, qui deviendrait
        vite fausse ou incomplète). On filtre les modèles qui ne sont
        clairement pas des modèles de chat (embeddings, TTS, génération
        d'image/vidéo...) car ils ne fonctionnent pas avec la boucle agent."""
        page = await self._client.models.list()
        ids = (m.id for m in page.data)
        return sorted(m for m in ids if _looks_like_chat_model(m))


# Sous-chaînes indiquant qu'un modèle n'est pas un modèle de chat texte
# classique (donc inutilisable par la boucle agent, qui envoie toujours un
# historique de chat + une liste d'outils). Volontairement large : mieux
# vaut cacher un modèle de chat par erreur que proposer un modèle qui va
# planter à la première mission.
_NON_CHAT_KEYWORDS = (
    "embedding", "embed", "whisper", "tts", "audio", "realtime",
    "moderation", "dall-e", "image", "imagen", "video", "veo",
    "davinci", "babbage", "computer-use",
)


def _looks_like_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(keyword in lowered for keyword in _NON_CHAT_KEYWORDS)


def _safe_json_loads(raw: str) -> dict:
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
