"""
Provider Mistral — utilise le SDK officiel `mistralai` (pas la couche de
compatibilité OpenAI) : c'est le provider historique du projet, déjà testé
de bout en bout.
"""

import json

from mistralai import Mistral

from app.agent.providers.base import NormalizedToolCall, ProviderTurn
from app.core.config import settings


def _extract_text(content) -> str:
    """Le SDK Mistral renvoie parfois `message.content` comme une simple chaîne,
    et parfois comme une liste de "chunks" (texte + références citées), notamment
    quand le modèle cite les sources renvoyées par un outil comme web_search.
    On aplatit toujours ça en texte brut, sinon SQLAlchemy tente d'insérer une
    liste d'objets dans une colonne texte."""
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


class MistralProvider:
    name = "mistral"

    def __init__(self, model: str | None = None) -> None:
        self._client = Mistral(api_key=settings.MISTRAL_API_KEY)
        # `model` permet de surcharger le modèle par défaut (settings.MISTRAL_MODEL)
        # au cas par cas, par ex. quand une mission choisit explicitement un modèle.
        self._model = model or settings.MISTRAL_MODEL

    async def complete(self, messages: list[dict], tools: list[dict]) -> ProviderTurn:
        response = await self._client.chat.complete_async(
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
                arguments=json.loads(tc.function.arguments),
            )
            for tc in (message.tool_calls or [])
        ]

        return ProviderTurn(content=_extract_text(message.content), tool_calls=tool_calls, raw_message=message)

    def to_assistant_history_entry(self, raw_message) -> dict:
        return {"role": "assistant", "content": raw_message.content, "tool_calls": raw_message.tool_calls}

    def to_tool_result_entry(self, tool_call: NormalizedToolCall, output: str) -> dict:
        return {"role": "tool", "name": tool_call.name, "content": output, "tool_call_id": tool_call.id}

    async def describe_image(self, image_b64: str, mime_type: str) -> str:
        """Décrit/transcrit une image via le SDK Mistral (même format de
        contenu "image_url" que l'API OpenAI). Ne fonctionne qu'avec un
        modèle Mistral supportant la vision (ex: pixtral-*) — si le modèle
        courant ne la supporte pas, l'appel échoue et l'appelant (read_file)
        retombe simplement sur l'OCR seul."""
        response = await self._client.chat.complete_async(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Décris cette image en détail et en français, pour quelqu'un "
                                "qui ne peut pas la voir. Si elle contient du texte, "
                                "retranscris-le intégralement."
                            ),
                        },
                        {"type": "image_url", "image_url": f"data:{mime_type};base64,{image_b64}"},
                    ],
                }
            ],
            max_tokens=600,
        )
        return _extract_text(response.choices[0].message.content)

    async def list_models(self) -> list[str]:
        """Interroge GET /v1/models avec la clé API configurée. Ne garde que
        les modèles qui supportent à la fois le chat et le function calling
        (obligatoire pour la boucle agent, qui donne toujours une liste
        d'outils au LLM) — un modèle sans function calling planterait dès la
        première mission qui a besoin d'un outil."""
        response = await self._client.models.list_async()
        cards = response.data or [] if response else []
        ids = {
            card.id
            for card in cards
            if getattr(card.capabilities, "completion_chat", False)
            and getattr(card.capabilities, "function_calling", False)
        }
        return sorted(ids)
