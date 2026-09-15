"""
Provider Gemini natif — utilise le SDK officiel `google-genai` plutôt que la
couche de compatibilité OpenAI utilisée par le reste du projet pour les
providers OpenAI-like (voir openai_compatible.py).

Pourquoi natif et pas OpenAI-compatible, contrairement à Grok/OpenAI : depuis
2026, Google AI Studio ne génère plus que des clés "Authorization" (préfixe
"AQ.") — l'ancien format "AIza..." disparaît progressivement. Ces clés AQ.
sont des jetons de type OAuth et ne sont PAS acceptées par l'endpoint
OpenAI-compatible (generativelanguage.googleapis.com/v1beta/openai/), qui
renvoie "Multiple authentication credentials received" ou un 401
ACCESS_TOKEN_TYPE_UNSUPPORTED selon les cas. Le SDK natif `google-genai`,
lui, gère en interne aussi bien les clés AQ. que les anciennes clés AIza...
— voir https://ai.google.dev/gemini-api/docs/api-key. Passer par ce SDK
plutôt que par l'endpoint OpenAI-compatible est donc la façon robuste de
supporter les deux formats de clé sans avoir à les distinguer nous-mêmes.

Format de message interne : les entrées ajoutées à l'historique par
to_assistant_history_entry() / to_tool_result_entry() ne sont relues que par
CE provider (une mission entière n'utilise qu'un seul provider du début à la
fin, voir app/agent/loop.py) — rien n'oblige donc à rester au format
"OpenAI-style" utilisé par les autres providers de ce projet. On stocke ici
directement des types.Part natifs Gemini pour les réponses du modèle, et un
petit dict "function_response" pour nos propres résultats d'outils.
"""

from google import genai
from google.genai import types

from app.agent.providers.base import NormalizedToolCall, ProviderTurn
from app.core.config import settings


def _tool_schemas_to_gemini(tools: list[dict]) -> list[types.Tool] | None:
    """Convertit les tool schemas au format OpenAI utilisés partout ailleurs
    dans le projet (voir app/agent/tools.py) vers le format natif Gemini :
    même schéma JSON pour les paramètres, mais sans l'enveloppe
    {"type": "function", "function": {...}} propre à l'API OpenAI."""
    if not tools:
        return None
    declarations = [
        types.FunctionDeclaration(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters=fn.get("parameters"),
        )
        for fn in (tool.get("function", tool) for tool in tools)
    ]
    return [types.Tool(function_declarations=declarations)]


def _to_gemini_content(message: dict) -> types.Content:
    """Convertit une entrée de `messages` (voir app/agent/loop.py) en
    types.Content natif Gemini. Trois formats y transitent :
    - le message initial de la mission : {"role": "user", "content": "..."}
    - nos entrées "assistant" (to_assistant_history_entry) :
      {"role": "model", "parts": [types.Part, ...]} — déjà au format natif,
      réutilisées telles quelles.
    - nos entrées "tool" (to_tool_result_entry) :
      {"role": "user", "parts": [{"function_response": {...}}]} — le dict
      brut est reconverti en types.Part ici.
    """
    role = message["role"]

    if "parts" in message:
        native_parts = [
            types.Part(
                function_response=types.FunctionResponse(
                    name=part["function_response"]["name"],
                    response=part["function_response"]["response"],
                )
            )
            if isinstance(part, dict) and "function_response" in part
            else part
            for part in message["parts"]
        ]
        return types.Content(role=role, parts=native_parts)

    return types.Content(role=role, parts=[types.Part(text=message.get("content") or "")])


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str | None = None) -> None:
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._model = model or settings.GEMINI_MODEL

    async def complete(self, messages: list[dict], tools: list[dict]) -> ProviderTurn:
        contents = [_to_gemini_content(m) for m in messages]
        config = types.GenerateContentConfig(tools=_tool_schemas_to_gemini(tools))

        response = await self._client.aio.models.generate_content(
            model=self._model, contents=contents, config=config
        )

        candidate = response.candidates[0] if response.candidates else None
        content = candidate.content if candidate is not None else None
        parts = content.parts if content is not None and content.parts else []

        text = "".join(part.text for part in parts if getattr(part, "text", None))
        tool_calls = [
            NormalizedToolCall(
                # Gemini n'attribue pas toujours un id explicite aux appels
                # de fonction : à défaut, le nom suffit à retrouver l'appel
                # correspondant côté to_tool_result_entry (une mission n'a
                # jamais deux appels strictement simultanés au même outil).
                id=part.function_call.id or part.function_call.name,
                name=part.function_call.name,
                # Contrairement à OpenAI/Mistral, `args` est déjà un dict
                # structuré chez Gemini — pas de json.loads() nécessaire.
                arguments=dict(part.function_call.args or {}),
            )
            for part in parts
            if getattr(part, "function_call", None)
        ]

        return ProviderTurn(content=text, tool_calls=tool_calls, raw_message=content)

    def to_assistant_history_entry(self, raw_message) -> dict:
        # raw_message est le `content` (types.Content) du tour précédent :
        # ses parts (texte + éventuels function_call) sont déjà au format
        # natif attendu en entrée du tour suivant, on les stocke telles quelles.
        parts = list(raw_message.parts) if raw_message is not None and raw_message.parts else []
        return {"role": "model", "parts": parts}

    def to_tool_result_entry(self, tool_call: NormalizedToolCall, output: str) -> dict:
        return {
            "role": "user",
            "parts": [{"function_response": {"name": tool_call.name, "response": {"result": output}}}],
        }

    async def describe_image(self, image_b64: str, mime_type: str) -> str:
        """Décrit/transcrit une image avec le SDK natif Gemini, nativement
        multimodal (contrairement aux autres providers, pas besoin de
        data-URL : on passe les octets bruts + le type MIME)."""
        import base64

        image_bytes = base64.b64decode(image_b64)
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                (
                    "Décris cette image en détail et en français, pour quelqu'un qui ne "
                    "peut pas la voir. Si elle contient du texte, retranscris-le intégralement."
                ),
            ],
        )
        return response.text or ""

    async def list_models(self) -> list[str]:
        """Interroge GET /models avec la clé API configurée. Ne garde que les
        modèles qui supportent generateContent (chat) — les modèles
        d'embedding, TTS, image... ne fonctionnent pas avec la boucle agent,
        qui envoie toujours un historique de chat + une liste d'outils."""
        models: set[str] = set()
        page = await self._client.aio.models.list()
        async for m in page:
            actions = m.supported_actions or []
            if "generateContent" not in actions:
                continue
            name = (m.name or "").removeprefix("models/")
            if name:
                models.add(name)
        return sorted(models)
