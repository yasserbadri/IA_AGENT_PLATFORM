import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.db.models import MissionStatus

LLMProviderName = Literal["mistral", "openai", "grok", "gemini"]


class MissionCreate(BaseModel):
    prompt: str
    provider: LLMProviderName = "mistral"
    # Modèle précis à utiliser chez ce provider (ex: "gpt-4o"). Optionnel :
    # si omis ou vide, le provider utilise son modèle par défaut configuré
    # côté serveur (settings.*_MODEL) — voir GET /providers pour la liste des
    # modèles par défaut de chaque provider.
    model: str | None = None


class MissionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    prompt: str
    status: MissionStatus
    provider: str
    model: str | None
    created_at: datetime
    updated_at: datetime


class ToolCallRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tool_name: str
    input_payload: str
    output_payload: str | None
    error: str | None
    created_at: datetime


class ReportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    content: str
    created_at: datetime


class ProviderInfo(BaseModel):
    name: LLMProviderName
    label: str
    default_model: str


class ProviderModelsRead(BaseModel):
    provider: LLMProviderName
    models: list[str]
    default_model: str


class ModelTestResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    ok: bool
    # Aperçu de la réponse du modèle si ok=True, message d'erreur si ok=False.
    detail: str | None = None


class ProviderModelsTestRead(BaseModel):
    provider: LLMProviderName
    results: list[ModelTestResult]


class ProviderStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: LLMProviderName
    label: str
    ok: bool
    model_count: int | None = None
    error: str | None = None


class UploadedFileRead(BaseModel):
    filename: str
    size_bytes: int
    content_type: str | None = None
    uploaded_at: datetime


class RunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: MissionStatus
    started_at: datetime
    finished_at: datetime | None
    tool_calls: list[ToolCallRead]
    report: ReportRead | None
