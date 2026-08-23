import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.db.models import MissionStatus


class MissionCreate(BaseModel):
    prompt: str


class MissionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    prompt: str
    status: MissionStatus
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


class RunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: MissionStatus
    started_at: datetime
    finished_at: datetime | None
    tool_calls: list[ToolCallRead]
    report: ReportRead | None
