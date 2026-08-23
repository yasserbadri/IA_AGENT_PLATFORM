import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class MissionStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Mission(Base):
    """Une mission = la tâche en langage naturel donnée par l'utilisateur à l'agent."""

    __tablename__ = "missions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[MissionStatus] = mapped_column(
        Enum(MissionStatus, name="mission_status"), default=MissionStatus.PENDING
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    runs: Mapped[list["AgentRun"]] = relationship(back_populates="mission", cascade="all, delete-orphan")


class AgentRun(Base):
    """Une exécution de l'agent pour une mission donnée (une mission peut être relancée)."""

    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    mission_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("missions.id"), nullable=False)
    status: Mapped[MissionStatus] = mapped_column(
        Enum(MissionStatus, name="run_status"), default=MissionStatus.PENDING
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    mission: Mapped["Mission"] = relationship(back_populates="runs")
    tool_calls: Mapped[list["ToolCall"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    report: Mapped["Report"] = relationship(back_populates="run", uselist=False, cascade="all, delete-orphan")


class ToolCall(Base):
    """Log d'un appel d'outil par l'agent (web search, SQL, fichier, API...)."""

    __tablename__ = "tool_calls"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_runs.id"), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False)
    input_payload: Mapped[str] = mapped_column(Text, nullable=False)   # JSON sérialisé
    output_payload: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON sérialisé
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped["AgentRun"] = relationship(back_populates="tool_calls")


class Report(Base):
    """Le rapport final généré par l'agent à la fin d'une exécution."""

    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_runs.id"), unique=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped["AgentRun"] = relationship(back_populates="report")
