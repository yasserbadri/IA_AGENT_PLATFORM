import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agent.loop import run_agent_loop
from app.api.schemas import RunRead
from app.db.models import AgentRun, Mission, MissionStatus, Report, ToolCall
from app.db.session import get_db

router = APIRouter(prefix="/missions", tags=["agent"])


@router.post("/{mission_id}/run")
async def run_mission(mission_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> dict:
    mission = await db.get(Mission, mission_id)
    if mission is None:
        raise HTTPException(status_code=404, detail="Mission not found")

    run = AgentRun(mission_id=mission.id, status=MissionStatus.RUNNING)
    mission.status = MissionStatus.RUNNING
    db.add(run)
    await db.commit()
    await db.refresh(run)

    async def log_tool_call(name: str, tool_input: dict, output: str) -> None:
        db.add(
            ToolCall(
                run_id=run.id,
                tool_name=name,
                input_payload=json.dumps(tool_input),
                output_payload=output,
            )
        )
        await db.commit()

    try:
        final_text = await run_agent_loop(mission.prompt, tool_call_logger=log_tool_call)
    except Exception as exc:  # noqa: BLE001 — on veut marquer le run comme FAILED, pas juste 500
        run.status = MissionStatus.FAILED
        run.finished_at = datetime.now(timezone.utc)
        mission.status = MissionStatus.FAILED
        await db.commit()
        raise HTTPException(status_code=500, detail=f"L'agent a échoué: {exc}") from exc

    report = Report(run_id=run.id, content=final_text)
    run.status = MissionStatus.DONE
    run.finished_at = datetime.now(timezone.utc)
    mission.status = MissionStatus.DONE
    db.add(report)
    await db.commit()

    return {"run_id": str(run.id), "report": final_text}


@router.get("/{mission_id}/runs", response_model=list[RunRead])
async def list_mission_runs(mission_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> list[AgentRun]:
    """Historique complet des exécutions d'une mission : statut, trace des outils appelés
    (dans l'ordre), et rapport final. Nécessaire pour l'interface, POST /run ne renvoyant
    le rapport qu'une seule fois au moment de l'exécution."""
    mission = await db.get(Mission, mission_id)
    if mission is None:
        raise HTTPException(status_code=404, detail="Mission not found")

    result = await db.execute(
        select(AgentRun)
        .where(AgentRun.mission_id == mission_id)
        .options(selectinload(AgentRun.tool_calls), selectinload(AgentRun.report))
        .order_by(AgentRun.started_at.desc())
    )
    runs = list(result.scalars().all())
    for run in runs:
        run.tool_calls.sort(key=lambda tc: tc.created_at)
    return runs
