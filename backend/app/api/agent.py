import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agent.loop import run_agent_loop
from app.api.schemas import RunRead
from app.db.models import AgentRun, Mission, MissionStatus, Report, ToolCall
from app.db.session import AsyncSessionLocal, get_db

router = APIRouter(prefix="/missions", tags=["agent"])


@router.post("/{mission_id}/run", status_code=202)
async def run_mission(
    mission_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Démarre l'exécution en tâche de fond et répond immédiatement (202 Accepted).
    L'appel LLM + les tool calls peuvent prendre plusieurs secondes voire
    dizaines de secondes (jusqu'à MAX_TURNS allers-retours) : bloquer la requête
    HTTP pendant tout ce temps empêchait l'interface de rester réactive. Le
    client suit la progression via GET /runs, qui se met à jour au fur et à
    mesure (chaque tool call est committé en base dès qu'il se termine)."""
    mission = await db.get(Mission, mission_id)
    if mission is None:
        raise HTTPException(status_code=404, detail="Mission not found")
    if mission.status == MissionStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Cette mission est déjà en cours d'exécution.")

    run = AgentRun(mission_id=mission.id, status=MissionStatus.RUNNING)
    mission.status = MissionStatus.RUNNING
    db.add(run)
    await db.commit()
    await db.refresh(run)

    background_tasks.add_task(_execute_run, run.id, mission.prompt)

    return {"run_id": str(run.id), "status": run.status.value}


async def _execute_run(run_id: uuid.UUID, mission_prompt: str) -> None:
    """Le vrai travail de la boucle agent, exécuté après que la réponse HTTP est
    déjà partie. Utilise sa propre session DB (celle de la requête d'origine est
    fermée dès que la réponse est envoyée) — c'est le même pattern que les outils
    sql_query/read_file, qui n'ont eux non plus jamais accès à une session de requête."""
    async with AsyncSessionLocal() as db:

        async def log_tool_call(name: str, tool_input: dict, output: str) -> None:
            db.add(
                ToolCall(
                    run_id=run_id,
                    tool_name=name,
                    input_payload=json.dumps(tool_input),
                    output_payload=output,
                )
            )
            await db.commit()

        run = await db.get(AgentRun, run_id)
        mission = await db.get(Mission, run.mission_id)

        try:
            final_text = await run_agent_loop(
                mission_prompt,
                tool_call_logger=log_tool_call,
                provider=mission.provider,
                model=mission.model,
            )
        except Exception as exc:  # noqa: BLE001 — tâche de fond : on marque FAILED, on ne relève jamais
            run.status = MissionStatus.FAILED
            run.finished_at = datetime.now(timezone.utc)
            mission.status = MissionStatus.FAILED
            await db.commit()
            return

        report = Report(run_id=run.id, content=final_text)
        run.status = MissionStatus.DONE
        run.finished_at = datetime.now(timezone.utc)
        mission.status = MissionStatus.DONE
        db.add(report)
        await db.commit()


@router.get("/{mission_id}/runs", response_model=list[RunRead])
async def list_mission_runs(mission_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> list[AgentRun]:
    """Historique complet des exécutions d'une mission : statut, trace des outils appelés
    (dans l'ordre), et rapport final. C'est via cet endpoint (pollé par l'interface)
    que le client observe la progression d'un run lancé en tâche de fond."""
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
