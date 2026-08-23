import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import MissionCreate, MissionRead
from app.db.models import Mission
from app.db.session import get_db

router = APIRouter(prefix="/missions", tags=["missions"])


@router.post("/", response_model=MissionRead, status_code=201)
async def create_mission(payload: MissionCreate, db: AsyncSession = Depends(get_db)) -> Mission:
    mission = Mission(prompt=payload.prompt)
    db.add(mission)
    await db.commit()
    await db.refresh(mission)
    return mission


@router.get("/", response_model=list[MissionRead])
async def list_missions(db: AsyncSession = Depends(get_db)) -> list[Mission]:
    result = await db.execute(select(Mission).order_by(Mission.created_at.desc()))
    return list(result.scalars().all())


@router.get("/{mission_id}", response_model=MissionRead)
async def get_mission(mission_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> Mission:
    mission = await db.get(Mission, mission_id)
    if mission is None:
        raise HTTPException(status_code=404, detail="Mission not found")
    return mission
