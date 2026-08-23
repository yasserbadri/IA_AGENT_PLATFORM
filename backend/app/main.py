from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.agent import router as agent_router
from app.api.health import router as health_router
from app.api.missions import router as missions_router
from app.core.config import settings

app = FastAPI(title=settings.APP_NAME)

app.include_router(health_router, tags=["health"])
app.include_router(missions_router)
app.include_router(agent_router)

# Interface "Mission Control" : une page unique HTML/CSS/JS, sans framework ni
# build step, servie directement par FastAPI à /ui (ne touche pas à / ni à l'API).
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


@app.get("/")
async def root() -> dict:
    return {"message": f"{settings.APP_NAME} is running", "env": settings.ENV}
