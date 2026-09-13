import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.session import Base, get_db
from app.main import app

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def client():
    """Client de test avec une base SQLite en mémoire, isolée entre chaque test.

    Important : la tâche de fond (_execute_run) utilise AsyncSessionLocal
    directement (import global), pas la dépendance get_db injectée par requête
    — elle survit à la requête HTTP d'origine, donc elle ne peut pas recevoir
    une session par Depends(). On patche donc app.api.agent.AsyncSessionLocal
    pour qu'elle pointe, elle aussi, vers la même base SQLite de test.
    """
    engine = create_async_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

    async def override_get_db():
        async with TestSessionLocal() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    with patch("app.api.agent.AsyncSessionLocal", TestSessionLocal):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            ac.session_factory = TestSessionLocal  # exposé pour les tests qui doivent toucher la DB directement
            yield ac

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.asyncio
async def test_run_returns_202_immediately_and_finishes_in_background(client: AsyncClient):
    created = (await client.post("/missions/", json={"prompt": "Météo à Tunis"})).json()

    with patch("app.api.agent.run_agent_loop", new=AsyncMock(return_value="Il fait beau à Tunis.")):
        response = await client.post(f"/missions/{created['id']}/run")

    # 202 Accepted : la requête ne bloque pas sur l'exécution de l'agent.
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "running"
    assert "run_id" in body

    # Avec ASGITransport, la BackgroundTask s'exécute avant que l'await ci-dessus
    # ne rende la main : au moment où on interroge /runs, le run est déjà DONE.
    runs = (await client.get(f"/missions/{created['id']}/runs")).json()
    assert len(runs) == 1
    assert runs[0]["status"] == "done"
    assert runs[0]["report"]["content"] == "Il fait beau à Tunis."

    mission = (await client.get(f"/missions/{created['id']}")).json()
    assert mission["status"] == "done"


@pytest.mark.asyncio
async def test_run_records_tool_calls_as_they_happen(client: AsyncClient):
    created = (await client.post("/missions/", json={"prompt": "Cherche la météo"})).json()

    async def fake_loop(prompt, tool_call_logger=None, provider=None, model=None):
        if tool_call_logger:
            await tool_call_logger("web_search", {"query": "météo"}, "18°C, ensoleillé")
        return "Rapport final"

    with patch("app.api.agent.run_agent_loop", new=fake_loop):
        await client.post(f"/missions/{created['id']}/run")

    runs = (await client.get(f"/missions/{created['id']}/runs")).json()
    assert len(runs[0]["tool_calls"]) == 1
    assert runs[0]["tool_calls"][0]["tool_name"] == "web_search"
    assert runs[0]["tool_calls"][0]["output_payload"] == "18°C, ensoleillé"


@pytest.mark.asyncio
async def test_run_failure_marks_mission_and_run_as_failed(client: AsyncClient):
    created = (await client.post("/missions/", json={"prompt": "Mission vouée à l'échec"})).json()

    with patch("app.api.agent.run_agent_loop", new=AsyncMock(side_effect=RuntimeError("Mistral timeout"))):
        response = await client.post(f"/missions/{created['id']}/run")

    assert response.status_code == 202  # la requête elle-même réussit toujours

    runs = (await client.get(f"/missions/{created['id']}/runs")).json()
    assert runs[0]["status"] == "failed"
    assert runs[0]["report"] is None

    mission = (await client.get(f"/missions/{created['id']}")).json()
    assert mission["status"] == "failed"


@pytest.mark.asyncio
async def test_run_uses_the_mission_chosen_provider(client: AsyncClient):
    """La mission choisit son provider à la création ; le run doit l'utiliser."""
    created = (
        await client.post("/missions/", json={"prompt": "Résume les actus IA", "provider": "openai"})
    ).json()
    assert created["provider"] == "openai"

    received_providers = []

    async def fake_loop(prompt, tool_call_logger=None, provider=None, model=None):
        received_providers.append(provider)
        return "ok"

    with patch("app.api.agent.run_agent_loop", new=fake_loop):
        await client.post(f"/missions/{created['id']}/run")

    assert received_providers == ["openai"]


@pytest.mark.asyncio
async def test_run_uses_the_mission_chosen_model(client: AsyncClient):
    """Une mission peut préciser un modèle précis pour son provider ; le run doit l'utiliser."""
    created = (
        await client.post(
            "/missions/",
            json={"prompt": "Résume les actus IA", "provider": "openai", "model": "gpt-4o"},
        )
    ).json()
    assert created["model"] == "gpt-4o"

    received_models = []

    async def fake_loop(prompt, tool_call_logger=None, provider=None, model=None):
        received_models.append(model)
        return "ok"

    with patch("app.api.agent.run_agent_loop", new=fake_loop):
        await client.post(f"/missions/{created['id']}/run")

    assert received_models == ["gpt-4o"]


@pytest.mark.asyncio
async def test_mission_without_model_defaults_to_none(client: AsyncClient):
    """Sans modèle précisé, la mission retombe sur le modèle par défaut du provider (model=None)."""
    created = (await client.post("/missions/", json={"prompt": "Mission sans modèle précisé"})).json()
    assert created["model"] is None


@pytest.mark.asyncio
async def test_mission_defaults_to_mistral_provider(client: AsyncClient):
    created = (await client.post("/missions/", json={"prompt": "Mission sans provider précisé"})).json()
    assert created["provider"] == "mistral"


@pytest.mark.asyncio
async def test_mission_rejects_unknown_provider(client: AsyncClient):
    response = await client.post(
        "/missions/", json={"prompt": "Mission avec provider invalide", "provider": "claude"}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_cannot_run_a_mission_already_running(client: AsyncClient):
    """On force le statut RUNNING directement en base (impossible à observer via
    l'API dans ce test : ASGITransport exécute la BackgroundTask de façon
    synchrone, donc un run se termine toujours avant que l'appel /run ne
    rende la main). C'est le comportement du endpoint qu'on veut vérifier ici,
    pas le timing réel d'un run en cours."""
    from app.db.models import Mission, MissionStatus

    created = (await client.post("/missions/", json={"prompt": "Mission déjà en cours"})).json()

    async with client.session_factory() as session:
        mission = await session.get(Mission, uuid.UUID(created["id"]))
        mission.status = MissionStatus.RUNNING
        await session.commit()

    response = await client.post(f"/missions/{created['id']}/run")
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_run_missing_mission_returns_404(client: AsyncClient):
    response = await client.post("/missions/00000000-0000-0000-0000-000000000000/run")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_runs_missing_mission_returns_404(client: AsyncClient):
    response = await client.get("/missions/00000000-0000-0000-0000-000000000000/runs")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_runs_empty_list_for_mission_never_run(client: AsyncClient):
    created = (await client.post("/missions/", json={"prompt": "Jamais lancée"})).json()
    runs = (await client.get(f"/missions/{created['id']}/runs")).json()
    assert runs == []
