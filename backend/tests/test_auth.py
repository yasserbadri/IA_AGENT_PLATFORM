from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core import auth as auth_module
from app.db.session import Base, get_db
from app.main import app

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def client():
    """Même fixture que test_agent_run.py : base SQLite en mémoire + patch de
    AsyncSessionLocal pour que la tâche de fond (_execute_run) écrive dedans."""
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
            yield ac

    app.dependency_overrides.clear()
    await engine.dispose()


# --- Authentification ------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_disabled_by_default(client: AsyncClient):
    """Sans API_KEYS configuré (défaut de ce projet, comme les clés LLM),
    aucune authentification n'est exigée."""
    response = await client.post("/missions/", json={"prompt": "Sans auth"})
    assert response.status_code == 201


@pytest.mark.asyncio
async def test_request_without_key_is_rejected_once_api_keys_is_set(client: AsyncClient, monkeypatch):
    monkeypatch.setattr(auth_module.settings, "API_KEYS", "secret-key")
    response = await client.post("/missions/", json={"prompt": "Sans clé"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_request_with_wrong_key_is_rejected(client: AsyncClient, monkeypatch):
    monkeypatch.setattr(auth_module.settings, "API_KEYS", "secret-key")
    response = await client.post(
        "/missions/", json={"prompt": "Mauvaise clé"}, headers={"X-API-Key": "nope"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_request_with_correct_key_succeeds(client: AsyncClient, monkeypatch):
    monkeypatch.setattr(auth_module.settings, "API_KEYS", "secret-key,another-key")
    response = await client.post(
        "/missions/", json={"prompt": "Bonne clé"}, headers={"X-API-Key": "another-key"}
    )
    assert response.status_code == 201


@pytest.mark.asyncio
async def test_root_and_health_never_require_auth(client: AsyncClient, monkeypatch):
    """/ et /health doivent rester joignables sans clé pour le monitoring."""
    monkeypatch.setattr(auth_module.settings, "API_KEYS", "secret-key")
    assert (await client.get("/")).status_code == 200
    assert (await client.get("/health")).status_code == 200


# --- Rate limiting -----------------------------------------------------
# Chaque test utilise une clé X-API-Key dédiée : la limite est comptée par
# clé (voir app/core/rate_limit.py), donc ça isole ces tests du reste de la
# suite (qui, elle, n'envoie jamais de header et partage le bucket "IP").


@pytest.mark.asyncio
async def test_create_mission_rate_limit_returns_429(client: AsyncClient):
    """20/minute sur la création de mission (app/api/missions.py)."""
    headers = {"X-API-Key": "rate-limit-test-create"}
    for _ in range(20):
        response = await client.post("/missions/", json={"prompt": "spam"}, headers=headers)
        assert response.status_code == 201

    response = await client.post("/missions/", json={"prompt": "spam"}, headers=headers)
    assert response.status_code == 429


@pytest.mark.asyncio
async def test_run_mission_rate_limit_returns_429(client: AsyncClient):
    """10/minute sur le lancement de l'agent (app/api/agent.py) — plus stricte
    que la création, car chaque run peut déclencher plusieurs appels LLM."""
    headers = {"X-API-Key": "rate-limit-test-run"}
    mission_ids = []
    for _ in range(11):
        created = (await client.post("/missions/", json={"prompt": "spam run"}, headers=headers)).json()
        mission_ids.append(created["id"])

    with patch("app.api.agent.run_agent_loop", new=AsyncMock(return_value="ok")):
        for mission_id in mission_ids[:10]:
            response = await client.post(f"/missions/{mission_id}/run", headers=headers)
            assert response.status_code == 202

        response = await client.post(f"/missions/{mission_ids[10]}/run", headers=headers)
        assert response.status_code == 429
