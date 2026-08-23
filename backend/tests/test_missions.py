import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.db.session import Base, get_db
from app.main import app

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def client():
    """Client de test avec une base SQLite en mémoire, isolée entre chaque test."""
    engine = create_async_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

    async def override_get_db():
        async with TestSessionLocal() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.asyncio
async def test_create_and_get_mission(client: AsyncClient):
    response = await client.post("/missions/", json={"prompt": "Résume les actus IA de la semaine"})
    assert response.status_code == 201
    created = response.json()
    assert created["status"] == "pending"
    assert created["prompt"] == "Résume les actus IA de la semaine"

    response = await client.get(f"/missions/{created['id']}")
    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


@pytest.mark.asyncio
async def test_list_missions(client: AsyncClient):
    await client.post("/missions/", json={"prompt": "Mission 1"})
    await client.post("/missions/", json={"prompt": "Mission 2"})

    response = await client.get("/missions/")
    assert response.status_code == 200
    assert len(response.json()) == 2


@pytest.mark.asyncio
async def test_get_missing_mission_returns_404(client: AsyncClient):
    response = await client.get("/missions/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
