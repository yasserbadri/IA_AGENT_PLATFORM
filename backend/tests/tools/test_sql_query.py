from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.tools import sql_query


@pytest_asyncio.fixture
async def test_sessionmaker():
    """Un vrai moteur SQLite en mémoire avec une table `missions` factice,
    pour tester la logique SQL sans dépendre de Postgres."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.execute(text("CREATE TABLE missions (id INTEGER, prompt TEXT, status TEXT)"))
        await conn.execute(text("INSERT INTO missions VALUES (1, 'test 1', 'done')"))
        await conn.execute(text("INSERT INTO missions VALUES (2, 'test 2', 'failed')"))

    Session = async_sessionmaker(bind=engine, expire_on_commit=False)
    yield Session
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_query_select_returns_rows(test_sessionmaker):
    with patch("app.agent.tools.sql_query.AsyncSessionLocal", test_sessionmaker):
        result = await sql_query.run("SELECT id, prompt, status FROM missions ORDER BY id")

    assert "test 1" in result
    assert "test 2" in result


@pytest.mark.asyncio
async def test_sql_query_rejects_non_select():
    result = await sql_query.run("DELETE FROM missions")
    assert "SELECT" in result


@pytest.mark.asyncio
async def test_sql_query_rejects_forbidden_keyword_anywhere_in_query():
    """Même si la requête commence par SELECT, un mot-clé de modification
    plus loin (ex: via un `;` empilé) doit être bloqué."""
    result = await sql_query.run("SELECT * FROM missions; DROP TABLE missions")
    assert "non autorisé" in result


@pytest.mark.asyncio
async def test_sql_query_invalid_sql_returns_clear_error(test_sessionmaker):
    with patch("app.agent.tools.sql_query.AsyncSessionLocal", test_sessionmaker):
        result = await sql_query.run("SELECT * FROM table_qui_nexiste_pas")

    assert "Erreur SQL" in result
