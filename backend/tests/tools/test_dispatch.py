from unittest.mock import AsyncMock, patch

import pytest

from app.agent.tools import TOOL_SCHEMAS, execute_tool


def test_all_tools_are_registered_in_schemas():
    schema_names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert schema_names == {"web_search", "sql_query", "read_file", "call_api"}


@pytest.mark.asyncio
async def test_execute_tool_routes_to_correct_handler():
    fake_handler = AsyncMock(return_value="ok")
    with patch.dict("app.agent.tools._HANDLERS", {"web_search": fake_handler}):
        result = await execute_tool("web_search", {"query": "test"})

    assert result == "ok"
    fake_handler.assert_awaited_once_with(query="test")


@pytest.mark.asyncio
async def test_execute_tool_unknown_name_raises():
    with pytest.raises(ValueError):
        await execute_tool("outil_inconnu", {})
