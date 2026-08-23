from unittest.mock import patch

import pytest

from app.agent.tools import read_file


@pytest.mark.asyncio
async def test_read_file_returns_content(tmp_path):
    (tmp_path / "note.txt").write_text("Bonjour le monde", encoding="utf-8")

    with patch("app.core.config.settings.FILES_DIR", str(tmp_path)):
        result = await read_file.run("note.txt")

    assert result == "Bonjour le monde"


@pytest.mark.asyncio
async def test_read_file_missing_file_returns_clear_error(tmp_path):
    with patch("app.core.config.settings.FILES_DIR", str(tmp_path)):
        result = await read_file.run("absent.txt")

    assert "n'existe pas" in result


@pytest.mark.asyncio
async def test_read_file_blocks_path_traversal(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (tmp_path / "secret.txt").write_text("top secret", encoding="utf-8")

    with patch("app.core.config.settings.FILES_DIR", str(sandbox)):
        result = await read_file.run("../secret.txt")

    assert "non autorisé" in result


@pytest.mark.asyncio
async def test_read_file_truncates_large_file(tmp_path):
    (tmp_path / "big.txt").write_text("x" * 6000, encoding="utf-8")

    with patch("app.core.config.settings.FILES_DIR", str(tmp_path)):
        result = await read_file.run("big.txt")

    assert "tronqué" in result
    assert len(result) < 6000
