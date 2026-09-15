from unittest.mock import AsyncMock, MagicMock, patch

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


@pytest.mark.asyncio
async def test_read_file_extracts_pdf_text(tmp_path):
    (tmp_path / "rapport.pdf").write_bytes(b"%PDF-1.4 fake bytes, content mocked below")

    fake_page = MagicMock()
    fake_page.extract_text.return_value = "Chiffre d'affaires en hausse de 12%."
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]

    with patch("app.core.config.settings.FILES_DIR", str(tmp_path)), \
         patch("pypdf.PdfReader", return_value=fake_reader):
        result = await read_file.run("rapport.pdf")

    assert "Chiffre d'affaires" in result
    assert "Page 1" in result


@pytest.mark.asyncio
async def test_read_file_pdf_without_text_layer_gives_clear_message(tmp_path):
    (tmp_path / "scan.pdf").write_bytes(b"%PDF-1.4 fake bytes")

    fake_page = MagicMock()
    fake_page.extract_text.return_value = ""
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]

    with patch("app.core.config.settings.FILES_DIR", str(tmp_path)), \
         patch("pypdf.PdfReader", return_value=fake_reader):
        result = await read_file.run("scan.pdf")

    assert "ne contient pas de texte extractible" in result


@pytest.mark.asyncio
async def test_read_file_image_combines_ocr_and_vision(tmp_path):
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff fake jpeg bytes")

    fake_provider = MagicMock()
    fake_provider.describe_image = AsyncMock(return_value="Une facture posée sur un bureau.")

    with patch("app.core.config.settings.FILES_DIR", str(tmp_path)), \
         patch("pytesseract.image_to_string", return_value="TOTAL: 42.00 EUR"), \
         patch("PIL.Image.open", return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))):
        result = await read_file.run("photo.jpg", provider=fake_provider)

    assert "TOTAL: 42.00 EUR" in result
    assert "Une facture posée sur un bureau." in result
    fake_provider.describe_image.assert_awaited_once()


@pytest.mark.asyncio
async def test_read_file_image_without_vision_provider_falls_back_to_ocr(tmp_path):
    (tmp_path / "photo.png").write_bytes(b"\x89PNG fake bytes")

    with patch("app.core.config.settings.FILES_DIR", str(tmp_path)), \
         patch("pytesseract.image_to_string", return_value="Bonjour"), \
         patch("PIL.Image.open", return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))):
        result = await read_file.run("photo.png", provider=None)

    assert "Bonjour" in result
    assert "modèle de vision" not in result
