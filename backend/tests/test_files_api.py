from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_upload_txt_file_is_stored(tmp_path):
    with patch("app.api.files.settings.FILES_DIR", str(tmp_path)):
        response = client.post(
            "/files/upload",
            files={"file": ("note.txt", b"contenu de test", "text/plain")},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "note.txt"
    assert body["size_bytes"] == len(b"contenu de test")
    assert (tmp_path / "note.txt").read_bytes() == b"contenu de test"


def test_upload_pdf_and_image_are_accepted(tmp_path):
    with patch("app.api.files.settings.FILES_DIR", str(tmp_path)):
        pdf_response = client.post(
            "/files/upload",
            files={"file": ("rapport.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        image_response = client.post(
            "/files/upload",
            files={"file": ("photo.jpg", b"\xff\xd8\xff fake jpeg", "image/jpeg")},
        )

    assert pdf_response.status_code == 201
    assert image_response.status_code == 201


def test_upload_rejects_disallowed_extension(tmp_path):
    with patch("app.api.files.settings.FILES_DIR", str(tmp_path)):
        response = client.post(
            "/files/upload",
            files={"file": ("script.exe", b"MZ fake binary", "application/octet-stream")},
        )

    assert response.status_code == 400
    assert "non autorisée" in response.json()["detail"]


def test_upload_avoids_overwriting_existing_file(tmp_path):
    with patch("app.api.files.settings.FILES_DIR", str(tmp_path)):
        first = client.post(
            "/files/upload", files={"file": ("note.txt", b"premier", "text/plain")}
        )
        second = client.post(
            "/files/upload", files={"file": ("note.txt", b"second", "text/plain")}
        )

    assert first.json()["filename"] == "note.txt"
    assert second.json()["filename"] == "note_1.txt"
    assert (tmp_path / "note.txt").read_bytes() == b"premier"
    assert (tmp_path / "note_1.txt").read_bytes() == b"second"


def test_upload_rejects_path_traversal_in_filename(tmp_path):
    with patch("app.api.files.settings.FILES_DIR", str(tmp_path)):
        response = client.post(
            "/files/upload",
            files={"file": ("../../etc/passwd.txt", b"data", "text/plain")},
        )

    assert response.status_code == 201
    # Le nom de fichier est neutralisé : rien n'est écrit hors de tmp_path.
    assert list(tmp_path.iterdir())[0].parent == tmp_path


def test_list_and_delete_files(tmp_path):
    with patch("app.api.files.settings.FILES_DIR", str(tmp_path)):
        client.post("/files/upload", files={"file": ("a.txt", b"a", "text/plain")})
        client.post("/files/upload", files={"file": ("b.txt", b"b", "text/plain")})

        listed = client.get("/files/")
        assert listed.status_code == 200
        names = {f["filename"] for f in listed.json()}
        assert names == {"a.txt", "b.txt"}

        deleted = client.delete("/files/a.txt")
        assert deleted.status_code == 204

        listed_after = client.get("/files/")
        assert {f["filename"] for f in listed_after.json()} == {"b.txt"}


def test_delete_missing_file_returns_404(tmp_path):
    with patch("app.api.files.settings.FILES_DIR", str(tmp_path)):
        response = client.delete("/files/absent.txt")

    assert response.status_code == 404
