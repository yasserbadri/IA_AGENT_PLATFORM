"""
Endpoints de gestion des fichiers partagés (backend/data/, voir settings.FILES_DIR).

Un fichier uploadé ici devient immédiatement utilisable par l'agent via
l'outil `read_file` (voir app/agent/tools/read_file.py) : il suffit de
mentionner son nom dans le prompt d'une mission, ex. "résume rapport.pdf".

Formats acceptés : texte brut (.txt, .md, .csv, .json, .log), PDF (texte
extrait automatiquement par read_file) et images (OCR + description par un
modèle de vision, toujours via read_file). Les autres extensions sont
refusées : ce dossier est un espace de travail pour l'agent, pas un stockage
de fichiers générique.
"""

import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.api.schemas import UploadedFileRead
from app.core.config import settings

router = APIRouter(prefix="/files", tags=["files"])

ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".log",
    ".pdf",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff",
}


def _files_dir() -> Path:
    base_dir = Path(settings.FILES_DIR).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _sanitize_filename(raw_name: str) -> str:
    """Ne garde que le nom de fichier (jamais un chemin), et le nettoie pour
    éviter tout caractère susceptible de poser problème sur le système de
    fichiers — path traversal déjà neutralisé en amont par Path(...).name,
    mais on normalise aussi les accents/espaces pour un nom prévisible."""
    name = Path(raw_name).name
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "fichier"
    return name


def _unique_path(base_dir: Path, filename: str) -> Path:
    """Évite d'écraser un fichier existant en suffixant _1, _2... plutôt que
    de renommer avec un UUID : on garde un nom lisible, réutilisable tel
    quel par l'agent (read_file("rapport.pdf"))."""
    candidate = base_dir / filename
    if not candidate.exists():
        return candidate

    stem, suffix = Path(filename).stem, Path(filename).suffix
    i = 1
    while (base_dir / f"{stem}_{i}{suffix}").exists():
        i += 1
    return base_dir / f"{stem}_{i}{suffix}"


@router.post("/upload", response_model=UploadedFileRead, status_code=201)
async def upload_file(file: UploadFile) -> UploadedFileRead:
    original_name = file.filename or ""
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Extension '{extension}' non autorisée. Formats acceptés : {allowed}.",
        )

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop volumineux (max {settings.MAX_UPLOAD_SIZE_MB} Mo).",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Fichier vide.")

    base_dir = _files_dir()
    target = _unique_path(base_dir, _sanitize_filename(original_name))
    target.write_bytes(data)

    return UploadedFileRead(
        filename=target.name,
        size_bytes=len(data),
        content_type=file.content_type,
        uploaded_at=datetime.now(timezone.utc),
    )


@router.get("/", response_model=list[UploadedFileRead])
async def list_files() -> list[UploadedFileRead]:
    base_dir = _files_dir()
    files = []
    for path in sorted(base_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_file():
            continue
        stat = path.stat()
        files.append(
            UploadedFileRead(
                filename=path.name,
                size_bytes=stat.st_size,
                content_type=None,
                uploaded_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            )
        )
    return files


@router.delete("/{filename}", status_code=204)
async def delete_file(filename: str) -> None:
    base_dir = _files_dir()
    candidate = (base_dir / _sanitize_filename(filename)).resolve()
    if not candidate.is_relative_to(base_dir) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    candidate.unlink()
