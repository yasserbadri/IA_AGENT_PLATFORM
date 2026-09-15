"""
Outil read_file : permet à l'agent de lire le contenu d'un fichier partagé
(texte, PDF ou photo), pour des missions comme "résume rapport.pdf" ou
"que dit ce ticket.jpg ?". Les fichiers arrivent dans le répertoire partagé
soit déposés manuellement, soit via POST /files/upload (voir app/api/files.py).

- Fichiers texte (.txt, .md, .csv, .json, .log...) : lus tels quels.
- PDF (.pdf) : texte extrait page par page avec pypdf. Un PDF scanné sans
  couche de texte ne renvoie rien d'exploitable (voir message renvoyé).
- Images (.png, .jpg, .jpeg, .webp, .gif, .bmp, .tiff) : deux analyses
  combinées quand possible :
    1. OCR (pytesseract) pour le texte visible sur l'image.
    2. Description par le modèle de vision du provider de la mission en
       cours (si le provider/modèle le supporte — voir describe_image() sur
       chaque provider dans app/agent/providers/).

Sécurité :
- Le chemin demandé est toujours résolu puis vérifié comme étant À
  L'INTÉRIEUR de FILES_DIR (protection contre le path traversal, ex:
  "../../etc/passwd" ou un chemin absolu détourné).
- Taille de contenu plafonnée pour ne pas noyer le contexte du LLM.
"""

import base64
import mimetypes
from pathlib import Path

from app.core.config import settings

MAX_CHARS = 5000

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Lit le contenu d'un fichier dans le répertoire partagé de la plateforme : "
            "fichier texte (contenu brut), PDF (texte extrait automatiquement) ou photo "
            "(texte détecté par OCR + description par un modèle de vision si disponible). "
            "Donne uniquement le nom du fichier, jamais un chemin absolu."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Nom du fichier à lire, ex: rapport.pdf ou photo.jpg",
                }
            },
            "required": ["filename"],
        },
    },
}


async def run(filename: str, provider=None) -> str:
    base_dir = Path(settings.FILES_DIR).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    candidate = (base_dir / filename).resolve()

    if not candidate.is_relative_to(base_dir):
        return "Erreur: chemin de fichier non autorisé."
    if not candidate.is_file():
        return f"Erreur: le fichier '{filename}' n'existe pas dans le répertoire partagé."

    suffix = candidate.suffix.lower()

    if suffix in PDF_EXTENSIONS:
        content = _extract_pdf_text(candidate)
    elif suffix in IMAGE_EXTENSIONS:
        content = await _extract_image_content(candidate, provider)
    else:
        content = candidate.read_text(encoding="utf-8", errors="replace")

    if len(content) > MAX_CHARS:
        return content[:MAX_CHARS] + f"\n... (tronqué, contenu de {len(content)} caractères au total)"
    return content


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "Erreur: le support PDF n'est pas installé sur le serveur (dépendance 'pypdf' manquante)."

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001 — PDF corrompu/chiffré : message clair plutôt qu'un stack trace
        return f"Erreur: impossible de lire le PDF '{path.name}' ({exc})."

    pages_text = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages_text.append(f"--- Page {i + 1} ---\n{text}")

    if not pages_text:
        return (
            f"Le PDF '{path.name}' ne contient pas de texte extractible "
            "(probablement un scan sans couche de texte — essaie de l'envoyer "
            "sous forme d'image pour bénéficier de l'OCR)."
        )
    return "\n\n".join(pages_text)


async def _extract_image_content(path: Path, provider) -> str:
    parts: list[str] = []

    ocr_text = _ocr_image(path)
    if ocr_text:
        parts.append(f"[Texte détecté sur l'image par OCR]\n{ocr_text}")

    description = await _describe_image_with_vision(path, provider)
    if description:
        parts.append(f"[Description de l'image par le modèle de vision]\n{description}")

    if not parts:
        return (
            f"Image '{path.name}' lue, mais aucun texte n'a été détecté par OCR et aucune "
            "description par un modèle de vision n'a pu être obtenue."
        )
    return "\n\n".join(parts)


def _ocr_image(path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""

    try:
        with Image.open(path) as img:
            text = pytesseract.image_to_string(img, lang="fra+eng")
        return text.strip()
    except Exception:  # noqa: BLE001 — OCR indisponible/échoué : on continue avec la vision seule
        return ""


async def _describe_image_with_vision(path: Path, provider) -> str:
    describe = getattr(provider, "describe_image", None)
    if describe is None:
        return ""

    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"

    try:
        image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        result = await describe(image_b64, mime_type)
        return (result or "").strip()
    except Exception as exc:  # noqa: BLE001 — modèle sans vision, quota, réseau... : on continue avec l'OCR seul
        return f"(analyse par modèle de vision indisponible : {exc})"
