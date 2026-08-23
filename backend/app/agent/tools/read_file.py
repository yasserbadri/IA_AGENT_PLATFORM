"""
Outil read_file : permet à l'agent de lire le contenu d'un fichier texte
partagé, pour des missions comme "résume le contenu de rapport.txt".

Sécurité :
- Le chemin demandé est toujours résolu puis vérifié comme étant À
  L'INTÉRIEUR de FILES_DIR (protection contre le path traversal, ex:
  "../../etc/passwd" ou un chemin absolu détourné).
- Taille de fichier plafonnée pour ne pas noyer le contexte du LLM.
"""

from pathlib import Path

from app.core.config import settings

MAX_CHARS = 5000

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Lit le contenu d'un fichier texte dans le répertoire partagé de la "
            "plateforme. Donne uniquement le nom du fichier, jamais un chemin absolu."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Nom du fichier à lire, ex: rapport.txt"}
            },
            "required": ["filename"],
        },
    },
}


async def run(filename: str) -> str:
    base_dir = Path(settings.FILES_DIR).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    candidate = (base_dir / filename).resolve()

    if not candidate.is_relative_to(base_dir):
        return "Erreur: chemin de fichier non autorisé."
    if not candidate.is_file():
        return f"Erreur: le fichier '{filename}' n'existe pas dans le répertoire partagé."

    content = candidate.read_text(encoding="utf-8", errors="replace")

    if len(content) > MAX_CHARS:
        return content[:MAX_CHARS] + f"\n... (tronqué, fichier de {len(content)} caractères au total)"
    return content
