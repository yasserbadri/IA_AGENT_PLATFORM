# AI Agent Operations Platform

Ceci est le point de départ du projet : une API FastAPI, connectée à PostgreSQL
et Redis, entièrement conteneurisée.

## Démarrer

```bash
docker-compose up --build
```

Puis va voir :
- http://localhost:8080/ → message de bienvenue
- http://localhost:8080/ui/ → interface **Mission Control** (créer une mission,
  la lancer, suivre la trace des outils en direct)
- http://localhost:8080/docs → doc Swagger auto-générée
- http://localhost:8080/health → vérifie que Postgres et Redis répondent (doit
  afficher `"database": "ok"` et `"redis": "ok"`)

> Ports hôte utilisés : API `8080`, Postgres `5433`, Redis `6380` (choisis pour
> éviter les conflits avec des services déjà lancés en local). Si l'un d'eux
> est encore pris chez toi, change-le dans `docker-compose.yml` (colonne de
> gauche du mapping `"HOST:CONTAINER"`).

## Lancer les tests

Le test fourni (`test_root.py`) ne dépend pas de la base, donc tu peux le
lancer même sans Docker :

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
pytest
```

## Structure du projet

```
ai-agent-platform/
├── docker-compose.yml
└── backend/
    ├── Dockerfile
    ├── requirements.txt
    ├── alembic.ini
    ├── pytest.ini
    ├── .env.example
    ├── migrations/
    │   ├── env.py
    │   ├── script.py.mako
    │   └── versions/          # généré par `alembic revision`
    ├── app/
    │   ├── main.py            # point d'entrée FastAPI (+ mount /ui)
    │   ├── static/index.html  # interface Mission Control (HTML/CSS/JS, sans build)
    │   ├── core/config.py     # configuration (variables d'env)
    │   ├── db/
    │   │   ├── session.py     # SQLAlchemy async (engine, session)
    │   │   ├── models.py      # Mission, AgentRun, ToolCall, Report
    │   │   └── redis_client.py
    │   ├── agent/
    │   │   ├── tools/          # un module par outil (web_search, sql_query, read_file, call_api)
    │   │   ├── providers/      # un module par LLM (mistral, openai, xAI/Grok, Gemini)
    │   │   └── loop.py         # boucle agent (function calling), agnostique du provider
    │   └── api/
    │       ├── health.py      # endpoint /health
    │       ├── schemas.py     # Pydantic: MissionCreate, MissionRead, RunRead...
    │       ├── missions.py    # CRUD /missions
    │       └── agent.py       # POST /missions/{id}/run (async) + GET /runs
    └── tests/
        ├── test_root.py
        ├── test_missions.py   # CRUD testé sur SQLite en mémoire
        ├── test_agent_loop.py # boucle agent testée avec provider mocké
        ├── test_agent_run.py  # endpoint /run (async) testé de bout en bout
        ├── tools/              # un fichier de test par outil
        └── providers/          # un fichier de test par provider LLM
```

## Prochaine étape (Étape 2 du plan) — ✅ déjà dans ce zip

Les modèles SQLAlchemy (`Mission`, `AgentRun`, `ToolCall`, `Report`), Alembic,
et un CRUD basique sur `Mission` sont déjà en place. Il te reste une seule
chose à faire : générer et appliquer la première migration (ça ne peut se
faire que contre une vraie base Postgres qui tourne, donc depuis ta machine) :

```bash
docker-compose up -d          # lance api + db + redis en arrière-plan
docker-compose exec api alembic revision --autogenerate -m "initial tables"
docker-compose exec api alembic upgrade head
```

Vérifie ensuite que ça marche :

```bash
curl -X POST http://localhost:8080/missions/ \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Résume les actus IA de la semaine"}'

curl http://localhost:8080/missions/
```

### Tests

`test_missions.py` teste le CRUD complet sans dépendre de Postgres (base
SQLite en mémoire injectée via `dependency_overrides` de FastAPI) :

```bash
cd backend
pip install -r requirements.txt
pytest -v
```

## Étape 3 — ✅ déjà dans ce zip : agent multi-outils, multi-LLM, interface incluse

Un vrai agent avec function calling natif, écrit à la main pour comprendre le
mécanisme avant d'ajouter LangGraph.

- `app/agent/tools/` — un module par outil, chacun testable indépendamment :
  - `web_search` — recherche web via **Tavily** (conçu pour les agents LLM :
    résumé déjà généré + contenu de page extrait, bien plus fiable que
    l'API "Instant Answer" de DuckDuckGo sur les questions d'actualité).
  - `sql_query` — interroge la base Postgres de la plateforme elle-même en
    lecture seule (SELECT uniquement, rollback systématique, résultats plafonnés).
  - `read_file` — lit un fichier texte dans `backend/data/` (sandboxé contre
    le path traversal).
  - `call_api` — appelle une API HTTP externe (GET/POST), avec un garde-fou
    basique contre le SSRF (cibles internes/privées bloquées).
- `app/agent/providers/` — un module par LLM, derrière une interface commune
  (`ProviderTurn`/`NormalizedToolCall`) : `mistral`, `openai`, `grok` (xAI),
  `gemini`. Choisi **par mission** (pas globalement) — voir section dédiée
  ci-dessous.
- `app/agent/loop.py` — la boucle brute, agnostique du provider : appelle le
  LLM choisi, exécute les outils demandés, boucle jusqu'à réponse finale (max
  `MAX_TURNS` tours).
- `app/api/agent.py` — `POST /missions/{id}/run` **lance la boucle en tâche de
  fond** (FastAPI `BackgroundTasks`) et répond immédiatement en `202 Accepted`
  avec le `run_id`. Chaque `ToolCall` est committé en base dès qu'il se termine,
  donc `GET /missions/{id}/runs` (pollé par l'interface toutes les 2.5s) montre
  la progression en direct plutôt que d'attendre la fin complète du run.
- `app/static/index.html` — interface **Mission Control** (`/ui/`) : créer une
  mission (avec choix du provider), la lancer, et suivre la trace des outils
  appelés en direct (entrée/sortie de chaque tool call dépliable), sans aucun
  framework front ni étape de build.

### Providers LLM disponibles

Quatre providers, choisis à la création d'une mission (`provider` dans le
payload de `POST /missions/`, ou via le menu déroulant de l'interface `/ui/`) :

| Provider | SDK utilisé | Endpoint | Modèle par défaut |
|---|---|---|---|
| `mistral` (défaut) | `mistralai` (officiel) | natif | `mistral-small-latest` |
| `openai` | `openai` (officiel) | natif | `gpt-4o-mini` |
| `grok` | `openai` (compatible) | `api.x.ai/v1` | `grok-4-0709` |
| `gemini` | `openai` (compatible) | `generativelanguage.googleapis.com/v1beta/openai/` | `gemini-3.7-flash` |

xAI (Grok) et Google (Gemini) exposent tous les deux une API compatible avec
le format OpenAI — un seul module (`app/agent/providers/openai_compatible.py`)
suffit donc pour les trois, seuls la clé API, l'URL et le modèle changent.
Mistral garde son propre SDK officiel (déjà testé de bout en bout).

Les noms de modèles évoluent vite d'un fournisseur à l'autre — si `OPENAI_MODEL`,
`XAI_MODEL` ou `GEMINI_MODEL` te renvoie une erreur "model not found", vérifie
la liste à jour sur la doc de chaque fournisseur et ajuste la variable dans `.env`.

Chaque provider n'a besoin d'être configuré (clé API) que si tu comptes
l'utiliser — une mission créée avec `provider: "openai"` sans `OPENAI_API_KEY`
échouera simplement au moment du run (statut `failed`, message d'erreur clair
dans le rapport), sans rien casser pour les autres providers.

### Pour l'utiliser

1. Crée un compte sur [console.mistral.ai](https://console.mistral.ai),
   génère une clé API (Settings → API Keys → Create new key).
2. Crée un compte sur [tavily.com](https://tavily.com) (tier gratuit,
   1000 requêtes/mois), récupère ta clé API dans le dashboard.
3. (Optionnel, un par un selon les providers que tu veux utiliser)
   - OpenAI : [platform.openai.com](https://platform.openai.com/api-keys)
   - xAI/Grok : [console.x.ai](https://console.x.ai) (pas de tier gratuit,
     crédits payants uniquement)
   - Gemini : [aistudio.google.com](https://aistudio.google.com/apikey)
     (tier gratuit disponible)
4. À la racine du projet (à côté de `docker-compose.yml`, **pas** dans
   `backend/`), copie `.env.example` en `.env` et colle tes clés :
   ```bash
   cp .env.example .env
   # puis édite .env et remplace les placeholders par tes vraies clés
   # (laisse vide les providers que tu ne comptes pas utiliser)
   ```
   Docker Compose lit ce fichier automatiquement et les injecte dans le
   conteneur `api` via `docker-compose.yml`.
5. Lance les services puis applique les migrations (la 2e ajoute la colonne
   `provider` sur `missions`) :
   ```bash
   docker compose up -d
   docker compose exec api alembic upgrade head
   ```
6. Va sur http://localhost:8080/ui/, crée une mission (choisis un provider
   dans le menu déroulant), clique "Lancer l'agent", et regarde la trace
   des outils se remplir en direct.

   Ou en ligne de commande :
   ```bash
   curl -X POST http://localhost:8080/missions/ \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Cherche la météo actuelle à Tunis et résume-la en une phrase", "provider": "openai"}'

   # récupère l'id retourné, puis :
   curl -X POST http://localhost:8080/missions/<ID>/run
   ```

### Tests

`test_agent_loop.py` teste la boucle avec un provider **factice** (aucune
dépendance à un vrai SDK) — la boucle ne connaît que l'interface commune
`ProviderTurn`/`NormalizedToolCall`.

`tests/providers/` teste chaque provider individuellement : `MistralProvider`
(y compris l'aplatissement des réponses avec citations), `OpenAICompatibleProvider`
(utilisé par openai/grok/gemini), et le registre `get_provider()` (bon `base_url`
par provider, cache d'instances, erreur claire sur un nom inconnu).

`tests/tools/` teste chaque outil (`web_search` via Tavily mocké, `sql_query`
sur SQLite en mémoire, `read_file` avec protection path-traversal, `call_api`
avec garde-fou SSRF).

`test_agent_run.py` teste `POST /missions/{id}/run` de bout en bout (202
immédiat, tool calls enregistrés en direct, échec propagé, double lancement
refusé, provider choisi bien transmis à la boucle).

Aucun de ces tests n'a besoin d'une vraie clé API ni de connexion réseau.

## Étape 4 (à venir) — orchestration LangGraph

Les outils et les providers sont solides. Prochaine étape : remplacer la
boucle manuelle par un graphe LangGraph pour gérer les cas plus complexes
(retries, state persistant via Redis, parallélisation d'outils).

## Checklist Git

```bash
git init
git add .
git commit -m "Étape 1: squelette FastAPI + Postgres + Redis via Docker"
git add -A
git commit -m "Étape 2: modèles SQLAlchemy + Alembic + CRUD Mission"
git add -A
git commit -m "Étape 3: agent (function calling) + interface + outils + multi-LLM"
```
