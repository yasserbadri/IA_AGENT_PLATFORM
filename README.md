# AI Agent Operations Platform

Ceci est le point de départ du projet : une API FastAPI, connectée à PostgreSQL
et Redis, entièrement conteneurisée.

## Démarrer

```bash
docker-compose up --build
```

Puis va voir :
- http://localhost:8080/ → message de bienvenue
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
    │   ├── main.py            # point d'entrée FastAPI
    │   ├── core/config.py     # configuration (variables d'env)
    │   ├── db/
    │   │   ├── session.py     # SQLAlchemy async (engine, session)
    │   │   ├── models.py      # Mission, AgentRun, ToolCall, Report
    │   │   └── redis_client.py
    │   ├── agent/
    │   │   ├── tools.py       # schéma + implémentation des outils
    │   │   └── loop.py        # boucle agent (function calling)
    │   └── api/
    │       ├── health.py      # endpoint /health
    │       ├── schemas.py     # Pydantic: MissionCreate, MissionRead
    │       ├── missions.py    # CRUD /missions
    │       └── agent.py       # POST /missions/{id}/run
    └── tests/
        ├── test_root.py
        ├── test_missions.py   # CRUD testé sur SQLite en mémoire
        └── test_agent_loop.py # boucle agent testée avec LLM mocké
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

## Étape 3 — ✅ déjà dans ce zip : premier agent (sans framework)

Un vrai agent avec function calling natif, écrit à la main pour comprendre le
mécanisme avant d'ajouter LangGraph. LLM utilisé : **Mistral** (tier gratuit
"Experiment" sur console.mistral.ai, aucune carte bancaire requise) :

- `app/agent/tools.py` — schéma des outils (`TOOL_SCHEMAS`, format compatible
  OpenAI/Mistral) + implémentation. Un seul outil pour l'instant : `web_search`
  (via l'API **Tavily**, pensée pour des agents LLM — résumé déjà généré,
  contenu de page extrait — bien plus fiable que DuckDuckGo Instant Answer
  sur les questions d'actualité ou non-encyclopédiques).
- `app/agent/loop.py` — la boucle brute : appelle le LLM, exécute les outils
  demandés, boucle jusqu'à réponse finale (max `MAX_TURNS` tours).
- `app/api/agent.py` — endpoint `POST /missions/{id}/run` qui déclenche la
  boucle, logge chaque `ToolCall` en base, et sauvegarde le `Report` final.

### Pour l'utiliser

1. Crée un compte sur [console.mistral.ai](https://console.mistral.ai),
   génère une clé API (Settings → API Keys → Create new key).
2. Crée un compte sur [tavily.com](https://tavily.com) (tier gratuit,
   1000 requêtes/mois), récupère ta clé API dans le dashboard.
3. À la racine du projet (à côté de `docker-compose.yml`, **pas** dans
   `backend/`), copie `.env.example` en `.env` et colle tes deux clés :
   ```bash
   cp .env.example .env
   # puis édite .env et remplace les placeholders par tes vraies clés
   ```
   Docker Compose lit ce fichier automatiquement et les injecte dans le
   conteneur `api` via `${MISTRAL_API_KEY}` et `${TAVILY_API_KEY}` dans
   `docker-compose.yml`.
4. Relance : `docker-compose up --build`
5. Crée une mission puis lance-la :
   ```bash
   curl -X POST http://localhost:8080/missions/ \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Cherche la météo actuelle à Tunis et résume-la en une phrase"}'

   # récupère l'id retourné, puis :
   curl -X POST http://localhost:8080/missions/<ID>/run
   ```

### Tests

`test_agent_loop.py` teste toute la logique (texte direct, appel d'outil,
limite anti-boucle-infinie) avec un client Mistral **mocké** — aucune clé
API ni connexion réseau nécessaire pour que `pytest` passe.

`test_tools.py` teste l'intégration Tavily (réponse avec résultats, absence
de clé API, aucun résultat trouvé) avec `httpx.AsyncClient.post` mocké —
là aussi, aucun appel réseau réel pendant les tests.

## Étape 4 (à venir) — plus d'outils + orchestration LangGraph

`web_search` (Tavily) est maintenant solide. Prochaine étape : ajouter les
outils SQL, fichier, API externe, chacun dans son propre module testable.
Puis remplacer la boucle manuelle par un graphe LangGraph pour gérer les cas
plus complexes (retries, state persistant via Redis).

## Checklist Git

```bash
git init
git add .
git commit -m "Étape 1: squelette FastAPI + Postgres + Redis via Docker"
git add -A
git commit -m "Étape 2: modèles SQLAlchemy + Alembic + CRUD Mission"
git add -A
git commit -m "Étape 3: premier agent (function calling + boucle manuelle)"
```
