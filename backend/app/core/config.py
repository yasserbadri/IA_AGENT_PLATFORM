from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration de l'application, lue depuis les variables d'environnement.
    En local, ces valeurs viennent du fichier .env (voir .env.example).
    En Docker, elles viennent de docker-compose.yml.
    """

    APP_NAME: str = "AI Agent Operations Platform"
    ENV: str = "development"

    # Postgres
    DATABASE_URL: str = "postgresql+asyncpg://agent_user:agent_pass@db:5432/agent_platform"

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    # LLM (Mistral — tier gratuit "Experiment" sur console.mistral.ai)
    MISTRAL_API_KEY: str = ""
    MISTRAL_MODEL: str = "mistral-small-latest"  # le moins cher, largement suffisant pour dev/tests

    # OpenAI (platform.openai.com). Vérifie platform.openai.com/docs/models
    # pour le modèle le plus récent/économique — ça change vite.
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # xAI Grok (console.x.ai). API 100% compatible OpenAI (même SDK, juste
    # base_url + clé différents) — voir docs.x.ai.
    XAI_API_KEY: str = ""
    XAI_MODEL: str = "grok-4-0709"

    # Google Gemini (aistudio.google.com), via la couche de compatibilité
    # OpenAI officielle de Google (mêmes SDK/format que ci-dessus, endpoint
    # différent) — voir ai.google.dev/gemini-api/docs/openai. Vérifie le nom
    # de modèle courant sur ai.google.dev/gemini-api/docs/models.
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.7-flash"

    # Provider utilisé par défaut si une mission n'en précise pas.
    DEFAULT_LLM_PROVIDER: str = "mistral"

    # Recherche web (Tavily — tier gratuit 1000 requêtes/mois, conçu pour les
    # agents LLM plutôt que pour un moteur de recherche humain classique)
    TAVILY_API_KEY: str = ""

    # Répertoire sandboxé où l'outil read_file va chercher ses fichiers.
    # Chemin relatif : résolu depuis le dossier de travail du process
    # (WORKDIR /app dans le conteneur Docker -> backend/data/ sur ta machine).
    FILES_DIR: str = "data"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
