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

    # Recherche web (Tavily — tier gratuit 1000 requêtes/mois, conçu pour les
    # agents LLM plutôt que pour un moteur de recherche humain classique)
    TAVILY_API_KEY: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
