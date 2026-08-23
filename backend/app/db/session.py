from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Classe de base pour tous les modèles SQLAlchemy (Mission, AgentRun, ToolCall, ...)."""
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency FastAPI: fournit une session DB par requête, la ferme après."""
    async with AsyncSessionLocal() as session:
        yield session
