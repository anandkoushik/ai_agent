import logging
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from typing import AsyncGenerator
from agent_system.config import Config

logger = logging.getLogger(__name__)

# Connection pooling enabled by default in SQLAlchemy
engine = create_async_engine(
    Config.DATABASE_URL,
    pool_pre_ping=True,  # Automatic reconnect handling
    pool_size=20,
    max_overflow=10,
    echo=False
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for providing an async database session.
    Automatically handles session lifecycle and rollback on exceptions.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()
