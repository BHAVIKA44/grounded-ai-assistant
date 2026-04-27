"""
Database connection and session management.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.database import Base

logger = get_logger(__name__)

# Global engine and session maker
_engine = None
_session_maker = None


async def init_db() -> None:
    """Initialize database connection and create tables."""
    global _engine, _session_maker
    settings = get_settings()

    logger.info("initializing_database", database_url=settings.database_url)

    _engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        poolclass=NullPool,  # Use NullPool for async
        pool_pre_ping=True,
    )

    _session_maker = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create tables
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("database_initialized_successfully")


async def close_db() -> None:
    """Close database connection."""
    global _engine
    if _engine:
        await _engine.dispose()
        logger.info("database_connection_closed")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session with automatic cleanup.
    
    Yields:
        AsyncSession: Database session
        
    Example:
        async with get_session() as session:
            result = await session.execute(query)
    """
    if _session_maker is None:
        await init_db()

    async with _session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.
    
    Yields:
        AsyncSession: Database session
    """
    async with get_session() as session:
        yield session