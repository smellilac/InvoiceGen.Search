"""Connection management for the Neon Postgres (pgvector) database.

A single async connection pool is created at application startup and closed at
shutdown. Nothing here issues queries yet — search logic lives in ``search.py``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings

_pool: AsyncConnectionPool | None = None


async def open_pool() -> None:
    """Create the connection pool. Called on application startup."""
    global _pool
    if _pool is not None:
        return

    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    _pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=10,
        open=False,
    )
    await _pool.open()


async def close_pool() -> None:
    """Close the connection pool. Called on application shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> AsyncConnectionPool:
    """Return the active pool, or raise if it has not been opened."""
    if _pool is None:
        raise RuntimeError("Connection pool is not open")
    return _pool


@asynccontextmanager
async def get_connection() -> AsyncGenerator[AsyncConnection]:
    """Acquire a connection from the pool as an async context manager."""
    async with get_pool().connection() as conn:
        yield conn


async def ping() -> bool:
    """Run a trivial query to confirm the database is reachable."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
            await cur.fetchone()
    return True
