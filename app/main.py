"""FastAPI application entrypoint for the Invoice Search Service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException, status

from app import __version__
from app.config import get_settings
from app.database import close_pool, open_pool, ping
from app.observability import verify_langfuse_connection
from app.search import SearchRequest, SearchResponse, hybrid_search

logger = logging.getLogger("invoice_search")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the DB pool on startup, close it on shutdown."""
    settings = get_settings()
    if settings.database_url:
        try:
            await open_pool()
            logger.info("Database pool opened")
        except Exception:  # noqa: BLE001 - startup should not hard-fail without DB configured
            logger.exception("Failed to open database pool")
    else:
        logger.warning("DATABASE_URL not set; starting without a database connection")

    # Verify Langfuse observability is reachable. Non-fatal: the service runs
    # fine without it, so a failure here only downgrades to a warning.
    # auth_check() is blocking, so run it off the event loop.
    try:
        if await anyio.to_thread.run_sync(verify_langfuse_connection):
            logger.info("Langfuse observability connected")
        else:
            logger.warning("Langfuse not configured; observability disabled")
    except Exception:  # noqa: BLE001 - observability must never block startup
        logger.warning(
            "Langfuse connection check failed; continuing without observability",
            exc_info=True,
        )

    yield
    await close_pool()
    logger.info("Database pool closed")


app = FastAPI(
    title="Invoice Search Service",
    description="Semantic search for InvoiceGen using pgvector and OpenAI embeddings.",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "version": __version__}


@app.get("/health/db")
async def health_db() -> dict[str, str]:
    """Readiness probe that verifies the database is reachable."""
    try:
        await ping()
        return {"status": "ok", "database": "reachable"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "degraded", "database": f"unreachable: {exc}"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Hybrid semantic + filter search over invoices.

    Not implemented yet — returns 501 until search logic lands.
    """
    try:
        return await hybrid_search(request)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc


def main() -> None:
    """Run the service with uvicorn (used by `python -m app.main`)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.log_level,
        reload=True,
    )


if __name__ == "__main__":
    main()
