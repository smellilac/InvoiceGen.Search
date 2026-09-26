"""FastAPI application entrypoint for the Invoice Search Service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import anyio
from fastapi import Depends, FastAPI, HTTPException, status

from app import __version__
from app.auth import require_user
from app.config import get_settings
from app.database import close_pool, open_pool, ping
from app.observability import verify_langfuse_connection
from app.search import (
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
    hybrid_search,
    index_document,
)

logger = logging.getLogger("invoice_search")


def _configure_logging(level: str) -> None:
    """Attach a handler to the app logger so its logs are actually emitted.

    Without this, the ``invoice_search`` logger has no handler of its own and
    falls back to uvicorn's WARNING-level "last resort" handler — which silently
    drops every ``logger.info`` line (e.g. "Langfuse observability connected").
    Runs on startup regardless of how uvicorn is launched.
    """
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the DB pool on startup, close it on shutdown."""
    settings = get_settings()
    _configure_logging(settings.log_level)
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
async def search(
    request: SearchRequest,
    user_id: str = Depends(require_user),
) -> SearchResponse:
    """Hybrid semantic + filter search over invoices.

    Requires a valid JWT bearer token issued by the InvoiceGen backend; the
    authenticated user id is threaded into the search for optional scoping.
    """
    try:
        return await hybrid_search(request, user_id=user_id)
    except RuntimeError as exc:
        # Raised when the database pool is not open (see app/database.py).
        logger.exception("Search failed: database unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@app.post("/index", response_model=IndexResponse)
async def index(request: IndexRequest) -> IndexResponse:
    """Generate and store the embedding for a single invoice.

    Intentionally **public** (no JWT): this is called server-to-server by the C#
    InvoiceGen backend, which authenticates via a separate mechanism rather than
    the end-user bearer tokens ``/search`` expects.

    Returns ``200`` on success and ``404`` if the document id does not exist in
    the ``documents`` table.
    """
    try:
        updated = await index_document(request)
    except RuntimeError as exc:
        # Raised when the database pool is not open (see app/database.py).
        logger.exception("Index failed: database unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {request.document_id} not found",
        )

    return IndexResponse(document_id=request.document_id)


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
