"""Hybrid search logic: vector similarity combined with SQL filters.

This module defines the request/response contracts and the ``hybrid_search``
entrypoint. The implementation is intentionally left as a stub — this PR only
sets up the project structure.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    """Structured SQL filters applied alongside vector similarity."""

    customer_id: int | None = None
    status: str | None = Field(default=None, description="e.g. draft, sent, paid, overdue")
    min_amount: float | None = None
    max_amount: float | None = None
    issued_after: date | None = None
    issued_before: date | None = None


class SearchRequest(BaseModel):
    """Incoming hybrid search request from the C# backend."""

    query: str = Field(..., description="Natural-language search text")
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(default=10, ge=1, le=100)


class SearchResult(BaseModel):
    """A single matched invoice."""

    invoice_id: int
    score: float
    snippet: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


async def hybrid_search(request: SearchRequest) -> SearchResponse:
    """Run hybrid vector + filter search.

    Not implemented yet. The eventual flow:
      1. Embed ``request.query`` via ``embeddings.embed_text``.
      2. Build a SQL query combining ``ORDER BY embedding <=> :query_vec``
         (pgvector cosine distance) with WHERE clauses from ``request.filters``.
      3. Execute against the Neon pool and map rows to ``SearchResult``.
    """
    # Langfuse observability: when step 1 (or any future re-ranking / query
    # rewriting) is driven through LangChain, pass the shared handler on the call
    # so the whole search is captured as one trace, e.g.:
    #
    #     from app.observability import get_langfuse_handler
    #     handler = get_langfuse_handler()
    #     callbacks = [handler] if handler else []
    #     result = await chain.ainvoke(request.query, config={"callbacks": callbacks})
    #
    # For direct OpenAI embedding calls, use the langfuse.openai wrapper instead
    # (see app/embeddings.py) — no handler needed there.
    raise NotImplementedError("Hybrid search is not implemented yet")
