"""Hybrid search logic: vector similarity combined with SQL filters.

This module defines the request/response contracts and the ``hybrid_search``
entrypoint. The implementation is intentionally left as a stub — this PR only
sets up the project structure.
"""

from __future__ import annotations

from datetime import date

from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from app import embeddings
from app.database import get_connection


class SearchFilters(BaseModel):
    """Structured SQL filters applied alongside vector similarity."""

    customer_id: str | None = Field(default=None, description="Customer UUID")
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

    invoice_id: str
    score: float
    snippet: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


def _build_filter_clauses(filters: SearchFilters) -> tuple[list[str], dict[str, object]]:
    """Translate structured filters into parameterized SQL ``WHERE`` fragments.

    Returns the list of conditions and the parameter map to bind. Only the
    filters that are actually set contribute a clause, so an empty
    ``SearchFilters`` narrows nothing beyond the always-on guards applied by the
    caller (soft-delete and a non-null embedding).
    """
    clauses: list[str] = []
    params: dict[str, object] = {}

    if filters.customer_id is not None:
        clauses.append('"CustomerId" = %(customer_id)s::uuid')
        params["customer_id"] = filters.customer_id
    if filters.status is not None:
        # Status is stored as the enum name (e.g. "Paid"); match case-insensitively
        # so callers can pass "paid" as documented on the request model.
        clauses.append('LOWER("Status") = LOWER(%(status)s)')
        params["status"] = filters.status
    if filters.min_amount is not None:
        clauses.append('"Total" >= %(min_amount)s')
        params["min_amount"] = filters.min_amount
    if filters.max_amount is not None:
        clauses.append('"Total" <= %(max_amount)s')
        params["max_amount"] = filters.max_amount
    if filters.issued_after is not None:
        clauses.append('"Date" >= %(issued_after)s')
        params["issued_after"] = filters.issued_after
    if filters.issued_before is not None:
        clauses.append('"Date" <= %(issued_before)s')
        params["issued_before"] = filters.issued_before

    return clauses, params


def _format_snippet(number: str | None, to: str | None, notes: str | None) -> str | None:
    """Compose a short human-readable summary of a matched invoice."""
    parts = [p for p in (number, to, notes) if p]
    if not parts:
        return None
    snippet = " — ".join(parts)
    return snippet if len(snippet) <= 200 else snippet[:197] + "..."


async def hybrid_search(
    request: SearchRequest, user_id: str | None = None
) -> SearchResponse:
    """Run hybrid vector + filter search over invoices.

    Flow:
      1. Embed ``request.query`` via :func:`app.embeddings.embed_text`.
      2. Build a single SQL query combining pgvector cosine distance
         (``"Embedding" <=> :query_vec``) with ``WHERE`` clauses from
         ``request.filters``, ordered by similarity and capped at
         ``request.limit``.
      3. Execute against the Neon pool and map rows to :class:`SearchResult`.

    ``user_id`` is the authenticated caller extracted from the JWT ``sub`` claim.
    It is threaded through for future per-user scoping (e.g. restricting results
    to a user's own invoices) but is not yet applied to the SQL.
    """
    # Langfuse observability: embedding calls made through app.embeddings are
    # traced automatically when the langfuse.openai wrapper is enabled there — no
    # handler is needed here. If a future re-ranking / query-rewriting step is
    # driven through LangChain, pass get_langfuse_handler() via
    # config={"callbacks": [...]} so the whole search is captured as one trace.
    query_vector = await embeddings.embed_text(request.query)
    # pgvector accepts a vector literal like "[0.1,0.2,...]"; casting the bound
    # parameter to ::vector keeps the value fully parameterized (no SQL injection
    # surface) without depending on per-connection type registration.
    vector_literal = "[" + ",".join(str(value) for value in query_vector) + "]"

    clauses, params = _build_filter_clauses(request.filters)
    params["query_vec"] = vector_literal
    params["limit"] = request.limit

    conditions = ['"DeletedAt" IS NULL', '"Embedding" IS NOT NULL', *clauses]
    where_sql = " AND ".join(conditions)

    sql = f"""
        SELECT
            "Id"::text AS invoice_id,
            "Number" AS number,
            "To" AS to_party,
            "Notes" AS notes,
            1 - ("Embedding" <=> %(query_vec)s::vector) AS score
        FROM documents
        WHERE {where_sql}
        ORDER BY "Embedding" <=> %(query_vec)s::vector ASC
        LIMIT %(limit)s
    """

    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()

    results = [
        SearchResult(
            invoice_id=row["invoice_id"],
            score=float(row["score"]),
            snippet=_format_snippet(row["number"], row["to_party"], row["notes"]),
        )
        for row in rows
    ]
    return SearchResponse(results=results)
