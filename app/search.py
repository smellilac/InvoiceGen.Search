"""Hybrid search logic: vector similarity combined with SQL filters.

This module defines the request/response contracts and the ``hybrid_search``
entrypoint. The implementation is intentionally left as a stub — this PR only
sets up the project structure.
"""

from __future__ import annotations

import datetime
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
    """A single matched invoice.

    Field names mirror the shape the InvoiceGen frontend consumes directly:
    ``id`` (used to navigate to ``/documents/{id}``) plus the columns rendered
    on the result card. ``score`` is the pgvector cosine similarity, kept for
    ranking/debugging (the frontend simply ignores it).
    """

    id: str
    number: str | None = None
    to: str | None = None
    date: datetime.date | None = None
    status: str | None = None
    total: float | None = None
    type: str | None = None
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class IndexRequest(BaseModel):
    """Request to (re)generate and store the embedding for a single invoice.

    Sent server-to-server by the C# backend whenever an invoice is created or
    its text changes. The text fields mirror the columns this service reads back
    when searching (see ``hybrid_search``): ``"Number"``, ``"To"``, and
    ``"Notes"``.
    """

    document_id: str = Field(..., description="Invoice/document UUID")
    number: str | None = Field(default=None, description='Invoice "Number"')
    to: str | None = Field(default=None, description='Invoice "To" party')
    notes: str | None = Field(default=None, description='Invoice "Notes"')


class IndexResponse(BaseModel):
    document_id: str
    indexed: bool = True


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
            "Id"::text AS id,
            "Number" AS number,
            "To" AS to_party,
            "Date" AS date,
            "Status" AS status,
            "Total" AS total,
            "Type" AS type,
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
            id=row["id"],
            number=row["number"],
            to=row["to_party"],
            date=row["date"],
            status=row["status"],
            total=float(row["total"]) if row["total"] is not None else None,
            type=row["type"],
            score=float(row["score"]),
        )
        for row in rows
    ]
    return SearchResponse(results=results)


def _build_index_text(request: IndexRequest) -> str:
    """Concatenate the invoice's text fields into a single string to embed.

    Only the fields that are actually present contribute, joined by newlines so
    the embedding model sees them as distinct lines rather than one run-on token.
    """
    parts = [p for p in (request.number, request.to, request.notes) if p]
    return "\n".join(parts)


async def index_document(request: IndexRequest) -> bool:
    """Embed an invoice's text fields and store the vector on its ``documents`` row.

    Embeds the concatenated text via :func:`app.embeddings.embed_text`, then runs
    a single ``UPDATE ... SET "Embedding" = ...`` keyed on the document id.

    Returns ``True`` if a row was updated, or ``False`` if no row matched the id
    (the caller maps this to ``404``).
    """
    text = _build_index_text(request)
    embedding = await embeddings.embed_text(text)
    # Same fully-parameterized vector literal used on the search path: bind the
    # value and cast to ::vector so there is no SQL-injection surface.
    vector_literal = "[" + ",".join(str(value) for value in embedding) + "]"

    sql = 'UPDATE documents SET "Embedding" = %(embedding)s::vector WHERE "Id" = %(id)s::uuid'

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                sql, {"embedding": vector_literal, "id": request.document_id}
            )
            updated = cur.rowcount

    return updated > 0
