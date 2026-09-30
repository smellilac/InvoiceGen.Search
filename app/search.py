"""Hybrid search logic: vector similarity combined with SQL filters.

This module defines the request/response contracts and the ``hybrid_search``
entrypoint. The implementation is intentionally left as a stub — this PR only
sets up the project structure.
"""

from __future__ import annotations

import datetime
from datetime import date

from psycopg import sql
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from app import embeddings
from app.database import get_connection


class SearchRequest(BaseModel):
    """Incoming hybrid search request from the C# backend.

    Filters are sent flat at the top level (not nested), matching the JSON the
    C# backend posts, e.g.::

        {"query": "...", "status": "draft", "min_amount": 100,
         "date_from": "2026-01-01", "date_to": "2026-03-31", "limit": 10}
    """

    query: str = Field(..., description="Natural-language search text")
    limit: int = Field(default=10, ge=1, le=100)
    customer_id: str | None = Field(default=None, description="Customer UUID")
    status: str | None = Field(default=None, description="e.g. draft, sent, paid, overdue")
    min_amount: float | None = None
    max_amount: float | None = None
    date_from: date | None = None
    date_to: date | None = None


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


def _build_filter_clauses(
    request: SearchRequest,
) -> tuple[list[sql.Composable], dict[str, object]]:
    """Translate structured filters into parameterized SQL ``WHERE`` fragments.

    Returns the list of conditions (as ``psycopg.sql`` composables) and the
    parameter map to bind. Column names are emitted via ``sql.Identifier`` and
    values stay bound through ``%(name)s`` placeholders — nothing is
    string-concatenated into the query text. Only the filters that are actually
    set contribute a clause, so a request with no filters narrows nothing beyond
    the always-on guards applied by the caller (soft-delete and a non-null
    embedding).
    """
    clauses: list[sql.Composable] = []
    params: dict[str, object] = {}

    if request.customer_id is not None:
        clauses.append(
            sql.SQL("{} = %(customer_id)s::uuid").format(sql.Identifier("CustomerId"))
        )
        params["customer_id"] = request.customer_id
    if request.status is not None:
        # Status is stored as the enum name (e.g. "Paid"); match case-insensitively
        # so callers can pass "paid" as documented on the request model.
        clauses.append(
            sql.SQL("LOWER({}) = LOWER(%(status)s)").format(sql.Identifier("Status"))
        )
        params["status"] = request.status
    if request.min_amount is not None:
        clauses.append(sql.SQL("{} >= %(min_amount)s").format(sql.Identifier("Total")))
        params["min_amount"] = request.min_amount
    if request.max_amount is not None:
        clauses.append(sql.SQL("{} <= %(max_amount)s").format(sql.Identifier("Total")))
        params["max_amount"] = request.max_amount
    if request.date_from is not None:
        clauses.append(sql.SQL("{} >= %(date_from)s").format(sql.Identifier("Date")))
        params["date_from"] = request.date_from
    if request.date_to is not None:
        clauses.append(sql.SQL("{} <= %(date_to)s").format(sql.Identifier("Date")))
        params["date_to"] = request.date_to

    return clauses, params


async def hybrid_search(
    request: SearchRequest, user_id: str | None = None
) -> SearchResponse:
    """Run hybrid vector + filter search over invoices.

    Flow:
      1. Embed ``request.query`` via :func:`app.embeddings.embed_text`.
      2. Build a single SQL query combining pgvector cosine distance
         (``"Embedding" <=> :query_vec``) with ``WHERE`` clauses from
         the request's structured filters, ordered by similarity and capped at
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

    clauses, params = _build_filter_clauses(request)
    params["query_vec"] = vector_literal
    params["limit"] = request.limit

    conditions: list[sql.Composable] = [
        sql.SQL("{} IS NULL").format(sql.Identifier("DeletedAt")),
        sql.SQL("{} IS NOT NULL").format(sql.Identifier("Embedding")),
        *clauses,
    ]
    where_sql = sql.SQL(" AND ").join(conditions)

    query = sql.SQL(
        """
        SELECT
            {id}::text AS id,
            {number} AS number,
            {to} AS to_party,
            {date} AS date,
            {status} AS status,
            {total} AS total,
            {type} AS type,
            1 - ({embedding} <=> %(query_vec)s::vector) AS score
        FROM {table}
        WHERE {where}
        ORDER BY {embedding} <=> %(query_vec)s::vector ASC
        LIMIT %(limit)s
        """
    ).format(
        id=sql.Identifier("Id"),
        number=sql.Identifier("Number"),
        to=sql.Identifier("To"),
        date=sql.Identifier("Date"),
        status=sql.Identifier("Status"),
        total=sql.Identifier("Total"),
        type=sql.Identifier("Type"),
        embedding=sql.Identifier("Embedding"),
        table=sql.Identifier("documents"),
        where=where_sql,
    )

    async with get_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(query, params)
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

    async with get_connection() as conn, conn.cursor() as cur:
        await cur.execute(
            sql, {"embedding": vector_literal, "id": request.document_id}
        )
        updated = cur.rowcount

    return updated > 0
