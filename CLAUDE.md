# invoice-search-service

A Python **FastAPI** microservice that provides **semantic search** for the
**InvoiceGen** invoice management application. It embeds natural-language queries
with **OpenAI embeddings** and runs **hybrid search** — pgvector similarity
combined with structured SQL filters — against the existing InvoiceGen Postgres
database hosted on **Neon**.

> Status: project scaffold only. The FastAPI app starts and exposes health
> checks, but search logic is not implemented yet (`/search` returns `501`).

## What this service does

- Accepts a search request (natural-language `query` + optional structured
  `filters`) from the C# backend.
- Turns the query into a vector using an OpenAI embedding model.
- Runs a hybrid query: pgvector cosine similarity (`<=>`) over invoice
  embeddings, narrowed by SQL `WHERE` filters (customer, status, amount, date).
- Returns ranked invoice matches with similarity scores.

## How it connects to the C# backend (InvoiceGen)

InvoiceGen's primary application is a C# backend. This service is a **sidecar
search API** that the C# backend calls over HTTP:

```
InvoiceGen (C#)  ──HTTP──▶  invoice-search-service (FastAPI)  ──▶  Neon Postgres (pgvector)
                                        │
                                        └──▶  OpenAI Embeddings API
```

- **Shared database.** Both the C# backend and this service point at the *same*
  Neon Postgres database. The C# backend owns the schema and writes invoice
  data; this service reads invoices and their `vector` embedding column. It does
  not run migrations.
- **Integration contract.** The C# backend sends `POST /search` with a JSON body
  matching `SearchRequest` (see `app/search.py`) and receives `SearchResponse`.
  OpenAPI docs are available at `/docs` when the service is running.
- **Auth / networking.** Intended to run inside the same private network /
  service mesh as the backend. Add auth (e.g. an API key or mTLS) before exposing
  it publicly — not yet implemented.

## Project layout

| File                 | Responsibility                                             |
| -------------------- | ---------------------------------------------------------- |
| `app/main.py`        | FastAPI app, lifespan (pool open/close), routes            |
| `app/config.py`      | Settings loaded from environment / `.env`                  |
| `app/database.py`    | Neon Postgres async connection pool (psycopg + pgvector)   |
| `app/embeddings.py`  | OpenAI embeddings client and helpers (stubbed)             |
| `app/search.py`      | Search request/response models and `hybrid_search` (stub)  |

## Local development

Dependencies are managed with **uv**. A `requirements.txt` is also provided for
plain-pip environments.

```bash
# Install dependencies into a local .venv
uv sync

# Configure environment
cp .env.example .env   # then fill in DATABASE_URL and OPENAI_API_KEY

# Run the service (http://localhost:8000, docs at /docs)
uv run uvicorn app.main:app --reload
```

### Endpoints

- `GET /health` — liveness (no DB required).
- `GET /health/db` — readiness; verifies the Neon database is reachable.
- `POST /search` — hybrid search (currently `501 Not Implemented`).

## Configuration

See `.env.example`. Key variables:

- `DATABASE_URL` — Neon Postgres connection string (`sslmode=require`).
- `OPENAI_API_KEY` — OpenAI API key.
- `OPENAI_EMBEDDING_MODEL` / `OPENAI_EMBEDDING_DIMENSIONS` — must match the
  dimensionality of the `vector` column in the database.

## Not implemented yet

- Actual embedding generation (`app/embeddings.py`).
- Hybrid search query construction and execution (`app/search.py`).
- Authentication between the C# backend and this service.
