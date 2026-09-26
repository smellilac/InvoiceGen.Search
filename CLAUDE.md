# invoice-search-service

A Python **FastAPI** microservice that provides **semantic search** for the
**InvoiceGen** invoice management application. It embeds natural-language queries
with **Google Gemini embeddings** and runs **hybrid search** — pgvector similarity
combined with structured SQL filters — against the existing InvoiceGen Postgres
database hosted on **Neon**.

> Status: hybrid search is implemented. The FastAPI app starts, exposes health
> checks, and `POST /search` runs a real pgvector + SQL-filter query against the
> InvoiceGen `documents` table. `POST /search` is protected by JWT bearer
> authentication using the same HS256 tokens the C# backend issues to its users.

## Development workflow (MANDATORY)

Follow these steps for **every** new feature or change. **Never skip steps.
Never commit directly to `master`.**

### When to trigger this workflow

**Trigger the workflow for:**

- Any change to tracked files in the repository (`.gitignore`, config
  files, source code, documentation, `requirements.txt`, etc.).
- Always — even for small, one-line changes.

**Do NOT trigger the workflow for:**

- Creating local-only files that are gitignored (`.env`, local configs,
  temp files).
- Read-only operations (searching, reading files, running tests without
  code changes).

Follow the workflow for everything else without exception.

1. **Git setup**
   - Fetch latest from origin: `git fetch origin`
   - Ensure `master` is up to date: `git checkout master && git pull origin master`
   - Create a feature branch: `git checkout -b feature/<short-description>`

2. **Planning**
   - Update `CLAUDE.md` or relevant docs to reflect what is about to change.
   - State clearly what will be implemented before writing any code.

3. **Implementation**
   - Write the code.
   - Follow existing conventions in the codebase.

4. **Verification**
   - Run available tests.
   - Verify the app starts without errors.
   - Check imports are clean.

5. **Commit & push & PR**
   - Stage relevant files only (**never commit `.env`**).
   - Write a descriptive commit message following conventional commits.
   - Push the branch to origin.
   - Create a Pull Request to `master` using the `gh` CLI:

     ```bash
     gh pr create --title "<short description>" --body "<what changed and why>" --base master
     ```

   - Report the PR link and **STOP**.
   - Wait for explicit confirmation before doing anything else.
   - Only after the user says "merge" or "approve" — run:

     ```bash
     gh pr merge <pr-number> --merge --delete-branch
     ```

   - **Never merge without explicit user confirmation.**

   > **IMPORTANT:** Creating the PR is the end of the development cycle.
   > The next task starts only after the PR is merged and the user gives a
   > new instruction.

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
- **Auth / networking.** `POST /search` requires a JWT bearer token — the same
  HS256 token the C# backend issues to its users. The service validates the
  token's signature, expiry, issuer, and audience (see `app/auth.py`) and passes
  the `sub` claim (user id) into the search. The health endpoints and the
  server-to-server `POST /index` endpoint stay public.

## Project layout

| File                 | Responsibility                                             |
| -------------------- | ---------------------------------------------------------- |
| `app/main.py`        | FastAPI app, lifespan (pool open/close), routes            |
| `app/config.py`      | Settings loaded from environment / `.env`                  |
| `app/database.py`    | Neon Postgres async connection pool (psycopg + pgvector)   |
| `app/embeddings.py`  | OpenAI embeddings client and helpers (stubbed)             |
| `app/search.py`      | Search request/response models and `hybrid_search` (stub)  |
| `app/auth.py`        | JWT bearer validation dependency (`require_user`)          |

## Local development

Dependencies are managed with **uv**. A `requirements.txt` is also provided for
plain-pip environments.

```bash
# Install dependencies into a local .venv
uv sync

# Configure environment
cp .env.example .env   # then fill in DATABASE_URL and OPENROUTER_API_KEY

# Run the service (http://localhost:8000, docs at /docs)
uv run uvicorn app.main:app --reload
```

### Endpoints

- `GET /health` — liveness (no DB required).
- `GET /health/db` — readiness; verifies the Neon database is reachable.
- `POST /search` — hybrid semantic + filter search over the `documents` table.
- `POST /index` — (re)generate and store the embedding for a single invoice.
  Public (no JWT): called server-to-server by the C# backend. Returns `200` on
  success, `404` if the document id is not found.

## Configuration

See `.env.example`. Key variables:

- `DATABASE_URL` — Neon Postgres connection string (`sslmode=require`).
- `OPENROUTER_API_KEY` — API key used by the embeddings client. Embeddings now
  run on Google Gemini's OpenAI-compatible endpoint, so this must be a **Gemini
  API key**.
- `EMBEDDING_MODEL` / `EMBEDDING_BASE_URL` / `EMBEDDING_DIMENSIONS` — Gemini
  embedding model (`gemini-embedding-001`), its OpenAI-compatible base URL, and
  the output dimensionality. `EMBEDDING_DIMENSIONS` is sent as the `dimensions`
  parameter (Gemini defaults to 3072) and must match the `Embedding` `vector`
  column in the database (currently `vector(1024)`).
- `JWT_SECRET_KEY` / `JWT_ISSUER` / `JWT_AUDIENCE` — shared HS256 signing key
  and the expected issuer/audience for validating the C# backend's JWTs. Must
  match the backend's token configuration.

## Search behaviour

`POST /search` embeds the query, then runs a single query that orders `documents`
by pgvector cosine distance (`<=>`) and applies the structured filters as SQL
`WHERE` clauses. Results are ranked by cosine similarity (`score = 1 - distance`)
and capped at `limit`. Soft-deleted rows (`DeletedAt IS NOT NULL`) and rows with
no embedding are excluded. Invoice and customer IDs are UUID strings, matching
the InvoiceGen schema this service reads.

Each result is shaped to match what the InvoiceGen frontend renders directly:
`id` (used for navigating to `/documents/{id}`), `number`, `to`, `date`,
`status`, `total`, and `type`, plus a `score` (cosine similarity) the frontend
ignores. These map straight onto the `documents` columns of the same name (`Id`,
`Number`, `To`, `Date`, `Status`, `Total`, `Type`).

## Authentication

`POST /search` is protected by the `require_user` dependency in `app/auth.py`.
Callers must send `Authorization: Bearer <jwt>`. The token is the HS256 JWT the
C# InvoiceGen backend issues to its users; the service verifies its signature
(`JWT_SECRET_KEY`), expiry, issuer (`JWT_ISSUER`), and audience (`JWT_AUDIENCE`),
then extracts the user id from the `sub` claim. A missing, malformed, expired, or
otherwise invalid token returns `401 Unauthorized`. The health endpoints
(`GET /health`, `GET /health/db`) and `POST /index` remain public — the latter is
a server-to-server call from the C# backend, which uses its own auth mechanism. The validated user id is
threaded into `hybrid_search` for future per-user result scoping (not yet applied
to the SQL).

## Not implemented yet

- Per-user filtering of search results (the `sub` user id is threaded through
  `hybrid_search` but not yet used in the SQL).
