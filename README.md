# invoice-search-service

Semantic search sidecar for the InvoiceGen application. FastAPI + pgvector
(Neon Postgres) + OpenAI embeddings.

See [CLAUDE.md](./CLAUDE.md) for architecture, the C# backend integration, and
project layout.

## Quick start

```bash
uv sync
cp .env.example .env   # fill in DATABASE_URL and OPENAI_API_KEY
uv run uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs.

> Scaffold only: `/search` returns `501` until search logic is implemented.
