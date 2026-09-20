"""OpenAI embeddings generation.

Wraps the OpenAI embeddings API so the rest of the service can turn text into
vectors without depending on the SDK directly. The actual embedding calls are
stubbed for now — the client is wired up but ``embed_text`` is not yet used by
any live search path.
"""

from __future__ import annotations

from functools import lru_cache

from openai import AsyncOpenAI

from app.config import get_settings

# ── Langfuse observability ────────────────────────────────────────────────────
# This service calls OpenAI directly, so the simplest way to trace embedding
# calls is Langfuse's drop-in wrapper: change the import above to
#     from langfuse.openai import AsyncOpenAI
# and nothing else — every request through `get_client()` is then traced.
#
# If an embedding call is ever routed through LangChain instead (e.g.
# `langchain_openai.OpenAIEmbeddings`), attach the shared handler on the call:
#     from app.observability import get_langfuse_handler
#     handler = get_langfuse_handler()
#     callbacks = [handler] if handler else []
#     await embeddings.aembed_query(text, config={"callbacks": callbacks})
# ──────────────────────────────────────────────────────────────────────────────


@lru_cache
def get_client() -> AsyncOpenAI:
    """Return a cached AsyncOpenAI client pointed at the embeddings provider.

    Embeddings are served through OpenRouter (see ``app/config.py``), which is
    OpenAI-API compatible, so the OpenAI SDK talks to it directly given the
    matching ``base_url`` and key.
    """
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.embedding_base_url,
    )


def _validate_dimensions(vector: list[float]) -> list[float]:
    """Ensure an embedding matches the configured (and DB column) dimensionality.

    A mismatch here means pgvector's ``<=>`` operator would raise at query time,
    so failing fast with a clear message is friendlier than a cryptic SQL error.
    """
    expected = get_settings().embedding_dimensions
    if len(vector) != expected:
        raise ValueError(
            f"Embedding has {len(vector)} dimensions, expected {expected}. "
            "Check EMBEDDING_MODEL / EMBEDDING_DIMENSIONS against the DB column."
        )
    return vector


async def embed_text(text: str) -> list[float]:
    """Generate an embedding vector for a single piece of text."""
    settings = get_settings()
    response = await get_client().embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    return _validate_dimensions(response.data[0].embedding)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts, preserving input order."""
    if not texts:
        return []
    settings = get_settings()
    response = await get_client().embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    return [_validate_dimensions(item.embedding) for item in ordered]
