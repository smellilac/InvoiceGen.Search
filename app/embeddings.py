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
    """Return a cached AsyncOpenAI client."""
    settings = get_settings()
    return AsyncOpenAI(api_key=settings.openai_api_key)


async def embed_text(text: str) -> list[float]:
    """Generate an embedding vector for a single piece of text.

    Not implemented yet — this is the seam where the OpenAI embeddings call
    will live once search logic is built out.
    """
    raise NotImplementedError("Embedding generation is not implemented yet")


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts.

    Not implemented yet.
    """
    raise NotImplementedError("Batch embedding generation is not implemented yet")
