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
