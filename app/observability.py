"""Langfuse observability wiring.

Centralizes Langfuse initialization so the rest of the service can emit traces
without importing the SDK directly. Two integration styles are supported, and
both flow through the single client initialized here:

- **Direct OpenAI SDK** (what this service uses today): swap the plain client
  for Langfuse's drop-in wrapper — ``from langfuse.openai import AsyncOpenAI`` —
  and every embeddings/completions call is traced automatically. See
  ``app/embeddings.py``.
- **LangChain**: pass ``get_langfuse_handler()`` as a callback via
  ``config={"callbacks": [handler]}`` on any runnable. See ``app/search.py`` and
  ``app/embeddings.py`` for the exact call sites.

Note: the official LangChain integration ships *inside* the ``langfuse`` package
(``langfuse.langchain.CallbackHandler``). There is no separate, Langfuse-published
``langfuse-langchain`` distribution — do not add one.

Langfuse is optional. When credentials are absent the helpers degrade to
``None`` so local development and tests don't require a Langfuse project.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import get_settings

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

logger = logging.getLogger("invoice_search")


@lru_cache
def get_langfuse() -> "Langfuse | None":
    """Return a cached Langfuse client, or ``None`` if credentials aren't set.

    Constructing the client registers it as the process-global Langfuse instance,
    which both ``get_langfuse_handler`` and Langfuse's OpenAI wrapper reuse.
    """
    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        logger.info("Langfuse credentials not set; observability disabled")
        return None

    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def get_langfuse_handler() -> "CallbackHandler | None":
    """Return a Langfuse LangChain ``CallbackHandler``, or ``None`` if disabled.

    Pass the result to any LangChain call as a callback::

        from app.observability import get_langfuse_handler

        handler = get_langfuse_handler()
        callbacks = [handler] if handler else []
        await chain.ainvoke(payload, config={"callbacks": callbacks})

    The handler reuses the global client created by :func:`get_langfuse`, so
    credentials/host come from there.
    """
    if get_langfuse() is None:
        return None

    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def verify_langfuse_connection() -> bool:
    """Validate Langfuse credentials against the server.

    Returns ``True`` when configured *and* the credentials authenticate. Returns
    ``False`` when Langfuse isn't configured. Raises if configured credentials
    are rejected by the server (so callers can log the reason). This call is
    blocking — run it off the event loop in async contexts.
    """
    client = get_langfuse()
    if client is None:
        return False
    return bool(client.auth_check())
