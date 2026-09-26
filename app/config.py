"""Application configuration loaded from environment variables / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Neon Postgres
    database_url: str = ""

    # OpenRouter (LLM + embeddings)
    openrouter_api_key: str = ""
    llm_model: str = "qwen/qwen3-8b:free"
    llm_base_url: str = "https://openrouter.ai/api/v1"

    # Embeddings (Google Gemini via the OpenAI-compatible endpoint).
    # gemini-embedding-001 defaults to 3072 dimensions; embedding_dimensions is
    # sent as the `dimensions` parameter to request a 1024-vector that matches
    # the pgvector column in the database.
    embedding_model: str = "gemini-embedding-001"
    embedding_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    embedding_dimensions: int = 1024

    # Langfuse observability (optional — disabled when keys are empty)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # JWT authentication. The same HMAC-SHA256 (HS256) tokens the C# InvoiceGen
    # backend issues to its users; this service validates signature, expiry,
    # issuer, and audience. jwt_secret_key is the shared symmetric key.
    jwt_secret_key: str = ""
    jwt_issuer: str = ""
    jwt_audience: str = ""

    # Service
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "info"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
