from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    app_env: str = Field(default="local")
    app_name: str = Field(default="ds-research-copilot")
    app_port: int = Field(default=8000)

    database_url: str = Field(default="postgresql+psycopg://postgres:postgres@localhost:5432/postgres")

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None
    cohere_api_key: str | None = None

    embeddings_provider: str = Field(default="sbert")
    rerank_provider: str = Field(default="bge")
    prefer_llm: str = Field(default="anthropic:claude-3-5-sonnet-latest")

    # LLM limits and timeouts
    max_context_tokens: int = Field(default=4000)
    max_output_tokens: int = Field(default=1000)
    llm_timeout_s: int = Field(default=60)

    observability: bool = Field(default=False)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None

    # Metrics and CORS
    metrics_enabled: bool = Field(default=True)
    cors_allowed_origins: list[str] = Field(default_factory=lambda: ["*"])

    redact_pii: bool = Field(default=True)
    store_raw_logs: bool = Field(default=False)

    # Optional API key protection
    api_key: str | None = None
    require_api_key: bool = Field(default=False)

    # Retrieval configuration
    lexical_method: str = Field(default="trgm", description="trgm | tsvector | bm25")

    # Rate limiting
    rate_limit_enabled: bool = Field(default=False)
    rate_limit_per_minute: int = Field(default=60)
    rate_limit_burst: int = Field(default=30)

    # Upload size caps (MB)
    max_upload_mb: int = Field(default=25)

    # Retrieval weighting
    vec_weight: float = Field(default=0.6)
    lex_weight: float = Field(default=0.4)

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }


settings = Settings()

