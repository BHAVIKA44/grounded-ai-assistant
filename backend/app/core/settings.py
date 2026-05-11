"""
Application settings loaded from environment variables.
"""

from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Grounded AI Assistant", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # Database
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="rag_user", alias="POSTGRES_USER")
    postgres_password: str = Field(default="rag_password", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="rag_db", alias="POSTGRES_DB")

    # Redis
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")
    redis_db: int = Field(default=0, alias="REDIS_DB")

    # Vector Store
    chroma_persist_dir: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIR")

    # Embeddings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")

    # LLM - OpenAI
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", alias="OPENAI_MODEL")

    # LLM - Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="llama2", alias="OLLAMA_MODEL")

    # Reranking
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANK_MODEL"
    )
    rerank_top_k: int = Field(default=5, alias="RERANK_TOP_K")

    # Retrieval
    bm25_top_k: int = Field(default=10, alias="BM25_TOP_K")
    vector_top_k: int = Field(default=10, alias="VECTOR_TOP_K")
    hybrid_top_k: int = Field(default=10, alias="HYBRID_TOP_K")

    # Observability
    langsmith_api_key: Optional[str] = Field(
        default=None, alias="LANGSMITH_API_KEY"
    )
    langsmith_project: str = Field(
        default="grounded-ai-assistant", alias="LANGSMITH_PROJECT"
    )
    langsmith_tracing: bool = Field(default=True, alias="LANGSMITH_TRACING")

    # Fine-tuning
    fine_tune_model_base: str = Field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", alias="FINE_TUNE_MODEL_BASE"
    )
    fine_tune_output_dir: str = Field(
        default="./models/lora", alias="FINE_TUNE_OUTPUT_DIR"
    )
    fine_tune_rank: int = Field(default=8, alias="FINE_TUNE_RANK")
    fine_tune_alpha: int = Field(default=16, alias="FINE_TUNE_ALPHA")
    fine_tune_dropout: float = Field(default=0.05, alias="FINE_TUNE_DROPOUT")

    # Security
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")
    allowed_extensions: List[str] = Field(
        default=["pdf", "txt", "docx"], alias="ALLOWED_EXTENSIONS"
    )
    rate_limit_per_minute: int = Field(default=60, alias="RATE_LIMIT_PER_MINUTE")

    @property
    def database_url(self) -> str:
        """Build async database URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """Build sync database URL for Alembic."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Build Redis URL."""
        if self.redis_password:
            return (
                f"redis://:{self.redis_password}@{self.redis_host}:"
                f"{self.redis_port}/{self.redis_db}"
            )
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
