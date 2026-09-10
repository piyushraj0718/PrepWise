from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env."""

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/prepwise"
    uploads_dir: Path = Path("data/uploads")
    log_level: str = "INFO"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    gemini_api_key: str | None = None
    gemini_model_name: str = "gemini-2.5-flash"
    gemini_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    llm_retry_base_delay_seconds: float = 0.5
    assessment_semantic_evaluator_enabled: bool = False
    assessment_semantic_evaluator_model_name: str | None = None
    assessment_semantic_evaluator_timeout_seconds: float = 30.0
    assessment_semantic_pass_threshold: float = 0.75

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
