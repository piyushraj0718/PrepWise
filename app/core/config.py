from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env."""

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/prepwise"
    uploads_dir: Path = Path("data/uploads")
    log_level: str = "INFO"
    embedding_model_name: str = "all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
