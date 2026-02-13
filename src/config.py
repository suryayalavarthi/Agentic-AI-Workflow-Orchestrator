"""Centralized configuration via Pydantic Settings.

All environment variables are consolidated here. Other modules should import
``get_settings()`` instead of calling ``os.environ`` directly.
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    anthropic_api_key: str = ""
    google_api_key: str = ""
    default_model: str = "claude-3-haiku-20240307"
    default_temperature: float = 0.0

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_project: str = "agentic-orchestrator"
    langsmith_tracing: bool = True

    # Storage
    sqlite_db_path: str = "./data/app.db"
    chroma_path: str = "./data/chroma"
    chroma_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # MCP
    mcp_fetch_command: Optional[str] = None
    mcp_fetch_args: str = ""

    # Runtime
    log_level: str = "INFO"
    thread_id: str = "default"
    max_loop_count: int = 15
    max_context_messages: int = 6
    recursion_limit: int = 25

    # Retry
    max_retries: int = 3
    retry_base_wait: float = 1.0

    # Web scraper
    web_scraper_timeout: int = 30


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings()
    return _settings


def llm_retry():
    """Retry decorator for LLM invocations (works with both sync and async)."""
    cfg = get_settings()
    return retry(
        stop=stop_after_attempt(cfg.max_retries),
        wait=wait_exponential(multiplier=cfg.retry_base_wait, min=1, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
