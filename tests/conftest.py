from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from src.config import Settings


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    """Reset the settings singleton before each test."""
    monkeypatch.setattr("src.config._settings", None)


@pytest.fixture()
def mock_settings(monkeypatch):
    """Provides a Settings instance with test defaults (no .env loading)."""
    settings = Settings(
        anthropic_api_key="test-key",
        chroma_path="/tmp/test_chroma",
        default_model="claude-3-haiku-20240307",
        default_temperature=0.0,
        log_level="DEBUG",
        max_loop_count=15,
        max_context_messages=6,
        max_retries=1,
    )
    monkeypatch.setattr("src.config._settings", settings)
    return settings


@pytest.fixture()
def sample_state():
    return {
        "messages": [HumanMessage(content="Compare SQLite vs PostgreSQL")],
        "summary": "",
        "research_results": [],
        "needs_more_research": True,
        "loop_count": 0,
    }


@pytest.fixture()
def mock_llm():
    """Returns a MagicMock that simulates ChatAnthropic."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content='{"next_agent":"researcher","reasoning":"test"}'
    )
    llm.with_structured_output.return_value = llm
    llm.bind_tools.return_value = llm
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="test response"))
    return llm


@pytest.fixture()
def mock_vector_db():
    db = MagicMock()
    db.store_research.return_value = "doc-id"
    db.retrieve_knowledge.return_value = ["chunk1"]
    db.retrieve_knowledge_with_sources.return_value = [
        {"text": "chunk1", "source_url": "https://example.com"}
    ]
    return db
