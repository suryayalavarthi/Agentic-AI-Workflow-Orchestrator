from __future__ import annotations

import pytest
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

from src.tools.registry import (
    _registry,
    get_all_tools,
    get_tools,
    list_categories,
    register_tools,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear the registry before and after each test."""
    saved = dict(_registry)
    _registry.clear()
    yield
    _registry.clear()
    _registry.update(saved)


def _make_provider(tools):
    @asynccontextmanager
    async def provider():
        yield tools
    return provider


@pytest.mark.asyncio
async def test_register_and_list(mock_settings):
    mock_tool = MagicMock()
    register_tools("test_cat", _make_provider([mock_tool]))
    assert "test_cat" in list_categories()


@pytest.mark.asyncio
async def test_get_tools_known_category(mock_settings):
    mock_tool = MagicMock()
    mock_tool.name = "my_tool"
    register_tools("test_cat", _make_provider([mock_tool]))

    async with get_tools("test_cat") as tools:
        assert len(tools) == 1
        assert tools[0].name == "my_tool"


@pytest.mark.asyncio
async def test_get_tools_unknown_category(mock_settings):
    async with get_tools("nonexistent") as tools:
        assert tools == []


@pytest.mark.asyncio
async def test_get_all_tools_merges(mock_settings):
    tool_a = MagicMock()
    tool_a.name = "tool_a"
    tool_b = MagicMock()
    tool_b.name = "tool_b"

    register_tools("cat_a", _make_provider([tool_a]))
    register_tools("cat_b", _make_provider([tool_b]))

    async with get_all_tools() as tools:
        names = {t.name for t in tools}
        assert names == {"tool_a", "tool_b"}
