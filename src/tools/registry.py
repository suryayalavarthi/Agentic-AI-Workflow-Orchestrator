"""Tool registration system for extensible tool management.

Providers are async context managers that yield lists of LangChain BaseTools.
Register them under a category name, then load them individually or all at once.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Dict, List

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

ToolProvider = Callable[[], AsyncIterator[List[BaseTool]]]

_registry: Dict[str, ToolProvider] = {}


def register_tools(category: str, provider: ToolProvider) -> None:
    _registry[category] = provider
    logger.info("Registered tool category: %s", category)


def list_categories() -> List[str]:
    return list(_registry.keys())


@asynccontextmanager
async def get_tools(category: str) -> AsyncIterator[List[BaseTool]]:
    provider = _registry.get(category)
    if provider is None:
        logger.warning("No tool provider for category: %s", category)
        yield []
        return
    async with provider() as tools:
        yield tools


@asynccontextmanager
async def get_all_tools() -> AsyncIterator[List[BaseTool]]:
    all_tools: List[BaseTool] = []
    for category, provider in _registry.items():
        try:
            async with provider() as tools:
                all_tools.extend(tools)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load tools for %s: %s", category, exc)
    yield all_tools
