from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from langchain_core.tools import BaseTool

from .registry import get_all_tools


@asynccontextmanager
async def get_search_tools() -> AsyncIterator[List[BaseTool]]:
    async with get_all_tools() as tools:
        yield tools
