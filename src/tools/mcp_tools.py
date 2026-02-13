from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.sessions import Connection, create_session
from langchain_mcp_adapters.tools import load_mcp_tools

logger = logging.getLogger(__name__)

def _server_connection() -> Connection:
    project_root = Path(__file__).resolve().parents[2]
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "src.server"],
        "cwd": str(project_root),
    }


@asynccontextmanager
async def get_research_tools() -> AsyncIterator[List[BaseTool]]:
    try:
        connection = _server_connection()
        async with create_session(connection) as session:
            await session.initialize()
            tools = await load_mcp_tools(
                session,
                server_name="local_mcp",
                tool_name_prefix=True,
            )
            yield tools
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load MCP tools: %s", exc)
        yield []


@asynccontextmanager
async def get_report_tools() -> AsyncIterator[List[BaseTool]]:
    yield []


# Register providers with the tool registry
from .registry import register_tools  # noqa: E402

register_tools("research", get_research_tools)
register_tools("report", get_report_tools)
