from __future__ import annotations

import logging
from typing import List, Optional

from ..config import get_settings

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self) -> None:
        cfg = get_settings()
        self._command = cfg.mcp_fetch_command
        args = cfg.mcp_fetch_args
        self._args = [arg for arg in args.split(" ") if arg]

    async def fetch(self, url: str) -> Optional[str]:
        if not self._command:
            return None
        try:
            from mcp import ClientSession, StdioServerParameters  # type: ignore
            from mcp.client.stdio import stdio_client  # type: ignore
        except ImportError as exc:
            logger.warning("MCP SDK not available: %s", exc)
            return None

        params = StdioServerParameters(command=self._command, args=self._args)
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("fetch", {"url": url})
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP fetch failed: %s", exc)
            return None

        return _extract_text_from_result(result)


def _extract_text_from_result(result: object) -> Optional[str]:
    content = getattr(result, "content", None)
    if not content:
        return None
    parts: List[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
        else:
            parts.append(str(item))
    return "\n".join(parts).strip() if parts else None
