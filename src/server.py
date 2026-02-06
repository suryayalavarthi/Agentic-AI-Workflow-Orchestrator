from __future__ import annotations

import logging
from typing import List

import requests
from bs4 import BeautifulSoup
from fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun

from .tools.memory import get_vector_db

logger = logging.getLogger(__name__)
mcp = FastMCP("AgenticOrchestrator")


@mcp.tool()
def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo."""
    search = DuckDuckGoSearchRun()
    return str(search.invoke(query))


@mcp.tool()
def web_scraper(url: str) -> str:
    """Fetch a URL and return the first 5000 characters of page text."""
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:5000]


@mcp.tool()
def store_research(text: str, source_url: str = "unknown") -> str:
    """Embed and store research text in the vector database."""
    return get_vector_db().store_research(
        text=text,
        source="web_scraper",
        source_url=source_url,
    )


@mcp.tool()
def retrieve_knowledge(query: str, k: int = 3) -> List[str]:
    """Retrieve semantically similar research entries from the vector database."""
    return get_vector_db().retrieve_knowledge(query=query, k=k)


if __name__ == "__main__":
    mcp.run()
