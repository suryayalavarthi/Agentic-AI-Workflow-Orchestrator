from __future__ import annotations

import logging
from typing import List

import requests
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun

from .config import get_settings
from .tools.memory import get_vector_db

logger = logging.getLogger(__name__)
mcp = FastMCP("AgenticOrchestrator")


@mcp.tool()
def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo."""
    search = DuckDuckGoSearchRun()
    return str(search.invoke(query))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    reraise=True,
)
def _fetch_url(url: str, timeout: int) -> requests.Response:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response


@mcp.tool()
def web_scraper(url: str) -> str:
    """Fetch a URL and return the first 5000 characters of page text."""
    response = _fetch_url(url, timeout=get_settings().web_scraper_timeout)
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
