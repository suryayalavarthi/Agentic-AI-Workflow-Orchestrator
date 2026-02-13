from __future__ import annotations

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.nodes import (
    _extract_comparison_rows,
    _extract_synthesis,
    _last_user_query,
    final_report_node,
)


def test_extract_synthesis_found(mock_settings):
    payload = json.dumps({
        "needs_more_research": False,
        "gaps": [],
        "re_research_instructions": "",
        "synthesis": "Final analysis here",
    })
    state = {"messages": [AIMessage(content=payload)]}
    assert _extract_synthesis(state) == "Final analysis here"


def test_extract_synthesis_none(mock_settings):
    state = {"messages": [HumanMessage(content="hello")]}
    assert _extract_synthesis(state) is None


def test_extract_synthesis_skips_needs_research(mock_settings):
    payload = json.dumps({
        "needs_more_research": True,
        "gaps": ["missing data"],
        "re_research_instructions": "search more",
        "synthesis": "",
    })
    state = {"messages": [AIMessage(content=payload)]}
    assert _extract_synthesis(state) is None


def test_last_user_query(mock_settings):
    state = {
        "messages": [
            HumanMessage(content="first"),
            AIMessage(content="response"),
            HumanMessage(content="second"),
        ]
    }
    assert _last_user_query(state) == "second"


def test_last_user_query_empty(mock_settings):
    state = {"messages": [AIMessage(content="no human")]}
    assert _last_user_query(state) == ""


def test_extract_comparison_rows(mock_settings):
    text = (
        "Some analysis.\n"
        "COMPARISON_DATA:\n"
        "Metric | SQLite | PostgreSQL\n"
        "Transport | stdio | SSE\n"
        "Latency | Ultra-low | Network-dependent\n"
    )
    rows = _extract_comparison_rows(text)
    assert len(rows) == 3
    assert rows[0] == ["Metric", "SQLite", "PostgreSQL"]
    assert rows[1] == ["Transport", "stdio", "SSE"]


def test_extract_comparison_rows_none(mock_settings):
    assert _extract_comparison_rows("No comparison data here") == []


def test_extract_comparison_rows_empty(mock_settings):
    assert _extract_comparison_rows("") == []


def test_final_report_includes_sources(mock_settings, monkeypatch, mock_vector_db):
    # Ensure the vector DB used by final_report_node is our mocked instance.
    monkeypatch.setattr("src.graph.nodes.get_vector_db", lambda: mock_vector_db)

    state = {
        "messages": [HumanMessage(content="Compare SQLite vs PostgreSQL")],
        "summary": "Summary of trade-offs.",
        "research_results": ["Result 1"],
        "needs_more_research": False,
        "loop_count": 0,
    }

    # final_report_node is async, so run it in an event loop.
    result = asyncio.run(final_report_node(state))

    assert "messages" in result
    report = result["messages"][0].content
    assert "## Sources & References" in report
    # The mock_vector_db fixture returns a single URL.
    assert "https://example.com" in report
