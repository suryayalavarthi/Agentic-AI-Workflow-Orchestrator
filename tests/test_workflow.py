from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.workflow import (
    _needs_summarization,
    _route_after_analyst,
    _route_after_supervisor,
    build_graph,
)


def test_route_after_supervisor_default(mock_settings):
    state = {
        "messages": [],
        "loop_count": 0,
        "summary": "",
        "research_results": [],
        "needs_more_research": False,
    }
    assert _route_after_supervisor(state) == "researcher"


def test_route_after_supervisor_parses_json(mock_settings):
    payload = json.dumps({"next_agent": "analyst", "reasoning": "test"})
    state = {
        "messages": [AIMessage(content=payload)],
        "loop_count": 0,
        "summary": "",
        "research_results": [],
        "needs_more_research": False,
    }
    assert _route_after_supervisor(state) == "analyst"


def test_route_after_supervisor_force_final_at_max_loops(mock_settings):
    state = {
        "messages": [HumanMessage(content="test")],
        "loop_count": 16,
        "summary": "",
        "research_results": [],
        "needs_more_research": False,
    }
    assert _route_after_supervisor(state) == "final_report"


def test_route_after_supervisor_triggers_summarizer(mock_settings):
    mock_settings.max_context_messages = 3
    state = {
        "messages": [HumanMessage(content=str(i)) for i in range(5)],
        "loop_count": 0,
        "summary": "",
        "research_results": [],
        "needs_more_research": False,
    }
    assert _route_after_supervisor(state) == "summarizer"


def test_route_after_analyst_needs_more(mock_settings):
    state = {
        "messages": [],
        "loop_count": 0,
        "needs_more_research": True,
        "summary": "",
        "research_results": [],
    }
    assert _route_after_analyst(state) == "researcher"


def test_route_after_analyst_done(mock_settings):
    state = {
        "messages": [],
        "loop_count": 0,
        "needs_more_research": False,
        "summary": "",
        "research_results": [],
    }
    assert _route_after_analyst(state) == "final_report"


def test_route_after_analyst_force_final_at_max_loops(mock_settings):
    state = {
        "messages": [],
        "loop_count": 16,
        "needs_more_research": True,
        "summary": "",
        "research_results": [],
    }
    assert _route_after_analyst(state) == "final_report"


def test_build_graph_nodes(mock_settings):
    graph = build_graph()
    node_names = set(graph.nodes.keys())
    expected = {"supervisor", "researcher", "analyst", "summarizer", "draft_outline", "final_report"}
    assert expected == node_names


def test_needs_summarization_true(mock_settings):
    mock_settings.max_context_messages = 6
    state = {"messages": [HumanMessage(content=str(i)) for i in range(7)]}
    assert _needs_summarization(state) is True


def test_needs_summarization_false(mock_settings):
    mock_settings.max_context_messages = 6
    state = {"messages": [HumanMessage(content=str(i)) for i in range(5)]}
    assert _needs_summarization(state) is False
