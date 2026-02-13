from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.state import AgentState, build_context_messages, prune_messages


def test_prune_messages_under_limit(mock_settings):
    msgs = [HumanMessage(content=str(i)) for i in range(3)]
    result = prune_messages(msgs, max_messages=6)
    assert len(result) == 3


def test_prune_messages_over_limit(mock_settings):
    msgs = [HumanMessage(content=str(i)) for i in range(10)]
    result = prune_messages(msgs, max_messages=6)
    assert len(result) == 6
    assert result[0].content == "4"


def test_prune_messages_uses_config_default(mock_settings):
    mock_settings.max_context_messages = 3
    msgs = [HumanMessage(content=str(i)) for i in range(5)]
    result = prune_messages(msgs)
    assert len(result) == 3


def test_build_context_messages_with_summary(mock_settings):
    state: AgentState = {
        "messages": [HumanMessage(content="hello")],
        "summary": "prior context",
        "research_results": [],
        "needs_more_research": False,
        "loop_count": 0,
    }
    result = build_context_messages(state)
    assert len(result) == 2
    assert isinstance(result[0], SystemMessage)
    assert "prior context" in result[0].content


def test_build_context_messages_no_summary(mock_settings):
    state: AgentState = {
        "messages": [HumanMessage(content="hello")],
        "summary": "",
        "research_results": [],
        "needs_more_research": False,
        "loop_count": 0,
    }
    result = build_context_messages(state)
    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
