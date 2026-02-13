from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from src.agents.supervisor import SupervisorDecision, supervisor_node


def test_supervisor_decision_model_valid():
    d = SupervisorDecision(next_agent="researcher", reasoning="need data")
    assert d.next_agent == "researcher"


def test_supervisor_decision_model_invalid():
    with pytest.raises(Exception):
        SupervisorDecision(next_agent="unknown_agent", reasoning="bad")


def test_supervisor_routes_to_researcher(mock_settings, sample_state, mock_llm):
    mock_llm.invoke.return_value = SupervisorDecision(
        next_agent="researcher", reasoning="need data"
    )
    mock_llm.with_structured_output.return_value = mock_llm

    with patch("src.agents.supervisor._build_llm", return_value=mock_llm):
        result = supervisor_node(sample_state)

    assert "messages" in result
    payload = json.loads(result["messages"][0].content)
    assert payload["next_agent"] == "researcher"
    assert result["loop_count"] == 1


def test_supervisor_detects_synthesis_completion(mock_settings, mock_llm):
    analyst_payload = json.dumps({
        "needs_more_research": False,
        "gaps": [],
        "re_research_instructions": "",
        "synthesis": "Final analysis complete.",
    })
    state = {
        "messages": [AIMessage(content=analyst_payload)],
        "summary": "",
        "research_results": [],
        "needs_more_research": False,
        "loop_count": 2,
    }
    mock_llm.invoke.return_value = SupervisorDecision(
        next_agent="researcher", reasoning="initial"
    )
    mock_llm.with_structured_output.return_value = mock_llm

    with patch("src.agents.supervisor._build_llm", return_value=mock_llm):
        result = supervisor_node(state)

    payload = json.loads(result["messages"][0].content)
    assert payload["next_agent"] == "final_report"
