from __future__ import annotations

from langchain_core.messages import ToolMessage

from src.agents.researcher import _extract_tool_outputs, _normalize_content


def test_normalize_content_string(mock_settings):
    assert _normalize_content("hello") == "hello"


def test_normalize_content_list(mock_settings):
    content = [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"},
    ]
    result = _normalize_content(content)
    assert "hello" in result
    assert "world" in result


def test_normalize_content_mixed_list(mock_settings):
    content = [
        {"type": "text", "text": "hello"},
        "plain string",
    ]
    result = _normalize_content(content)
    assert "hello" in result
    assert "plain string" in result


def test_extract_tool_outputs_with_url(mock_settings):
    messages = [
        ToolMessage(content="page content", tool_call_id="1"),
    ]
    args = [{"url": "https://example.com"}]
    outputs = _extract_tool_outputs(messages, args)
    assert len(outputs) == 1
    assert outputs[0] == ("page content", "https://example.com")


def test_extract_tool_outputs_no_url(mock_settings):
    messages = [
        ToolMessage(content="search results", tool_call_id="1"),
    ]
    args = [{"query": "test"}]
    outputs = _extract_tool_outputs(messages, args)
    assert len(outputs) == 1
    assert outputs[0] == ("search results", "unknown")


def test_extract_tool_outputs_empty(mock_settings):
    assert _extract_tool_outputs([], []) == []
