from __future__ import annotations

from typing import List, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage

from .config import get_settings


class AgentState(TypedDict):
    messages: List[BaseMessage]
    summary: str
    research_results: List[str]
    needs_more_research: bool
    loop_count: int


def prune_messages(
    messages: List[BaseMessage],
    max_messages: int | None = None,
) -> List[BaseMessage]:
    if max_messages is None:
        max_messages = get_settings().max_context_messages
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]


def build_context_messages(state: AgentState) -> List[BaseMessage]:
    messages = prune_messages(state.get("messages", []))
    summary = state.get("summary", "").strip()
    if summary:
        return [SystemMessage(content=f"Running summary:\n{summary}")] + messages
    return messages
