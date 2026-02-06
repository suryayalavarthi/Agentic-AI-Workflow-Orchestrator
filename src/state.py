from __future__ import annotations

from typing import List, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage

MAX_CONTEXT_MESSAGES = 6


class AgentState(TypedDict):
    messages: List[BaseMessage]
    summary: str
    research_results: List[str]
    needs_more_research: bool
    loop_count: int


def prune_messages(
    messages: List[BaseMessage],
    max_messages: int = MAX_CONTEXT_MESSAGES,
) -> List[BaseMessage]:
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]


def build_context_messages(state: AgentState) -> List[BaseMessage]:
    messages = prune_messages(state.get("messages", []))
    summary = state.get("summary", "").strip()
    if summary:
        return [SystemMessage(content=f"Running summary:\n{summary}")] + messages
    return messages
