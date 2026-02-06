from __future__ import annotations

import json
import logging
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ..agents.analyst import analyst_node
from ..agents.researcher import researcher_node
from ..agents.supervisor import supervisor_node
from ..state import AgentState
from .nodes import draft_outline_node, final_report_node, summarizer_node

logger = logging.getLogger(__name__)

def _needs_summarization(state: AgentState) -> bool:
    return len(state.get("messages", [])) > 6

def _route_after_supervisor(state: AgentState) -> str:
    if state.get("loop_count", 0) > 15:  # Increased limit, handled by recursion_limit too
        logger.warning("Max loops reached after supervisor, forcing final report")
        return "final_report"
    if _needs_summarization(state):
        return "summarizer"

    next_agent: Literal[
        "researcher", "analyst", "draft_outline", "final_report"
    ] = "researcher"
    if state.get("messages"):
        last_message = state["messages"][-1]
        try:
            # Check if content is already a dict (some LLMs return objects directly)
            if isinstance(last_message.content, dict):
                payload = last_message.content
            else:
                payload = json.loads(last_message.content)
            
            if payload.get("next_agent") in {
                "researcher",
                "analyst",
                "draft_outline",
                "final_report",
            }:
                next_agent = payload["next_agent"]
        except (TypeError, json.JSONDecodeError, AttributeError) as exc:
            logger.warning("Supervisor routing parse failed: %s", exc)

    return next_agent

def _route_after_analyst(state: AgentState) -> str:
    if state.get("loop_count", 0) > 15:
        logger.warning("Max loops reached after analyst, forcing final report")
        return "final_report"
    if state.get("needs_more_research"):
        return "researcher"
    return "final_report"

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("draft_outline", draft_outline_node)
    graph.add_node("final_report", final_report_node)

    graph.set_entry_point("supervisor")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("draft_outline", "final_report")
    graph.add_edge("summarizer", "supervisor")
    graph.add_edge("final_report", END)

    graph.add_conditional_edges("supervisor", _route_after_supervisor)
    graph.add_conditional_edges("analyst", _route_after_analyst)

    return graph

def compile_graph():
    """Compiles the graph with an in-memory checkpointer."""
    saver = MemorySaver()
    return build_graph().compile(
        checkpointer=saver,
    )