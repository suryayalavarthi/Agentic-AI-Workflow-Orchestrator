"""MCP-based Agentic Orchestrator.

Python app acts as the Host, the SDK is the Client, and external tools are Servers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, Optional

from langchain_core.messages import HumanMessage

from .config import get_settings
from .graph import compile_graph
from .state import AgentState

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    cfg = get_settings()
    logging.basicConfig(
        level=cfg.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _load_state(graph, config: Dict[str, Dict[str, str]]) -> Optional[AgentState]:
    snapshot = graph.get_state(config)
    return snapshot.values if snapshot and snapshot.values else None


# CHANGED: Added 'async' and switched to 'async for' with 'astream'
async def _stream_with_state(
    graph, state: AgentState, config: Dict[str, Dict[str, str]]
) -> AgentState:
    latest_state: AgentState = dict(state)
    async for event in graph.astream(state, config=config):
        for node_name, output in event.items():
            if isinstance(output, tuple):
                try:
                    if not output:
                        continue
                    output = output[0]
                except (IndexError, TypeError):
                    continue
            if not isinstance(output, dict):
                continue
            latest_state.update(output)
            messages = output.get("messages", [])
            if not messages:
                continue
            last_message = messages[-1]
            logger.info("%s> %s", node_name, last_message.content)
    return latest_state

def _extract_synthesis(state: AgentState) -> Optional[str]:
    for message in reversed(state.get("messages", [])):
        try:
            payload = json.loads(message.content)
        except (TypeError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("needs_more_research") is False
            and payload.get("synthesis")
        ):
            return payload["synthesis"]
        if (
            isinstance(payload, dict)
            and payload.get("needs_more_research") is False
            and not payload.get("synthesis")
        ):
            last_message = state.get("messages", [])[-1] if state.get("messages") else None
            if last_message and getattr(last_message, "content", None):
                return str(last_message.content)
    return None


# CHANGED: Main loop now wrapped in an async function
async def run_cli() -> None:
    cfg = get_settings()
    _configure_logging()
    graph = compile_graph()
    config = {
        "configurable": {"thread_id": cfg.thread_id},
        "recursion_limit": cfg.recursion_limit,
    }

    print("Agentic Orchestrator CLI. Type 'exit' to quit.")
    while True:
        # Note: input() is blocking, which is fine for a simple CLI loop
        user_input = input("\nUser> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        current_state = _load_state(graph, config)
        if not current_state:
            current_state = {
                "messages": [],
                "summary": "",
                "research_results": [],
                "needs_more_research": True,
                "loop_count": 0,
            }
        else:
            # Reset per-turn research artifacts to avoid stale final reports.
            current_state["research_results"] = []
            current_state["summary"] = ""
            current_state["needs_more_research"] = True
            current_state["loop_count"] = 0

        current_state["messages"] = list(current_state.get("messages", [])) + [
            HumanMessage(content=user_input)
        ]

        # CHANGED: Await the async stream
        current_state = await _stream_with_state(graph, current_state, config)

        snapshot = graph.get_state(config)
        if snapshot and snapshot.values:
            synthesis = _extract_synthesis(snapshot.values)
            if synthesis:
                print("\nFinal Report:\n")
                print(synthesis)
                continue

        if snapshot and snapshot.next and "final_report" in snapshot.next:
            approval = input(
                "\nDraft outline ready. Approve or provide feedback: "
            ).strip()
            if approval:
                resume_state = snapshot.values or current_state
                resume_state["messages"] = list(resume_state.get("messages", [])) + [
                    HumanMessage(content=approval)
                ]
                current_state = await _stream_with_state(graph, resume_state, config)


if __name__ == "__main__":
    # CHANGED: Entry point to start the asyncio event loop
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\nExiting...")