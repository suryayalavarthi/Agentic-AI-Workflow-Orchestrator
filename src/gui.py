from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()
_has_api_key = bool(
    __import__("os").environ.get("ANTHROPIC_API_KEY")
    or __import__("os").environ.get("OPENAI_API_KEY")
)
print(f"DEBUG: API Key found: {_has_api_key}")

import asyncio
import json
from typing import Dict, List, Optional, Tuple

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.graph import compile_graph

def _init_session() -> None:
    if "graph" not in st.session_state:
        st.session_state.graph = compile_graph()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "streamlit"
    if "graph_state" not in st.session_state:
        st.session_state.graph_state = {
            "messages": [],
            "summary": "",
            "research_results": [],
            "needs_more_research": True,
            "loop_count": 0,
        }
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_log" not in st.session_state:
        st.session_state.agent_log = []
    if "final_report" not in st.session_state:
        st.session_state.final_report = ""

def _append_chat(role: str, content: str) -> None:
    st.session_state.chat_history.append({"role": role, "content": content})

def _is_internal_message(content: str) -> bool:
    content = content.strip()
    if content.startswith("{") and content.endswith("}"):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return False
        if isinstance(payload, dict) and (
            "next_agent" in payload or "needs_more_research" in payload
        ):
            return True
    return False

def _render_agent_log(placeholder, entries: List[Dict[str, str]]) -> None:
    if not entries:
        placeholder.markdown("_No activity yet._")
        return
    with placeholder.container():
        for entry in entries[-10:]:
            status_state = (
                "complete" if entry.get("status") == "complete" else "running"
            )
            with st.status(entry["node"], state=status_state, expanded=False):
                st.caption(entry["content"])


async def _run_graph_async(
    graph,
    state: Dict,
    config: Dict,
    log_placeholder,
    status_placeholder,
) -> Dict:
    latest_state = dict(state)
    _render_agent_log(log_placeholder, st.session_state.agent_log)
    status = None
    if hasattr(st, "status"):
        status = status_placeholder.status("Running graph...", expanded=True)

    async for event in graph.astream(state, config=config):
        for node_name, output in event.items():
            if isinstance(output, tuple):
                if not output:
                    continue
                output = output[0]
            if not isinstance(output, dict):
                continue

            latest_state.update(output)
            
            log_entry = f"{node_name} completed execution."
            if "reasoning" in output:
                log_entry = f"{node_name}: {output['reasoning']}"
            print(f"[node] {node_name}")

            st.session_state.agent_log.append(
                {"node": node_name, "content": log_entry, "status": "complete"}
            )
            _render_agent_log(log_placeholder, st.session_state.agent_log)
            if status is not None:
                status.update(label=f"Running: {node_name}")
            
            messages = output.get("messages", [])
            if messages:
                last_msg = messages[-1]
                # Assistant messages are not pushed into the main chat; only Final
                # Report is rendered separately for a cleaner UX.
            
            if node_name == "final_report":
                st.session_state.final_report = str(last_msg.content)

    if status is not None:
        status.update(label="Completed", state="complete")
    return latest_state

def _run_graph(
    graph,
    state: Dict,
    config: Dict,
    log_placeholder,
    status_placeholder,
) -> Dict:
    try:
        return asyncio.run(
            _run_graph_async(graph, state, config, log_placeholder, status_placeholder)
        )
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                _run_graph_async(
                    graph,
                    state,
                    config,
                    log_placeholder,
                    status_placeholder,
                )
            )
        finally:
            loop.close()

def main() -> None:
    st.set_page_config(page_title="Agentic Orchestrator", layout="wide")
    _init_session()

    st.title("🤖 Agentic Orchestrator Dashboard")
    if not _has_api_key:
        st.warning("No API Key detected in .env file. Please check your configuration.")

    with st.sidebar:
        st.header("🕵️ Agent Thought Process")
        if st.button("Clear Logs"):
            st.session_state.agent_log = []
        log_placeholder = st.empty()
        status_placeholder = st.empty()
        _render_agent_log(log_placeholder, st.session_state.agent_log)

    # Main Chat Interface
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            with st.chat_message("user"):
                st.markdown(entry["content"])

    if not st.session_state.final_report:
        last_message = (
            st.session_state.graph_state.get("messages", [])[-1]
            if st.session_state.graph_state.get("messages")
            else None
        )
        if last_message and getattr(last_message, "content", None):
            st.session_state.final_report = str(last_message.content)

    if st.session_state.final_report:
        st.divider()
        with st.container(border=True):
            st.subheader("📝 Final Research Synthesis")
            exec_tab, deep_tab, sources_tab = st.tabs(
                ["Executive Summary", "Deep Dive", "Sources"]
            )
            with exec_tab:
                st.markdown(st.session_state.final_report.split("##", maxsplit=1)[0])
            with deep_tab:
                st.markdown(st.session_state.final_report)
            with sources_tab:
                with st.expander("View Raw Data", expanded=False):
                    st.markdown(st.session_state.final_report)

    # Input handling
    user_input = st.chat_input("Ask the orchestrator about scaling, security, or architecture...")
    if user_input:
        _append_chat("user", user_input)
        graph = st.session_state.graph
        config = {
            "configurable": {"thread_id": st.session_state.thread_id},
            "recursion_limit": 25,
        }
        
        current_state = dict(st.session_state.graph_state)
        # Reset per-turn state
        current_state["loop_count"] = 0
        current_state["research_results"] = []
        current_state["needs_more_research"] = True
        
        # Clear messages to start fresh context for the new turn.
        # Historical context is preserved in the 'summary' field.
        current_state["messages"] = [HumanMessage(content=user_input)]
        
        st.session_state.graph_state = current_state

        try:
            with st.spinner("Running agents..."):
                st.session_state.graph_state = _run_graph(
                    graph,
                    current_state,
                    config,
                    log_placeholder,
                    status_placeholder,
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Graph execution failed: {exc}")
        st.rerun()

if __name__ == "__main__":
    main()