from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, List

import streamlit as st
from langchain_core.messages import HumanMessage

from src.config import get_settings
from src.graph import compile_graph

logger = logging.getLogger(__name__)

_cfg = get_settings()
_has_api_key = bool(_cfg.anthropic_api_key)

_NODE_ICONS = {
    "supervisor": "routing",
    "researcher": "searching",
    "analyst": "analyzing",
    "summarizer": "summarizing",
    "draft_outline": "outlining",
    "final_report": "writing",
}


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


def _render_agent_log(placeholder, entries: List[Dict]) -> None:
    if not entries:
        placeholder.markdown("_No activity yet._")
        return
    with placeholder.container():
        current_loop = None
        for entry in entries[-15:]:
            loop = entry.get("loop_count", 0)
            if loop != current_loop:
                current_loop = loop
                st.markdown(f"**--- Loop {loop} ---**")
            icon = _NODE_ICONS.get(entry["node"], "processing")
            status_state = (
                "complete" if entry.get("status") == "complete" else "running"
            )
            with st.status(
                f"{entry['node']} ({icon})", state=status_state, expanded=False
            ):
                st.caption(entry["content"])
                ts = entry.get("timestamp")
                if ts:
                    st.caption(
                        f"at {time.strftime('%H:%M:%S', time.localtime(ts))}"
                    )


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

    try:
        async for event in graph.astream(state, config=config):
            for node_name, output in event.items():
                # Mirror tuple handling from the CLI runner.
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

                log_entry = f"{node_name} completed execution."
                if "reasoning" in output:
                    log_entry = f"{node_name}: {output['reasoning']}"
                logger.info("Node: %s", node_name)

                st.session_state.agent_log.append({
                    "node": node_name,
                    "content": log_entry,
                    "status": "complete",
                    "timestamp": time.time(),
                    "loop_count": output.get("loop_count", 0),
                })
                _render_agent_log(log_placeholder, st.session_state.agent_log)
                if status is not None:
                    status.update(label=f"Running: {node_name}")

                messages = output.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                else:
                    last_msg = None

                if node_name == "final_report" and last_msg is not None:
                    st.session_state.final_report = str(last_msg.content)
    except Exception as exc:  # noqa: BLE001
        # Surface errors both in logs and in the returned state so the UI
        # can display a clear message instead of silently falling back.
        logger.exception("Graph execution failed in Streamlit: %s", exc)
        latest_state["error"] = {
            "type": "graph_failed",
            "detail": str(exc),
        }
    finally:
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

    st.title("Agentic Orchestrator Dashboard")
    if not _has_api_key:
        st.warning("No API Key detected in .env file. Please check your configuration.")

    cfg = get_settings()

    with st.sidebar:
        # --- Settings Panel ---
        with st.expander("Settings", expanded=False):
            model_choice = st.selectbox(
                "LLM Model",
                ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"],
                index=0,
            )
            max_loops = st.slider(
                "Max Research Loops", 1, 20, cfg.max_loop_count
            )
            temperature = st.slider(
                "Temperature", 0.0, 1.0, cfg.default_temperature, 0.1
            )
            st.session_state.model_override = model_choice
            st.session_state.max_loops_override = max_loops
            st.session_state.temperature_override = temperature

        st.header("Agent Thought Process")
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
    # Only fall back to showing the last message as a \"report\" when there
    # is no real final_report and no explicit error from the graph.
    if (
        not st.session_state.final_report
        and not st.session_state.graph_state.get("error")
    ):
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
            st.subheader("Final Research Synthesis")
            exec_tab, deep_tab, sources_tab = st.tabs(
                ["Executive Summary", "Deep Dive", "Sources"]
            )
            with exec_tab:
                # For the Executive Summary tab, hide the top-level markdown
                # heading (\"# Executive Summary\") to avoid duplicating the
                # surrounding \"Final Research Synthesis\" title.
                report = st.session_state.final_report
                lines = report.splitlines()
                if lines and lines[0].lstrip().startswith("#"):
                    exec_md = "\n".join(lines[1:])
                else:
                    exec_md = report.split("##", maxsplit=1)[0]
                st.markdown(exec_md)
            with deep_tab:
                st.markdown(st.session_state.final_report)
            with sources_tab:
                report = st.session_state.final_report
                if "## Sources & References" in report:
                    sources_section = report.split("## Sources & References", 1)[1]
                    st.markdown("## Sources & References" + sources_section)
                else:
                    st.info("No source references found in this report.")

    # Input handling
    user_input = st.chat_input("Ask the orchestrator about scaling, security, or architecture...")
    if user_input:
        _append_chat("user", user_input)
        graph = st.session_state.graph
        config = {
            "configurable": {"thread_id": st.session_state.thread_id},
            "recursion_limit": cfg.recursion_limit,
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
            logger.exception("Graph execution failed in Streamlit: %s", exc)
            st.session_state.graph_state = dict(st.session_state.graph_state)
            st.session_state.graph_state["error"] = {
                "type": "graph_failed",
                "detail": str(exc),
            }
            st.error(f"Graph execution failed: {exc}")
        st.rerun()


if __name__ == "__main__":
    main()
