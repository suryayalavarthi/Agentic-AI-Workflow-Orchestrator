from __future__ import annotations

import logging
import json
import re
from typing import List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from ..config import get_settings, llm_retry
from ..state import AgentState, prune_messages
from ..tools.memory import get_vector_db

logger = logging.getLogger(__name__)
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
            return str(payload["synthesis"])
    return None


def _last_user_query(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _extract_comparison_rows(text: str) -> List[List[str]]:
    if not text:
        return []
    match = re.search(
        r"COMPARISON_DATA\s*:\s*(.+?)(?:\n\n|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    raw_block = match.group(1).strip()
    if not raw_block:
        return []
    rows: List[List[str]] = []
    for line in raw_block.splitlines():
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|") if part.strip()]
        if len(parts) >= 2:
            rows.append(parts)
    return rows


SUMMARIZER_SYSTEM = """You are the Summarizer node.
Condense the conversation into a concise running summary.
Keep key decisions, tool outputs, and open questions.
"""

DRAFT_OUTLINE_SYSTEM = """You are the Draft Outline node.
Create a detailed outline for a final report using the research results.
Focus on structure, not full prose.
"""

FINAL_REPORT_SYSTEM = """You are the Final Report node.
Write a professional, 2000-word report using the outline and research data.
If possible, save the report using the filesystem tool.
"""


def _build_llm() -> ChatAnthropic:
    cfg = get_settings()
    return ChatAnthropic(
        model=cfg.default_model,
        temperature=cfg.default_temperature,
        api_key=cfg.anthropic_api_key,
    )


async def _run_tool_calls(
    response: BaseMessage,
    tools: List[BaseTool],
) -> List[ToolMessage]:
    tool_messages: List[ToolMessage] = []
    tool_map = {tool.name: tool for tool in tools}
    for call in getattr(response, "tool_calls", []) or []:
        tool = tool_map.get(call.get("name"))
        if not tool:
            continue
        if hasattr(tool, "ainvoke"):
            result = await tool.ainvoke(call.get("args", {}))
        else:
            result = tool.invoke(call.get("args", {}))
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=call.get("id", ""))
        )
    return tool_messages


def summarizer_node(state: AgentState) -> AgentState:
    llm = _build_llm()
    system_message = SystemMessage(content=SUMMARIZER_SYSTEM)
    prior_messages = [
        message
        for message in prune_messages(state.get("messages", []))
        if not isinstance(message, SystemMessage)
    ]
    messages: List[BaseMessage] = [system_message] + prior_messages

    @llm_retry()
    def _invoke(msgs):
        return llm.invoke(msgs)

    response = _invoke(messages)

    logger.info("Summarizer updated running summary")

    # Preserve the last human message so context is not lost after summarization
    last_human_message = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        None
    )
    new_messages = [last_human_message] if last_human_message else []

    return {
        "messages": new_messages,
        "summary": response.content,
        "research_results": state.get("research_results", []),
        "needs_more_research": state.get("needs_more_research", False),
        "loop_count": state.get("loop_count", 0) + 1,
    }


def draft_outline_node(state: AgentState) -> AgentState:
    llm = _build_llm()
    system_parts = [DRAFT_OUTLINE_SYSTEM]
    summary = state.get("summary", "").strip()
    if summary:
        system_parts.append(f"Running summary:\n{summary}")
    if state.get("research_results"):
        results_text = "\n---\n".join(
            str(result) for result in state["research_results"]
        )
        system_parts.append(f"Research results:\n{results_text}")
    system_message = SystemMessage(content="\n\n".join(system_parts))
    prior_messages = [
        message
        for message in prune_messages(state.get("messages", []))
        if not isinstance(message, SystemMessage)
    ]
    messages: List[BaseMessage] = [system_message] + prior_messages

    @llm_retry()
    def _invoke(msgs):
        return llm.invoke(msgs)

    response = _invoke(messages)

    logger.info("Draft outline created")

    return {
        "messages": [AIMessage(content=response.content)],
        "summary": state.get("summary", ""),
        "research_results": state.get("research_results", []),
        "needs_more_research": state.get("needs_more_research", False),
        "loop_count": state.get("loop_count", 0) + 1,
    }


async def final_report_node(state: AgentState) -> AgentState:
    summary = state.get("summary", "").strip()
    results = [str(result) for result in state.get("research_results", []) if result]
    # Build a more readable \"Research Results\" section by showing short,
    # plain-text snippets instead of full raw JSON/tool payloads.
    snippets: List[str] = []
    for idx, item in enumerate(results, 1):
        text = str(item).strip().replace("\n", " ")
        if len(text) > 400:
            text = text[:400].rstrip() + "..."
        snippets.append(f"- **Source {idx}**: {text}")
    synthesis = _extract_synthesis(state)

    report_parts = ["# Executive Summary"]
    if summary:
        report_parts.append(summary)
    elif synthesis:
        report_parts.append(synthesis.split("\n\n", maxsplit=1)[0])
    elif results:
        report_parts.append("Research completed. See details below for the gathered information.")
    else:
        report_parts.append("No summary or research results available for the current query.")

    comparison_rows = _extract_comparison_rows(synthesis or "")
    if comparison_rows:
        report_parts.append("## Comparison Matrix")
        header = comparison_rows[0]
        report_parts.append("| " + " | ".join(header) + " |")
        report_parts.append("| " + " | ".join("---" for _ in header) + " |")
        for row in comparison_rows[1:]:
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            report_parts.append("| " + " | ".join(row[: len(header)]) + " |")

    if synthesis:
        report_parts.append("## Detailed Analysis")
        report_parts.append("### 🔍 Analyst's Final Assessment")
        report_parts.append(synthesis)

    report_parts.append("## Research Results")
    if snippets:
        report_parts.extend(snippets)
    else:
        report_parts.append("No research results available.")

    query = synthesis or summary or _last_user_query(state)
    if query:
        source_hits = get_vector_db().retrieve_knowledge_with_sources(query, k=6)
        urls: List[str] = []
        for hit in source_hits:
            url = (hit.get("source_url") or "").strip()
            # Skip placeholder / unknown entries so the Sources section only
            # shows real, clickable URLs.
            if not url or url.lower() == "unknown":
                continue
            if url not in urls:
                urls.append(url)
        if urls:
            report_parts.append("## Sources & References")
            report_parts.extend(f"- [{url}]({url})" for url in urls)

    report = "\n\n".join(report_parts)
    logger.info("Final report generated")

    return {
        "messages": [AIMessage(content=report)],
        "summary": state.get("summary", ""),
        "research_results": state.get("research_results", []),
        "needs_more_research": False,
        "loop_count": state.get("loop_count", 0) + 1,
    }
