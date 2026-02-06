from __future__ import annotations

import logging
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from ..state import AgentState, prune_messages
from ..tools.mcp_tools import get_research_tools
from ..tools.memory import get_vector_db

logger = logging.getLogger(__name__)

RESEARCHER_SYSTEM = """You are the Researcher agent.
Use MCP tools to gather data.
If tool output is unclear, state that you lack information.
Use duckduckgo_search to find relevant links, then use web_scraper to read the
content of the most promising official documentation or technical articles to
get deep details.
"""

def _build_llm() -> ChatAnthropic:
    # Ensuring we use the 2026 stable model string
    return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)

async def _run_tool_calls(
    response: BaseMessage,
    tools: List[BaseTool],
) -> tuple[List[ToolMessage], List[dict]]:
    tool_messages: List[ToolMessage] = []
    tool_args: List[dict] = []
    tool_map = {tool.name: tool for tool in tools}
    
    for call in getattr(response, "tool_calls", []) or []:
        tool = tool_map.get(call.get("name"))
        if not tool:
            continue
        
        # Check if the tool itself is async or sync and handle accordingly
        if hasattr(tool, "ainvoke"):
            result = await tool.ainvoke(call.get("args", {}))
        else:
            result = tool.invoke(call.get("args", {}))
            
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=call.get("id", ""))
        )
        tool_args.append(call.get("args", {}) or {})
    return tool_messages, tool_args

def _normalize_content(content: object) -> str:
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            else:
                text_parts.append(str(item))
        return "\n".join(part for part in text_parts if part)
    return str(content)

def _extract_tool_outputs(
    tool_messages: List[ToolMessage],
    tool_args: List[dict],
) -> List[tuple[str, str]]:
    outputs: List[tuple[str, str]] = []
    for message, args in zip(tool_messages, tool_args, strict=False):
        if message.content:
            source_url = str(args.get("url", "unknown"))
            outputs.append((str(message.content), source_url))
    return outputs


async def _store_research_via_tool(
    tools: List[BaseTool],
    text: str,
    source_url: str,
) -> None:
    store_tool = next(
        (tool for tool in tools if tool.name.endswith("store_research")),
        None,
    )
    if not store_tool:
        return
    payload = {"text": text, "source_url": source_url}
    if hasattr(store_tool, "ainvoke"):
        await store_tool.ainvoke(payload)
    else:
        store_tool.invoke(payload)

async def researcher_node(state: AgentState) -> AgentState:
    llm = _build_llm()

    async with get_research_tools() as tools:
        tool_aware = llm.bind_tools(tools)
        system_parts = [RESEARCHER_SYSTEM]
        summary = state.get("summary", "").strip()
        if summary:
            system_parts.append(f"Running summary:\n{summary}")
        system_message = SystemMessage(content="\n\n".join(system_parts))
        prior_messages = [
            message
            for message in prune_messages(state.get("messages", []))
            if not isinstance(message, SystemMessage)
        ]
        messages: List[BaseMessage] = [system_message] + prior_messages

        response = await tool_aware.ainvoke(messages)
        tool_messages, tool_args = await _run_tool_calls(response, tools)

        final_response = response
        if tool_messages:
            final_response = await tool_aware.ainvoke(
                messages + [response] + tool_messages
            )

        normalized_content = _normalize_content(final_response.content)
        research_results = state.get("research_results", [])
        tool_outputs = _extract_tool_outputs(tool_messages, tool_args)
        for output, source_url in tool_outputs:
            research_results.append(output)
            await _store_research_via_tool(tools, output, source_url)
        get_vector_db().store_research(
            output,
            source="web_scraper",
            source_url=source_url,
        )
        if normalized_content:
            research_results.append(normalized_content)

    logger.info("Researcher produced %d tool messages", len(tool_messages))

    return {
        "messages": [AIMessage(content=normalized_content)],
        "summary": state.get("summary", ""),
        "research_results": research_results,
        "needs_more_research": state.get("needs_more_research", False),
        "loop_count": state.get("loop_count", 0) + 1,
    }