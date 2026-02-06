from __future__ import annotations

import json
import logging
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..state import AgentState, prune_messages
from ..tools.mcp_tools import get_research_tools
from ..tools.memory import get_vector_db

logger = logging.getLogger(__name__)


ANALYST_SYSTEM = """You are the Analyst agent.
Evaluate the Researcher's output for gaps.
If evidence is missing, request more research instead of guessing.
Return structured output with gaps and re-research instructions.
You must look at the RAW TOOL OUTPUTS in the message history to find technical
details like JSON-RPC versions. Do not just rely on the Researcher's summaries.
If you need to verify a technical detail from previous research before making
your final assessment, use the retrieve_knowledge tool to query your internal
memory.
When comparing two technologies, you MUST end your synthesis with a section
titled COMPARISON_DATA containing a multi-line list of rows. Each row must be
pipe-separated with values for BOTH technologies. Example:
COMPARISON_DATA:
Metric | SQLite | PostgreSQL
Transport | stdio | SSE
Latency | Ultra-low | Network-dependent
Your output must be a valid tool call/JSON object. The "gaps" field MUST be a
JSON array of strings (even if empty). Do NOT use Markdown bullet points inside
any JSON field values.
"""


class AnalystAssessment(BaseModel):
    needs_more_research: bool = Field(
        ...,
        description="True if more research is required.",
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="JSON array of strings listing specific missing details.",
    )
    re_research_instructions: str = Field(
        "",
        description="Concrete instructions for the Researcher if gaps exist.",
    )
    synthesis: str = Field(
        "",
        description="Structured synthesis if research is sufficient.",
    )


def _build_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)


async def analyst_node(state: AgentState) -> AgentState:
    async with get_research_tools() as tools:
        tool_aware = _build_llm().bind_tools(tools).with_structured_output(
            AnalystAssessment
        )
    system_parts = [ANALYST_SYSTEM]
    summary = state.get("summary", "").strip()
    if summary:
        system_parts.append(f"Running summary:\n{summary}")
    if state.get("research_results"):
        results_text = "\n---\n".join(
            str(result) for result in state["research_results"]
        )
        system_parts.append(f"Research results:\n{results_text}")
    last_user_message = next(
        (
            message.content
            for message in reversed(state.get("messages", []))
            if isinstance(message, HumanMessage)
        ),
        "",
    )
    if last_user_message:
        vector_hits = get_vector_db().retrieve_knowledge_with_sources(
            last_user_message, k=4
        )
        if vector_hits:
            vector_lines = [
                f"[{hit['source_url']}] {hit['text']}"
                for hit in vector_hits
                if hit.get("text")
            ]
            vector_text = "\n---\n".join(vector_lines)
            system_parts.append(f"Vector DB facts:\n{vector_text}")
    system_message = SystemMessage(content="\n\n".join(system_parts))
    prior_messages = [
        message
        for message in prune_messages(state.get("messages", []))
        if not isinstance(message, SystemMessage)
    ]
    messages: List[BaseMessage] = [system_message] + prior_messages
    assessment = await tool_aware.ainvoke(messages)

    payload = assessment.model_dump()
    logger.info("Analyst assessment: %s", payload)

    if state.get("loop_count", 0) >= 3 and assessment.needs_more_research:
        synthesis_prompt = "\n\n".join(
            system_parts
            + [
                "Provide a final synthesis using only the available information. "
                "Do not ask for more research.",
            ]
        )
        synthesis_message = SystemMessage(content=synthesis_prompt)
        synthesis_messages: List[BaseMessage] = [synthesis_message] + prior_messages
        synthesis_response = await tool_aware.ainvoke(synthesis_messages)
        assessment = AnalystAssessment(
            needs_more_research=False,
            gaps=assessment.gaps,
            re_research_instructions="",
            synthesis=str(synthesis_response.synthesis),
        )
        payload = assessment.model_dump()
        logger.info("Analyst forced final synthesis after loop limit")

    content = json.dumps(payload)
    response_messages = [AIMessage(content=content)]
    if assessment.needs_more_research and assessment.re_research_instructions:
        response_messages.append(
            AIMessage(
                content=(
                    "Re-research instructions:\n"
                    f"{assessment.re_research_instructions}"
                )
            )
        )

    return {
        "messages": response_messages,
        "summary": state.get("summary", ""),
        "research_results": state.get("research_results", []),
        "needs_more_research": assessment.needs_more_research,
        "loop_count": state.get("loop_count", 0) + 1,
    }
