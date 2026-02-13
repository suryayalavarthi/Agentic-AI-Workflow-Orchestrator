from __future__ import annotations

import json
import logging
from typing import List, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from pydantic import BaseModel, Field

from ..config import get_settings, llm_retry
from ..state import AgentState, prune_messages

logger = logging.getLogger(__name__)


SUPERVISOR_SYSTEM = """You are the Supervisor agent.
You are a coordinator. You NEVER use tools like web_scraper or duckduckgo_search
yourself. If research or scraping is needed, you MUST set next_agent to
"researcher". The Researcher is the ONLY agent allowed to use tools.
You route tasks to specialists and keep the session on track.
If tool output is missing or ambiguous, state that you lack information.
Decide the next agent to call and explain why.
"""


class SupervisorDecision(BaseModel):
    next_agent: Literal["researcher", "analyst", "final_report"] = Field(
        ...,
        description="Next agent to run.",
    )
    reasoning: str = Field(..., description="Short routing rationale.")


def _build_llm() -> ChatAnthropic:
    cfg = get_settings()
    return ChatAnthropic(
        model=cfg.default_model,
        temperature=cfg.default_temperature,
        api_key=cfg.anthropic_api_key,
    )


def supervisor_node(state: AgentState) -> AgentState:
    llm = _build_llm().with_structured_output(SupervisorDecision)
    system_parts = [SUPERVISOR_SYSTEM]
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

    @llm_retry()
    def _invoke(msgs):
        return llm.invoke(msgs)

    decision = _invoke(messages)

    if state.get("messages"):
        last_message = state["messages"][-1]
        try:
            payload = json.loads(last_message.content)
            if (
                isinstance(payload, dict)
                and payload.get("needs_more_research") is False
                and payload.get("synthesis")
            ):
                decision = SupervisorDecision(
                    next_agent="final_report",
                    reasoning="Analyst completed synthesis; proceed to final report.",
                )
        except (TypeError, json.JSONDecodeError):
            pass

    payload = decision.model_dump()
    logger.info("Supervisor decision: %s", payload)

    return {
        "messages": [AIMessage(content=json.dumps(payload))],
        "summary": state.get("summary", ""),
        "research_results": state.get("research_results", []),
        "needs_more_research": state.get("needs_more_research", False),
        "loop_count": state.get("loop_count", 0) + 1,
    }
