# Agentic Orchestrator Model Card

## Model Details
- **Model name**: Agentic AI Workflow Orchestrator
- **Architecture**: Multi-Agent RAG System (LangGraph)
- **Primary Agents**: Claude 3.5 Sonnet / Gemini 1.5 Pro (Configurable)
- **Version**: 1.0.0
- **Date released**: 2026-02-06
- **Intended use case**: Technical research, document synthesis, and architectural analysis.

## System Overview
- **Orchestration**: Built on **LangGraph** for stateful, cyclic task execution.
- **Supervisor Role**: Routes tasks to specialized workers (Researcher/Analyst) based on intent.
- **Researcher Role**: Leverages **MCP (Model Context Protocol)** to gather live data via Brave Search and Web Scrapers.
- **Analyst Role**: Evaluates research output, identifies gaps, and requests further research if data is insufficient.
- **Memory**: Utilizes **ChromaDB** for vector-grounded retrieval and long-term research persistence.
- **Summarizer Role**: Condenses conversation history while preserving the active HumanMessage context.

## Hallucination Resistance
- **Supervisor-Analyst-Researcher Loop**: The system implements a mandatory review loop. The Analyst must explicitly confirm that research is sufficient before moving to final synthesis.
- **Memory Grounding**: Every final response is cross-referenced with sources stored in ChromaDB, ensuring all claims are backed by retrieved data.

## Data Sources
- **Primary sources**: Real-time web search and official technical documentation.
- **MCP tools used**: DuckDuckGo/Brave Search, Web Scraper, Filesystem access.
- **Data refresh cadence**: Real-time per query.

## Performance
- **Evaluation criteria**: Factuality (RAG grounding), Synthesis coherence, and Tool-calling accuracy.
- **Known strengths**: Deep technical research, multi-step reasoning, standardized tool interface.
- **Known weaknesses**: Latency scales with query complexity; limited by LLM context window (mitigated by summarization).

## Safety & Bias
- **Mitigations**: Hard loop limits (15 steps) and infrastructure-level recursion limits (25 steps) to prevent accidental infinite API calls and rate-limiting.

## Human Oversight
- **Human-in-the-loop**: The system provides transparency through a real-time "Agent Thought Process" log in the GUI, allowing users to watch routing decisions.
- **Audit trail**: Persistent graph state via LangGraph Checkpointers allows for turn-by-turn debugging.

## Update & Maintenance
- **Owner**: Surya Yalavarthi
- **Monitoring signals**: Loop counts, tool-failure rates, and RAG retrieval hitting scores.
