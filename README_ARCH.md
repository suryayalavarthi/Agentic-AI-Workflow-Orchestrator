# System Blueprint: suryayalavarthi/Agentic-AI-Workflow-Orchestrator

> This orchestrator uses LangGraph and MCP for stateful, multi-agent research. A Supervisor routes tasks to Researchers and Analysts who store and retrieve data via ChromaDB RAG. It features cyclic refinement and context summarization to produce hallucination-resistant reports with full source traceability.
>
> Auto-generated on 2026-02-13 by Repo-to-Blueprint Architect

## Project Purpose
This is a multi-agent research orchestration system that uses LangGraph to coordinate Supervisor, Researcher, and Analyst agents. The system performs iterative research tasks with RAG-based memory (ChromaDB), produces synthesized reports with source traceability, and exposes both CLI and Streamlit interfaces.

## Technical Stack
- **Language**: Python 3.12
- **Framework**: LangGraph (stateful agent orchestration), LangChain (LLM abstractions), Streamlit (web UI)
- **Key Dependencies**:
  - Agent/LLM: `langgraph`, `langchain-core`, `langchain-anthropic`, `langchain-community`, `langchain-mcp-adapters`
  - MCP: `mcp`, `fastmcp`
  - RAG/Embeddings: `chromadb`, `sentence-transformers`
  - Tools: `duckduckgo-search`, `requests`, `beautifulsoup4`
  - Config: `python-dotenv`, `pydantic`, `pydantic-settings`
  - Testing: `pytest`, `pytest-asyncio`
- **Infrastructure**: GitHub Actions CI (`.github/workflows/ci.yml` — linting with Ruff, unit tests with pytest)

## Architecture Blueprint

```mermaid
flowchart TD
    subgraph UI["Presentation Layer"]
        CLI["CLI Interface<br/>(main.py)"]
        GUI["Streamlit GUI<br/>(gui.py)"]
        SRV["FastAPI Server<br/>(server.py)"]
    end

    subgraph ORCH["Orchestration Layer"]
        WF["LangGraph Workflow<br/>(graph/workflow.py)"]
        NODES["Graph Nodes<br/>(graph/nodes.py)"]
        STATE["Agent State<br/>(state.py)"]
    end

    subgraph AGENTS["Agent Layer"]
        SUP["Supervisor Agent<br/>(agents/supervisor.py)"]
        RES["Researcher Agent<br/>(agents/researcher.py)"]
        ANA["Analyst Agent<br/>(agents/analyst.py)"]
    end

    subgraph TOOLS["Tool Layer"]
        REG["Tool Registry<br/>(tools/registry.py)"]
        MEM["Memory Tool<br/>(tools/memory.py)"]
        SRCH["Search Tool<br/>(tools/search.py)"]
        FS["Filesystem Tool<br/>(tools/filesystem.py)"]
        SQL["SQL Tool<br/>(tools/sql.py)"]
        MCPT["MCP Tools<br/>(tools/mcp_tools.py)"]
    end

    subgraph EXT["External Services"]
        MCP["MCP Client<br/>(mcp_logic/client.py)"]
        CHROMA[("ChromaDB<br/>Vector Store")]
        LLM["Anthropic LLM<br/>(langchain-anthropic)"]
    end

    CLI --> WF
    GUI --> WF
    SRV --> WF
    WF --> NODES
    NODES --> STATE
    NODES --> SUP
    NODES --> RES
    NODES --> ANA
    SUP --> REG
    RES --> REG
    ANA --> REG
    REG --> MEM
    REG --> SRCH
    REG --> FS
    REG --> SQL
    REG --> MCPT
    MEM --> CHROMA
    MCPT --> MCP
    SUP --> LLM
    RES --> LLM
    ANA --> LLM

    style CLI fill:#1f6feb,stroke:#58a6ff,color:#fff
    style GUI fill:#1f6feb,stroke:#58a6ff,color:#fff
    style SRV fill:#1f6feb,stroke:#58a6ff,color:#fff
    style WF fill:#238636,stroke:#3fb950,color:#fff
    style NODES fill:#238636,stroke:#3fb950,color:#fff
    style STATE fill:#238636,stroke:#3fb950,color:#fff
    style SUP fill:#238636,stroke:#3fb950,color:#fff
    style RES fill:#238636,stroke:#3fb950,color:#fff
    style ANA fill:#238636,stroke:#3fb950,color:#fff
    style REG fill:#238636,stroke:#3fb950,color:#fff
    style MEM fill:#238636,stroke:#3fb950,color:#fff
    style SRCH fill:#238636,stroke:#3fb950,color:#fff
    style FS fill:#238636,stroke:#3fb950,color:#fff
    style SQL fill:#238636,stroke:#3fb950,color:#fff
    style MCPT fill:#238636,stroke:#3fb950,color:#fff
    style CHROMA fill:#da3633,stroke:#f85149,color:#fff
    style MCP fill:#8b949e,stroke:#c9d1d9,color:#fff
    style LLM fill:#8b949e,stroke:#c9d1d9,color:#fff

```

## Request Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI as main.py
    participant WF as LangGraph Workflow
    participant SUP as Supervisor Agent
    participant RES as Researcher Agent
    participant ANA as Analyst Agent
    participant MEM as Memory Tool
    participant CHROMA as ChromaDB
    participant LLM as Anthropic LLM

    User->>CLI: Submit research query
    CLI->>WF: Initialize AgentState with HumanMessage
    WF->>SUP: Route to supervisor node
    SUP->>LLM: Generate task delegation plan
    LLM-->>SUP: Return routing decision (JSON)

    alt Route to Researcher
        SUP->>WF: Update state with "researcher" next
        WF->>RES: Execute researcher node
        RES->>MEM: store_memory(research_data)
        MEM->>CHROMA: Embed and persist vectors
        CHROMA-->>MEM: Confirm storage
        RES->>LLM: Generate research findings
        LLM-->>RES: Return research content
        RES->>WF: Update messages with findings
    end

    alt Route to Analyst
        SUP->>WF: Update state with "analyst" next
        WF->>ANA: Execute analyst node
        ANA->>MEM: retrieve_memory(query)
        MEM->>CHROMA: Vector similarity search
        CHROMA-->>MEM: Return relevant chunks
        ANA->>LLM: Synthesize with context
        LLM-->>ANA: Return analysis (JSON with needs_more_research flag)
        ANA->>WF: Update messages with synthesis
    end

    WF->>SUP: Check needs_more_research flag

    alt Needs refinement
        SUP->>WF: Loop back to researcher/analyst
    else Complete
        WF->>CLI: Return final AgentState
        CLI->>User: Display synthesis
    end

```

## Evidence-Based Risks

1. **Blocking I/O in async context** (`src/main.py:95-96`): `input()` call inside `async def run_cli()` blocks the event loop; should use `asyncio.to_thread(input, ...)` or aioconsole for non-blocking input.

2. **Missing API key validation** (`src/config.py` not shown, but `tests/test_server.py` and CI workflow use `ANTHROPIC_API_KEY: "test-key-ci"`): No evidence of startup validation for required API keys; runtime failures likely if keys missing.

3. **Unbounded recursion risk** (`src/main.py:86` sets `recursion_limit` from config, `src/graph/workflow.py` not shown): LangGraph cyclic refinement loop depends on `needs_more_research` flag in LLM JSON output; malformed LLM responses could cause infinite loops until recursion limit hit.

4. **ChromaDB persistence not configured** (`requirements.txt:19` includes `chromadb`, but no evidence of persistence path in `.env.example` or config): Default in-memory mode means all RAG data lost on restart; no durable storage configured.

5. **Test isolation issues** (`tests/conftest.py` not shown, but `pytest.ini` exists): 11 test files with shared state objects (AgentState, ChromaDB client) risk cross-test contamination without proper fixtures; `conftest.py` contents unknown.

---

## Repository Stats
| Metric | Value |
|--------|-------|
| Total Files | 44 |
| Total Directories | 9 |
| Generated | 2026-02-13 |
| Source | [suryayalavarthi/Agentic-AI-Workflow-Orchestrator](https://github.com/suryayalavarthi/Agentic-AI-Workflow-Orchestrator) |

---

*Generated by Repo-to-Blueprint Architect via n8n*
