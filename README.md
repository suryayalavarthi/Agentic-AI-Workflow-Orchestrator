# 🤖 Agentic AI Workflow Orchestrator
### Production-Grade Multi-Agent RAG with LangGraph, MCP, & ChromaDB

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange)](https://github.com/langchain-ai/langgraph)
[![MCP](https://img.shields.io/badge/Protocol-MCP-green)](https://modelcontextprotocol.io/)
[![ChromaDB](https://img.shields.io/badge/Memory-ChromaDB-lightgrey)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)

A stateful, multi-agent research system designed to automate complex information gathering and synthesis with **hallucination resistance** and **factual grounding**. This orchestrator employs cyclic reasoning to research, analyze, and verify data, producing high-precision reports backed by a local **ChromaDB** vector store.

---

## � High-Level Value Proposition

### 🛡️ Factual Grounding & Hallucination Resistance
Unlike standard LLM chats, this orchestrator never "guesses." If the **Analyst** identifies a gap in the gathered data, it forces the **Researcher** back into the field. All synthesis is cross-referenced against raw tool outputs and vector DB sources.

### 🔌 Standardized Tooling via MCP
Leveraging the **Model Context Protocol (MCP)**, the system interacts with external tools (Search, Scrapers, Filesystems) through a standardized interface, ensuring decoupled, scalable, and reliable tool execution.

---

## 🏗️ System Architecture

The orchestrator utilizes a **Supervisor-Worker** pattern with integrated context compression and recursive guardrails.

```mermaid
flowchart TD
    User([User Prompt]) --> GUI[Streamlit UI]
    GUI --> Graph[LangGraph Orchestrator]
    
    subgraph "Orchestration Loop"
        Supervisor{Supervisor Node}
        Summarizer[Summarizer Node]
        Researcher[Researcher Agent]
        Analyst[Analyst Agent]
        Final[Final Report Node]
    end

    subgraph "Knowledge Stack"
        MCP[FastMCP: Brave/Search/Scraper]
        VectorDB[(ChromaDB)]
    end

    Supervisor -->|eval & route| Researcher
    Researcher -->|Standardized Tools| MCP
    MCP -->|Raw Data| Researcher
    Researcher -->|Persistence| VectorDB

    Researcher --> Analyst
    Analyst -->|Check Memory| VectorDB
    Analyst -->|Gap Detected| Supervisor
    Analyst -->|Sufficient Data| Final

    Final -->|Complete| END([End Session])
    
    Supervisor -.->|Trigger if Context > 6 Msg| Summarizer
    Summarizer -.->|Compressed Context| Supervisor
```

---

## 🧠 Technical Deep Dive

### 🧼 State Hygiene
To prevent "topic bleed" and stale reasoning, the system implements strict state cleanup at the start of every user turn:
- **`messages` Reset**: The message history is cleared for the active turn, keeping only the new query.
- **`research_results` Flush**: Per-turn research artifacts are cleared to ensure the final report reflects the current query's findings.
- **`loop_count` Reset**: Tracking resets to zero to allow a full reasoning cycle for new tasks.

### 🛡️ Recursive Guardrails
Agentic loops can become expensive or infinite. We implement two layers of protection:
1. **Application Layer (`loop_count`)**: The `Supervisor` and `Analyst` track iterations; if research exceeds 15 steps, the system forces a "Best Effort" synthesis to the `Final Report` node.
2. **Infrastructure Layer (`recursion_limit`)**: LangGraph is configured with a hard `recursion_limit=25`. If the graph exceeds this threshold, execution halts to prevent API rate-limit exhaustion.

### 📉 Context Compression
The **Summarizer Node** monitors token pressure. When the message history exceeds 6 entries, it condenses the historical metadata into a `summary` field.
- **Active HumanMessage Preservation**: The summarizer is specifically tuned to purge system metadata while preserving the *last user message*, ensuring the Supervisor never loses sight of the current objective.

---

## 🛠️ The Knowledge Stack

- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) (Stateful, cyclic graph execution).
- **Intelligence**: Anthropic Claude 3.5 Sonnet / Gemini 1.5 Pro.
- **Memory**: [ChromaDB](https://www.trychroma.com/) (Vector store for source-grounded retrieval).
- **Protocol**: [FastMCP](https://github.com/jlowin/fastmcp) (Standardized tool calling interface).

---

## 📂 Project Structure

```text
src/
├── agents/             # Specialist logic (Supervisor, Researcher, Analyst)
├── graph/
│   ├── workflow.py     # Graph definition & conditional routing
│   └── nodes.py        # Shared node implementations (Summarizer, Final Report)
├── mcp_logic/          # MCP Client & Server implementations
├── tools/              # RAG & specialized toolsets (ChromaDB, Search)
├── gui.py              # Streamlit dashboard
├── main.py             # CLI entry point
└── state.py            # TypedDict state definitions
```

---

## 🚀 Installation & Usage

### 1️⃣ Prerequisites
- Python 3.10+
- Anthropic or OpenAI API Key
- [Optional] LangSmith API Key for tracing

### 2️⃣ Setup
```bash
# Clone the repository
git clone <repo-url>
cd agentic-orchestrator

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Configuration
Create a `.env` file in the root directory:
```env
ANTHROPIC_API_KEY=your_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
```

### 4️⃣ Execution
**Launch the Dashboard (Recommended):**
```bash
streamlit run src/gui.py
```

**Launch the CLI:**
```bash
python -m src.main
```

---
*Built with precision by the Agentic Workflow Team.*