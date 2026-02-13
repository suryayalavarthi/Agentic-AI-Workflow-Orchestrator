## Contributing Guide

Thank you for your interest in improving the **Agentic AI Workflow Orchestrator**.

### Local setup

```bash
git clone <your-fork-url>
cd agentic-orchestrator

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file from `.env.example` and set at least:

- `ANTHROPIC_API_KEY`

### Running the app

In one terminal:

```bash
python -m src.server
```

In another terminal:

```bash
export PYTHONPATH="$PWD"
python -m streamlit run src/gui.py
```

### Tests & linting

```bash
pytest tests/ -v
ruff check src tests
```

### Pull requests

- Keep changes focused and well-described.
- Add or update tests when changing behavior.
- Update the README if you modify key flows or setup.

