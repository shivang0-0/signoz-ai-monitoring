# SigNoz AI Monitoring

FastAPI + MCP server for AI monitoring pipelines with automated Root Cause Analysis.

## Setup

```bash
# Create and activate a Python 3.13 venv
python3.13 -m venv .venv
source .venv/bin/activate

# Install the project and dev dependencies
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file at the project root:

```env
LLM_MODEL=qwen2.5:0.5b
LLM_BASE_URL=http://127.0.0.1:11434/v1

SIGNOZ_URL=https://your-signoz-instance.com
SIGNOZ_API_KEY=your_signoz_api_key
LOG_LEVEL=info
```

`LLM_MODEL` defaults to `qwen2.5:0.5b` (any Ollama model works).  
`LLM_BASE_URL` defaults to `http://127.0.0.1:11434/v1` (local Ollama).  
`SIGNOZ_URL` and `SIGNOZ_API_KEY` are required for telemetry data access.  
`LOG_LEVEL` controls both server and SigNoz MCP log verbosity.

## Services

### DB Performance Diagnostics MCP Server

Exposes a `diagnose_db_performance` tool for analyzing slow database queries.

**What it does:** Takes a SigNoz service name and runs a 4-node LangGraph pipeline:
1. Aggregates DB spans (db.system EXISTS) grouped by query text and span name
2. Analyzes raw SQL for anti-patterns via local LLM (Ollama)
3. Formats the analysis into a structured table with priorities
4. Provisions a saved Explorer view in SigNoz scoped to the service

**Run as MCP server (streamable-http — visible logs in terminal):**

```bash
python mcp_server.py
```

This starts the server on `http://127.0.0.1:9009` (override with `MCP_HOST` and `MCP_PORT` env vars).
The `signoz-mcp-server` binary spawns as a stdio subprocess — all logs from both servers appear in your terminal.

Add to your agent's MCP list:

```bash
cmd mcp add --scope user --transport http sigNoz-ai-monitoring \
  http://127.0.0.1:9009/mcp
```

No `--env` flags needed — the server already reads `.env` from its own working directory.

Place the `signoz-mcp-server` binary at the project root.

**Tool signature:**

```
diagnose_db_performance(service_name: str, time_range: str = "1h") -> str
```

Returns a formatted table with query breakdowns and prioritized recommendations.

```bash
uvicorn main:app --reload
```

The server starts at `http://127.0.0.1:9009`. Hit `/health` to verify it's running:

```bash
curl http://127.0.0.1:9009/health
# {"status":"ok"}
```

## Quality Checks

All checks must pass before committing. Run them in order:

```bash
ruff format .          # formatting
ruff check . --fix     # linting
mypy .                 # type checking
pytest tests/ -x --cov # all tests with coverage
```

## Project Structure

```
main.py                # FastAPI app entry point
mcp_server.py          # MCP server entry point (stdio)
routers/               # API endpoint definitions + MCP tool registration
models/                # SQLAlchemy database models
schemas/               # Pydantic validation schemas + LangGraph state models
services/              # Core business logic (RCA graph pipeline)
clients/               # External API integrations (SigNoz MCP, Command Code LLM)
telemetry/             # OpenTelemetry instrumentation
tests/
  api/                 # Endpoint tests (TestClient)
  services/            # Service unit tests
  architecture/        # Module boundary tests
```
