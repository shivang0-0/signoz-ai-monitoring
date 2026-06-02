# SigNoz AI Monitoring

FastAPI service for AI monitoring pipelines.

## Setup

```bash
# Create and activate a Python 3.13 venv
python3.13 -m venv .venv
source .venv/bin/activate

# Install the project and dev dependencies
pip install -e ".[dev]"
```

## Run

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
main.py          # FastAPI app entry point
routers/         # API endpoint definitions
models/          # SQLAlchemy database models
schemas/         # Pydantic validation schemas
services/        # Core business logic
clients/         # External API integrations
telemetry/       # OpenTelemetry instrumentation
tests/
  api/           # Endpoint tests (TestClient)
  services/      # Service unit tests
  architecture/  # Module boundary tests
```
