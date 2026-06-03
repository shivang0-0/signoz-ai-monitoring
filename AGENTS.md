# AGENTS.md

<!-- 
REPO-LOCAL INSTRUCTIONS
This file acts as the primary "harness" for the agent. It shapes the environment around the AI agent so it can work reliably. By placing this directly in the repository, it prevents context drift across long-running coding sessions.
-->

## 1. Project Topology & Bootstrapping

<!-- 
FEEDFORWARD GUIDES (INFERENTIAL & COMPUTATIONAL)
These instructions anticipate the agent's behavior and steer it *before* it acts, increasing the probability of a good result on the first attempt. 
Defining a strict topology narrows the output space, making the agent much easier to regulate.
-->

**Stack:** FastAPI (Python 3.13) with SigNoz for observability (traces, metrics, logs via OpenTelemetry).

**Architecture:** You must strictly adhere to the following module boundaries:
- `routers/`: All API endpoint definitions.
- `models/`: Database models (SQLAlchemy).
- `schemas/`: Pydantic validation schemas.
- `services/`: Core business logic — AI monitoring pipelines, span ingestion, anomaly detection.
- `clients/`: External API integrations (SigNoz query API, LLM provider APIs).
- `telemetry/`: OpenTelemetry instrumentation — tracers, meters, exporters.

**Bootstrapping:** To test your changes locally, always run the server using:
```
uvicorn main:app --reload
```

## 2. Maintainability & Code Quality Rules

<!-- 
MAINTAINABILITY HARNESS
Regulates internal code quality. Explicit rules about types and complexity act as inferential feedforward guidance.
-->

- **Strict Typing:** Never bypass Pydantic validations or Python type hints. `Any` is strictly forbidden. All function signatures, return types, and class attributes must be fully typed.
- **Simplicity:** Avoid over-engineering. If a function exceeds 50 lines, refactor it into smaller, composable helper functions.
- **Domain Language:** Use SigNoz-native terminology consistently: spans, traces, metrics, exemplars, resources, attributes. Do not invent your own terms for existing OpenTelemetry concepts.

## 3. Observability Standards

<!-- 
ARCHITECTURE FITNESS HARNESS
Defines and checks the architecture characteristics — performance and observability standards.
-->

- **Structured Logging:** Use `structlog` for all logging. Every log entry must include `trace_id` and `span_id` for correlation with SigNoz traces.
- **Distributed Tracing:** Every endpoint handler must create or propagate a span. Use the OpenTelemetry instrumentation middleware — never start raw spans manually unless instrumenting custom business logic.
- **Metrics:** Expose counters and histograms via `/metrics` (Prometheus format) for request duration, error rates, and AI-specific metrics (token usage, latency, model calls).
- **Reflection:** After creating a new endpoint, reflect on whether its telemetry would be debuggable in SigNoz. Can someone trace a request end-to-end? Can they filter by model, status, or tenant?

## 4. Verification & Testing 

<!-- 
FEEDBACK CONTROLS (COMPUTATIONAL SENSORS)
Tests, linters, and type-checkers are "computational sensors" — deterministic, fast tools that catch structural drift and style violations before human review.
-->

Before finalizing any task, you must run these computational sensors in order:
1. **Formatting:** `ruff format .`
2. **Linting:** `ruff check . --fix`
3. **Type Checking:** `mypy .`
4. **Structural Tests:** `pytest tests/architecture/` — verifies module boundaries are not violated.
5. **Unit/Integration Tests:** `pytest tests/ -x --cov`

If any sensor fails, fix the failures before presenting your work. Do not bypass this gate.

## 5. Functional Behavior Checks

<!-- 
BEHAVIOUR HARNESS & INFERENTIAL SENSORS
Regulates whether the application functionally behaves as intended. Uses both computational sensors (tests) and inferential sensors (semantic judgment).
-->

- **Test Coverage:** All new endpoints must have functional tests in `tests/api/`. All new service functions must have unit tests in `tests/services/`.
- **Telemetry Tests:** Endpoints involving span creation or metric emission must have tests asserting the correct telemetry is produced.
- **Inferential Check:** After generating tests, use your semantic judgment to review whether the suite covers edge cases — not just the happy path. If a test is redundant, brute-forced, or semantically duplicated, rewrite it. Specifically check:
  - Missing tenant/isolation boundary
  - Empty payloads, missing required fields, malformed JSON
  - Race conditions or concurrent access patterns
  - Timeout and upstream failure scenarios

## 6. Context Engineering & Memory Management

<!-- 
CONTEXT ENGINEERING & BACKPRESSURE
Rules for managing working memory budget, preventing context window burn, and enabling safe pause/resume of long sessions.
-->

- **State Tracking:** If you encounter a failing test loop that you cannot solve after 3 attempts, stop iterating. Write your current state, useful failure outputs, and hypotheses into `DEBUG_LOG.md`. Then pause and ask for help. Do not burn through tokens endlessly.
- **Incremental Checkpointing:** For multi-step features, commit working intermediate states with descriptive messages. This lets sessions be resumed without losing context.

## 7. Testing Discipline

<!--
TESTING REQUIREMENT
Every new service addition must be accompanied by tests. This is non-negotiable.
-->

- **Always Test New Services:** For each new service module added to `services/`, write corresponding unit tests in `tests/services/`. The test file should cover the happy path, error paths, edge cases, and verify that external dependencies are called with correct parameters.
- **Mock External Dependencies:** Services that depend on external clients (SigNoz MCP, LLM providers) must be testable by injecting mock clients — do not require live external connections in unit tests.

