"""LangGraph workflow for analyzing slow database queries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from clients.llm_client import get_llm
from clients.telemetry_mcp_client import TelemetryMCPClient, telemetry_client
from schemas.graph_state import QueryPerfRow, QueryPerfState

logger = structlog.get_logger(__name__)
TOP_N = 5

# Deterministic prompt: positive structure only, no negative examples or placeholder syntax
ANALYSIS_PROMPT = """\
You are a database performance engineer. Below are the {n} slowest database queries \
from the service `{service}` in the last `{time_range}`. Produce a structured report \
using ONLY the data provided.

## Report Structure

Produce three sections in this exact order:

### Section 1: Query Analysis Table

Fill this table with one row per query. Column meanings:
- SNo: 1, 2, 3... sequentially.
- DB Name/Type: Infer from the span name. If the span contains "postgresql", write \
"postgresql". If it contains "cassandra", write "cassandra". Otherwise use the span \
name prefix before the first dot.
- Table Name: Parse the SQL to extract the primary table. If no table name can be \
determined, write the literal word Unknown. This column MUST contain only a table \
name or the word Unknown — never write "None required", recommendations, or any \
other text here.
- Query: Copy the exact SQL text provided. Keep it verbatim — do not rewrite, \
shorten, or embellish it.
- Avg Time: Convert the provided duration to a human-readable form. Use "s" for \
seconds, "ms" for milliseconds.
- Severity: Assign using ONLY these absolute thresholds. Do NOT compare queries \
relative to each other. The thresholds are:
  * >= 10 seconds → Critical
  * >= 1 second → High
  * >= 100 milliseconds → Medium
  * < 100 milliseconds → Low

### Section 2: Recommendations Table

For each query, write one recommendation. Rules:
- Use the actual table and column names found in the provided SQL.
- If the query is a transaction-control statement \
(ROLLBACK, COMMIT, BEGIN, START TRANSACTION), write "None required".
- Never invent table names, column names, or index suggestions \
that do not appear in the provided SQL.
- If no meaningful optimization exists, write "None required".

IMPORTANT: The phrase "None required" is valid ONLY in the Recommendation column. \
Never use it in Table Name, SNo, DB Name/Type, Query, Avg Time, or Severity.

### Section 3: Summary

One sentence stating the slowest query, its duration, and the single most impactful fix.

## Formatting Rules

- Output raw Markdown. Do NOT wrap the response in triple backticks.
- Use exactly two Markdown tables with these exact headers:
  | SNo | DB Name/Type | Table Name | Query | Avg Time | Severity |
  | Query | Recommendation |
- Precede the first table with the heading "## Query Analysis Table".
- Precede the second table with the heading "## Recommendations Table".
- Precede the summary with the heading "## Summary".

## Example

Input:
### Query 1
- Span: postgresql.execute
- Avg Duration: 20.52s
- Call Count: 1
- Avg per Call: 20.52s
- SQL: DELETE FROM sessions WHERE id = ?

Output:
## Query Analysis Table

| SNo | DB Name/Type | Table Name | Query | Avg Time | Severity |
|-----|-------------|------------|-------|----------|----------|
| 1 | postgresql | sessions | DELETE FROM sessions WHERE id = ? | 20.52s | Critical |

## Recommendations Table

| Query | Recommendation |
|-------|---------------|
| DELETE FROM sessions WHERE id = ? | Add index on sessions(id) |

## Summary
The DELETE FROM sessions query takes 20.52s and would benefit most from an index on sessions(id).

## Input Data

{query_data}\
"""


def _ns_to_human(ns: float) -> str:
    """Convert nanoseconds to a human-readable duration string."""
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.2f}s"
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f}ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.2f}µs"
    return f"{ns:.0f}ns"


# SQL keyword patterns for table name extraction
_TABLE_PATTERNS = [
    (r"(?i)\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)", 1),
    (r"(?i)\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)", 1),
    (r"(?i)\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*)", 1),
    (r"(?i)\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)", 1),
    (r"(?i)\bTABLE\s+([a-zA-Z_][a-zA-Z0-9_]*)", 1),
]


def _extract_table_names(sql_text: str) -> set[str]:
    """Parse SQL text to extract table names using common keyword patterns."""
    import re

    names: set[str] = set()
    for pattern, group in _TABLE_PATTERNS:
        for match in re.finditer(pattern, sql_text):
            name = match.group(group).lower()
            # Filter out SQL keywords that aren't table names
            if name not in {"from", "where", "set", "values", "select", "table"}:
                names.add(name)
    return names


def _clean_llm_output(raw: str) -> str:
    """Strip markdown code fences and surrounding whitespace from LLM output."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    return cleaned.strip()


def _parse_table_names_from_output(output: str) -> set[str]:
    """Extract table names from the Query Analysis Table in the LLM output."""
    import re

    names: set[str] = set()
    # Match table rows: | N | db | table_name | ...
    for match in re.finditer(r"\|\s*\d+\s*\|\s*[^|]+\s*\|\s*([^|]+?)\s*\|", output, re.IGNORECASE):
        cell = match.group(1).strip()
        if cell and cell.lower() != "unknown":
            names.add(cell.lower())
    return names


def _validate_output(output: str, known_tables: set[str]) -> list[str]:
    """Validate LLM output against known input data. Returns list of error messages."""
    import re

    errors: list[str] = []

    # Check 1: Forbidden angle-bracket placeholder syntax
    if re.search(r"<[a-zA-Z_]+>", output, re.IGNORECASE):
        errors.append(
            "Contains angle-bracket placeholders (e.g. <table>, <query>). "
            "Replace with 'Unknown' or the actual value from the input data."
        )

    # Check 2: Cross-reference table names against input SQL
    output_tables = _parse_table_names_from_output(output)
    if known_tables:
        invented = output_tables - known_tables
        if invented:
            errors.append(
                f"Invented table names not present in input SQL: {', '.join(sorted(invented))}. "
                f"Valid tables from input: {', '.join(sorted(known_tables))}. "
                "Replace invented names with actual ones or 'Unknown'."
            )

    # Check 3: Required sections present
    for section in ["Query Analysis Table", "Recommendations Table", "Summary"]:
        if f"## {section}" not in output:
            errors.append(f"Missing required section heading: '## {section}'")

    # Check 4: Severity must use only valid values
    sev_section = (
        output.split("## Recommendations")[0] if "## Recommendations" in output else output
    )
    severities = re.findall(r"\|\s*(Low|Medium|High|Critical)\s*\|", sev_section, re.IGNORECASE)
    allowed = {"low", "medium", "high", "critical"}
    for sev in severities:
        if sev.lower() not in allowed:
            errors.append(f"Invalid severity value: '{sev}'. Must be Low/Medium/High/Critical.")

    # Check 5: "None required" must only appear in the Recommendation column
    analysis_section = output.split("## Recommendations")[0]
    nn_rows = re.findall(
        r"\|\s*\d+\s*\|\s*[^|]+\s*\|\s*([^|]+?)\s*\|", analysis_section, re.IGNORECASE
    )
    for cell in nn_rows:
        cell_stripped = cell.strip()
        if "none required" in cell_stripped.lower() and cell_stripped.lower() != "unknown":
            errors.append(
                f"'None required' leaked into Table Name column: '{cell_stripped}'. "
                "'None required' is only valid in the Recommendation column."
            )
            break

    # Check 6: Severity must match absolute thresholds relative to Avg Time
    for row_match in re.finditer(
        r"\|\s*\d+\s*\|\s*[^|]+\s*\|\s*[^|]+\s*\|\s*[^|]+\s*\|\s*([\d.]+)\s*([a-zµ]+)\s*\|\s*(Low|Medium|High|Critical)\s*\|",
        analysis_section,
        re.IGNORECASE,
    ):
        value = float(row_match.group(1))
        unit = row_match.group(2).lower()
        declared_sev = row_match.group(3).lower()
        # Convert to seconds
        duration_s: float
        if unit == "s":
            duration_s = value
        elif unit == "ms":
            duration_s = value / 1000
        elif unit in ("µs", "us"):
            duration_s = value / 1_000_000
        elif unit == "ns":
            duration_s = value / 1_000_000_000
        else:
            continue

        expected = "low"
        if duration_s >= 10:
            expected = "critical"
        elif duration_s >= 1:
            expected = "high"
        elif duration_s >= 0.1:
            expected = "medium"

        if declared_sev != expected:
            errors.append(
                f"Severity mismatch: {value}{unit} ({duration_s}s) should be "
                f"'{expected.title()}' but was marked '{declared_sev.title()}'. "
                "Thresholds: >=10s=Critical, >=1s=High, >=100ms=Medium, <100ms=Low."
            )
            break

    return errors


def _build_correction_prompt(errors: list[str], known_tables: set[str]) -> str:
    """Build a one-shot correction message pointing out specific validation errors."""
    lines: list[str] = ["Your previous response had the following issues:"]
    for err in errors:
        lines.append(f"- {err}")
    lines.append("")
    lines.append("Please regenerate the full report with these issues fixed.")
    if known_tables:
        lines.append(
            f"Valid table names from the input data: {', '.join(sorted(known_tables))}. "
            "Use only these names or 'Unknown'."
        )
    return "\n".join(lines)


def _format_correction_note(errors: list[str]) -> str:
    """Format a visible correction note appended to output when validation fails."""
    return "\n\n> **System Correction**: " + " | ".join(errors)


def _build_query_perf_graph(
    client: TelemetryMCPClient | None = None,
    llm_factory: Callable[[], BaseChatModel] | None = None,
) -> StateGraph:  # type: ignore[type-arg]
    """Construct and compile the query performance analysis LangGraph."""
    _client = client if client is not None else telemetry_client
    _llm_factory = llm_factory if llm_factory is not None else get_llm

    async def node_aggregate_queries(state: QueryPerfState) -> dict[str, object]:
        """Node 1: Fetch avg duration + call count, merge, sort by avg/call ratio."""
        service = state["service_name"]
        time_range = state.get("time_range", "1h")
        log = logger.bind(service_name=service, time_range=time_range)
        log.info("Node 1: QueryAggregator — running dual aggregation")

        try:
            # Fetch top N by average duration
            avg_rows = await _client.aggregate_traces(
                aggregation="avg",
                service=service,
                group_by="db.query.text,name",
                time_range=time_range,
                limit=TOP_N,
                error=False,
                aggregate_on="durationNano",
                filter_expr="db.system EXISTS",
            )
            # Fetch count aggregation to compute call volume
            count_rows = await _client.aggregate_traces(
                aggregation="count",
                service=service,
                group_by="db.query.text,name",
                time_range=time_range,
                limit=TOP_N,
                error=False,
                filter_expr="db.system EXISTS",
            )
        except Exception as exc:
            log.error("Node 1: QueryAggregator — aggregation failed", error=str(exc))
            return {
                "final_rca": (
                    "## Query Performance Report\n\n"
                    f"**Error**: Aggregation failed.\n\n```\n{exc}\n```"
                )
            }

        # Parse average rows: key=(query_text, span_name) -> avg_duration_ns
        avg_map: dict[tuple[str, str], float] = {}
        for row in avg_rows:
            if isinstance(row, list) and len(row) >= 3:
                key = (str(row[0]) if row[0] else "", str(row[1]).strip() if row[1] else "")
                val = float(row[2]) if row[2] else 0.0
                if key[0] or key[1]:
                    avg_map[key] = val

        # Parse count rows: key=(query_text, span_name) -> call_count
        count_map: dict[tuple[str, str], int] = {}
        for row in count_rows:
            if isinstance(row, list) and len(row) >= 3:
                key = (str(row[0]) if row[0] else "", str(row[1]).strip() if row[1] else "")
                val = int(float(row[2])) if row[2] else 1
                if key[0] or key[1]:
                    count_map[key] = val

        # Merge: build QueryPerfRow with ratio = avg_duration_ns / call_count
        spans: list[QueryPerfRow] = []
        for key, avg_ns in avg_map.items():
            qtext, sname = key
            calls = count_map.get(key, 1)
            spans.append(
                {
                    "query_text": qtext or "(parameterized)",
                    "span_name": sname,
                    "avg_duration_ns": avg_ns,
                    "call_count": calls,
                }
            )

        # Sort by ratio descending (highest avg-per-call first)
        spans.sort(key=lambda s: s["avg_duration_ns"] / max(s["call_count"], 1), reverse=True)
        spans = spans[:TOP_N]

        log.info(
            "Node 1: QueryAggregator — dual aggregation complete",
            span_count=len(spans),
            avg_rows=len(avg_rows),
            count_rows=len(count_rows),
        )

        if not spans:
            return {
                "final_rca": (
                    f"## Query Performance Report\n\n"
                    f"**No database spans found** for `{service}` in `{time_range}`.\n\n"
                    f"Filter: `db.system EXISTS` | "
                    f"Avg rows: {len(avg_rows)}, Count rows: {len(count_rows)}\n\n"
                    f"Possible: no db.system instrumentation, casing mismatch, or no traffic."
                )
            }
        return {"query_spans": spans}

    async def node_analyze_queries(state: QueryPerfState) -> dict[str, object]:
        """Node 2: Format query data, invoke LLM, validate output, retry on hallucination."""
        spans = state.get("query_spans", [])
        service = state["service_name"]
        time_range = state.get("time_range", "1h")
        log = logger.bind(service_name=service)
        log.info("Node 2: QueryAnalyzer — calling LLM", count=len(spans))

        # Build query data block and extract known table names from input SQL
        lines: list[str] = []
        known_tables: set[str] = set()
        for i, s in enumerate(spans, 1):
            dur = _ns_to_human(s["avg_duration_ns"])
            ratio = _ns_to_human(s["avg_duration_ns"] / max(s["call_count"], 1))
            lines.extend(
                [
                    f"### Query {i}",
                    f"- Span: {s['span_name']}",
                    f"- Avg Duration: {dur}",
                    f"- Call Count: {s['call_count']}",
                    f"- Avg per Call: {ratio}",
                    f"- SQL: {s['query_text']}",
                    "",
                ]
            )
            known_tables |= _extract_table_names(s["query_text"])

        query_data = "\n".join(lines)
        prompt = ANALYSIS_PROMPT.format(
            n=len(spans), service=service, time_range=time_range, query_data=query_data
        )
        messages = [SystemMessage(content=prompt)]

        llm = _llm_factory()
        response = await llm.ainvoke(messages)
        rca = _clean_llm_output(str(response.content))
        log.info(
            "Node 2: QueryAnalyzer — initial response",
            output_chars=len(rca),
            known_tables=sorted(known_tables),
        )

        # Validation sensor: cross-reference output table names against input SQL
        validation_errors = _validate_output(rca, known_tables)
        if validation_errors:
            log.warning(
                "Node 2: QueryAnalyzer — validation failed, retrying with correction",
                errors=validation_errors,
            )
            correction = _build_correction_prompt(validation_errors, known_tables)
            retry_messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=correction),
            ]
            retry_response = await llm.ainvoke(retry_messages)
            rca = _clean_llm_output(str(retry_response.content))
            retry_errors = _validate_output(rca, known_tables)
            if retry_errors:
                log.warning(
                    "Node 2: QueryAnalyzer — retry also failed validation, "
                    "appending correction note",
                    errors=retry_errors,
                )
                rca += _format_correction_note(retry_errors)

        log.info("Node 2: QueryAnalyzer — analysis complete", output_chars=len(rca))
        return {"final_rca": rca}

    def route_after_aggregate(state: QueryPerfState) -> Literal["node_analyze_queries", "__end__"]:
        """Conditional edge: route to END if aggregation failed or found no spans."""
        return "__end__" if state.get("final_rca") else "node_analyze_queries"

    # Initialize and configure the graph structure
    graph = StateGraph(QueryPerfState)
    graph.add_node("node_aggregate_queries", node_aggregate_queries)
    graph.add_node("node_analyze_queries", node_analyze_queries)
    graph.set_entry_point("node_aggregate_queries")
    graph.add_conditional_edges(
        "node_aggregate_queries",
        route_after_aggregate,
        {"node_analyze_queries": "node_analyze_queries", "__end__": END},
    )
    graph.add_edge("node_analyze_queries", END)

    return graph


async def run_query_perf_analysis(
    service_name: str,
    time_range: str = "1h",
    client: TelemetryMCPClient | None = None,
    llm_factory: Callable[[], BaseChatModel] | None = None,
) -> str:
    """Execute the query performance analysis pipeline for a given service."""
    log = logger.bind(service_name=service_name, time_range=time_range)
    log.info("Query Perf Pipeline — starting analysis")

    graph = _build_query_perf_graph(client=client, llm_factory=llm_factory).compile()
    initial_state: QueryPerfState = {"service_name": service_name, "time_range": time_range}
    final_state = await graph.ainvoke(initial_state)

    log.info("Query Perf Pipeline — analysis complete")
    return str(final_state.get("final_rca", "No query analysis produced."))
