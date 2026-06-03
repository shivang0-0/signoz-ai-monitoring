"""LangGraph state definitions for the query performance analysis pipeline."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class QueryPerfRow(TypedDict):
    """Represents a single aggregated database query row."""

    query_text: str
    span_name: str
    avg_duration_ns: float
    call_count: int


class QueryPerfState(TypedDict):
    """State passed between nodes in the query performance analysis graph."""

    service_name: str
    time_range: str
    query_spans: NotRequired[list[QueryPerfRow]]
    final_rca: NotRequired[str]
