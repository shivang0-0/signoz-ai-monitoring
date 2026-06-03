"""MCP tool registration for the AI Monitoring server."""

from __future__ import annotations

import structlog
from mcp.server.fastmcp import FastMCP

from clients.telemetry_mcp_client import telemetry_client
from services.query_perf_graph import run_query_perf_analysis

logger = structlog.get_logger(__name__)
mcp_server = FastMCP("ai-monitoring")


@mcp_server.tool()
async def analyze_slow_queries(service_name: str, time_range: str = "1h") -> str:
    """Analyze top slow database queries for a service with actionable recommendations."""
    log = logger.bind(service_name=service_name, time_range=time_range)
    log.info("analyze_slow_queries — tool invoked")

    result = await run_query_perf_analysis(
        service_name=service_name, time_range=time_range, client=telemetry_client
    )

    log.info("analyze_slow_queries — tool completed")
    return result
