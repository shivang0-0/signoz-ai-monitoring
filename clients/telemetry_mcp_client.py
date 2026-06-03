"""Client for interacting with the local telemetry MCP server."""

import json
import os
import sys
from typing import Any, cast

import structlog
from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent

load_dotenv()

logger = structlog.get_logger(__name__)

# Path to the local telemetry MCP binary (configurable via env var)
_TELEMETRY_BINARY = os.environ.get(
    "TELEMETRY_MCP_BINARY", os.path.join(os.path.dirname(__file__), "..", "signoz-mcp-server")
)


class TelemetryMCPClient:
    """Manages the lifecycle and communication with the telemetry MCP binary."""

    def __init__(self) -> None:
        """Configure the stdio parameters for the telemetry MCP subprocess."""
        self._server_params = StdioServerParameters(
            command=_TELEMETRY_BINARY,
            args=[],
            env={**os.environ, "LOG_LEVEL": os.environ.get("LOG_LEVEL", "info")},
        )
        self._ctx: Any = None
        self._session_cm: Any = None
        self._session: ClientSession | None = None

    async def start(self) -> None:
        """Initialize the stdio connection and MCP session."""
        logger.info("Telemetry MCP client starting", binary=_TELEMETRY_BINARY)
        ctx = stdio_client(self._server_params, errlog=sys.stderr)
        read, write = await ctx.__aenter__()
        session_cm = ClientSession(read, write)
        session = await session_cm.__aenter__()
        await session.initialize()

        self._ctx = ctx
        self._session_cm = session_cm
        self._session = session
        logger.info("Telemetry MCP client connected — session ready")

    async def stop(self) -> None:
        """Gracefully close the MCP session and stdio connection."""
        if self._session is not None:
            await self._session_cm.__aexit__(None, None, None)
            self._session = None
            self._session_cm = None
        if self._ctx is not None:
            await self._ctx.__aexit__(None, None, None)
            self._ctx = None
        logger.info("Telemetry MCP client stopped")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute an MCP tool and parse the JSON response, raising on errors."""
        if self._session is None:
            raise RuntimeError("Telemetry MCP client not started")

        result = await self._session.call_tool(tool_name, arguments)
        if result.isError:
            error_text = ""
            if result.content and isinstance(result.content[0], TextContent):
                error_text = result.content[0].text
            raise RuntimeError(f"Telemetry tool '{tool_name}' failed: {error_text[:500]}")

        content = result.content
        if content and isinstance(content[0], TextContent):
            return cast(dict[str, Any], json.loads(content[0].text))
        return {}

    async def aggregate_traces(
        self,
        aggregation: str,
        service: str,
        group_by: str = "traceID,name",
        time_range: str = "1h",
        limit: int = 5,
        error: bool = True,
        aggregate_on: str | None = None,
        filter_expr: str | None = None,
    ) -> list[dict[str, object]]:
        """Query the telemetry backend for aggregated trace metrics."""
        logger.info("Aggregating traces", service=service, aggregation=aggregation)

        order_expr = (
            f"{aggregation}({aggregate_on}) desc" if aggregate_on else f"{aggregation}() desc"
        )
        args: dict[str, Any] = {
            "aggregation": aggregation,
            "service": service,
            "groupBy": group_by,
            "limit": str(limit),
            "error": "true" if error else "false",
            "timeRange": time_range,
            "requestType": "scalar",
            "orderBy": order_expr,
        }
        if aggregate_on:
            args["aggregateOn"] = aggregate_on
        if filter_expr:
            args["filter"] = filter_expr

        result = await self.call_tool("signoz_aggregate_traces", args)
        logger.info("signoz_aggregate_traces — raw response", args=args, raw_result=result)

        # Parse the nested telemetry response structure
        outer = result.get("data", {}) if isinstance(result, dict) else {}
        inner = outer.get("data", {}) if isinstance(outer, dict) else {}
        response_results = inner.get("results", [])
        rows = response_results[0].get("data", []) if response_results else []

        logger.info("aggregate_traces — parsed rows", row_count=len(rows), rows=rows)
        return rows

    async def create_view(self, view_payload: dict[str, Any]) -> dict[str, Any]:
        """Create a saved view in the telemetry explorer."""
        logger.info("Creating telemetry saved view")
        return await self.call_tool("signoz_create_view", view_payload)


# Singleton instance for shared use across the application
telemetry_client = TelemetryMCPClient()
