#!/usr/bin/env python3
"""MCP Server entry point for AI Monitoring.

Streamable HTTP transport on port 9009 — run in your terminal:
    python mcp_server.py

Connect agents with:
    cmd mcp add --scope user --transport http ai-monitoring \
      http://127.0.0.1:9009/mcp
"""

from __future__ import annotations

import os

import anyio
import structlog
from dotenv import load_dotenv

load_dotenv()

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=False),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

_log = structlog.get_logger("mcp_server")


async def _startup() -> None:
    """Initialize the telemetry MCP client."""
    from clients.telemetry_mcp_client import telemetry_client

    await telemetry_client.start()
    _log.info("Telemetry MCP client ready")


async def _run_server() -> None:
    """Configure and start the MCP server."""
    from routers.mcp_tools import mcp_server  # noqa: E402

    mcp_server.settings.host = os.environ.get("MCP_HOST", "127.0.0.1")
    mcp_server.settings.port = int(os.environ.get("MCP_PORT", "9009"))

    _log.info(
        "MCP server ready for connections",
        host=mcp_server.settings.host,
        port=mcp_server.settings.port,
    )
    await mcp_server.run_streamable_http_async()


async def _main() -> None:
    """Main entry point managing the server lifecycle."""
    from clients.telemetry_mcp_client import telemetry_client

    await _startup()
    try:
        await _run_server()
    finally:
        await telemetry_client.stop()


if __name__ == "__main__":
    anyio.run(_main)
