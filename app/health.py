"""HTTP health endpoints for container orchestrators (Azure Container Apps).

Serves three probe endpoints from a daemon thread using only the standard
library — no new dependencies, no container-local file writes:

* ``/healthz/live``     – 200 while the process is able to serve HTTP.
                          A failure means the interpreter is dead or wedged;
                          the orchestrator should restart the container.
* ``/healthz/startup``  – 503 until :func:`mark_started` is called (heavy
                          imports finished and the entrypoint reached
                          ``main()``), then 200.
* ``/healthz/ready``    – 503 until :func:`mark_ready` is called (pipeline
                          wiring complete, processing begins), then 200
                          until :func:`mark_not_ready`.

The server binds ``0.0.0.0:$HEALTH_PORT`` (default 8080) and is enabled by
default; set ``HEALTH_PROBES_ENABLED=0`` to opt out (e.g. local one-shot
runs outside an orchestrator).
"""

from __future__ import annotations

import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger(__name__)

DEFAULT_HEALTH_PORT = 8080

_started = threading.Event()
_ready = threading.Event()


def mark_started() -> None:
    """Signal that process bootstrap is complete (startup probe passes)."""
    _started.set()


def mark_ready() -> None:
    """Signal that the pipeline is wired and processing (readiness passes)."""
    _ready.set()


def mark_not_ready() -> None:
    """Signal that the pipeline is no longer processing (readiness fails)."""
    _ready.clear()


def reset() -> None:
    """Clear all probe state. Intended for tests."""
    _started.clear()
    _ready.clear()


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — http.server API
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/healthz/live":
            self._respond(200, "alive")
        elif path == "/healthz/startup":
            if _started.is_set():
                self._respond(200, "started")
            else:
                self._respond(503, "starting")
        elif path == "/healthz/ready":
            if _ready.is_set():
                self._respond(200, "ready")
            else:
                self._respond(503, "not ready")
        else:
            self._respond(404, "not found")

    def _respond(self, status: int, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 — http.server API
        # Probes fire every few seconds; routing them through the access log
        # would drown the pipeline's own output.
        pass


def start_health_server(port: int | None = None) -> ThreadingHTTPServer | None:
    """Start the probe server in a daemon thread and return it.

    Returns ``None`` without binding when ``HEALTH_PROBES_ENABLED=0``.
    Pass ``port=0`` to bind an ephemeral port (tests).
    """
    if os.environ.get("HEALTH_PROBES_ENABLED", "1") == "0":
        logger.info("Health probe server disabled (HEALTH_PROBES_ENABLED=0)")
        return None

    if port is None:
        port = int(os.environ.get("HEALTH_PORT", DEFAULT_HEALTH_PORT))

    server = ThreadingHTTPServer(("0.0.0.0", port), _HealthHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="health-probes", daemon=True)
    thread.start()
    logger.info("Health probe server listening on port %d", server.server_address[1])
    return server
