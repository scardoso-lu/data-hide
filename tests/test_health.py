"""Tests for the container health probe endpoints (app/health.py)."""

from __future__ import annotations

import http.client

import pytest

from app import health


@pytest.fixture
def health_server(monkeypatch):
    monkeypatch.delenv("HEALTH_PROBES_ENABLED", raising=False)
    health.reset()
    server = health.start_health_server(port=0)  # ephemeral port
    assert server is not None
    yield server
    server.shutdown()
    server.server_close()
    health.reset()


def _get(server, path: str) -> tuple[int, str]:
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        return response.status, response.read().decode("utf-8")
    finally:
        conn.close()


class TestLivenessProbe:
    def test_live_returns_200_immediately(self, health_server):
        status, body = _get(health_server, "/healthz/live")
        assert status == 200
        assert body == "alive"

    def test_live_unaffected_by_startup_and_readiness_state(self, health_server):
        health.mark_started()
        health.mark_ready()
        health.mark_not_ready()
        status, _ = _get(health_server, "/healthz/live")
        assert status == 200


class TestStartupProbe:
    def test_startup_503_before_mark_started(self, health_server):
        status, _ = _get(health_server, "/healthz/startup")
        assert status == 503

    def test_startup_200_after_mark_started(self, health_server):
        health.mark_started()
        status, _ = _get(health_server, "/healthz/startup")
        assert status == 200


class TestReadinessProbe:
    def test_ready_503_before_mark_ready(self, health_server):
        status, _ = _get(health_server, "/healthz/ready")
        assert status == 503

    def test_ready_200_after_mark_ready(self, health_server):
        health.mark_ready()
        status, _ = _get(health_server, "/healthz/ready")
        assert status == 200

    def test_ready_503_again_after_mark_not_ready(self, health_server):
        health.mark_ready()
        health.mark_not_ready()
        status, _ = _get(health_server, "/healthz/ready")
        assert status == 503


class TestServerBehaviour:
    def test_unknown_path_returns_404(self, health_server):
        status, _ = _get(health_server, "/healthz/nope")
        assert status == 404

    def test_trailing_slash_and_query_string_tolerated(self, health_server):
        status, _ = _get(health_server, "/healthz/live/?probe=1")
        assert status == 200

    def test_disabled_via_env_returns_none(self, monkeypatch):
        monkeypatch.setenv("HEALTH_PROBES_ENABLED", "0")
        assert health.start_health_server(port=0) is None
