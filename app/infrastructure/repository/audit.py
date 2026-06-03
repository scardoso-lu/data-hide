"""PostgreSQL audit persistence — run records, column events, alerts, config, exclusions."""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from datetime import datetime
from typing import Generator

from ._types import TableMapping

try:
    import psycopg2
    import psycopg2.extras
except ModuleNotFoundError:
    class _MissingPsycopg2:
        class extras:
            @staticmethod
            def execute_values(*args, **kwargs):
                raise ModuleNotFoundError("No module named 'psycopg2'")

        @staticmethod
        def connect(*args, **kwargs):
            raise ModuleNotFoundError("No module named 'psycopg2'")

    psycopg2 = _MissingPsycopg2()

PIPELINE_VERSION = "2.3.0"

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────────────────────────────────────

_DDL_RUNS = """
CREATE TABLE IF NOT EXISTS pii_pipeline_runs (
    run_id UUID PRIMARY KEY,
    pipeline_version TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    table_name TEXT,
    source_uri TEXT NOT NULL,
    target_uri TEXT NOT NULL,
    total_rows INTEGER,
    total_columns INTEGER,
    columns_scanned INTEGER,
    columns_hit JSONB,
    entities_total INTEGER,
    entity_counts JSONB,
    unique_entities JSONB,
    free_text_cols JSONB,
    k_anonymity_k INTEGER,
    quasi_columns JSONB,
    suppressed_rows INTEGER NOT NULL DEFAULT 0,
    residual_pii INTEGER NOT NULL DEFAULT 0,
    column_renames JSONB,
    gps_columns JSONB,
    timestamp_cols_binned JSONB,
    numeric_cols_binned JSONB,
    hashed_columns JSONB,
    key_vault_key_version TEXT,
    stage_seconds JSONB,
    purview_ok BOOLEAN NOT NULL DEFAULT FALSE,
    purview_flagged JSONB,
    purview_diffs JSONB,
    output_type TEXT NOT NULL DEFAULT 'anonymized_rows',
    aggregate_cells INTEGER,
    status TEXT NOT NULL DEFAULT 'running',
    error_msg TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_DDL_RUNS_MIGRATIONS = [
    "ALTER TABLE pii_pipeline_runs ADD COLUMN IF NOT EXISTS key_vault_key_version TEXT",
    "ALTER TABLE pii_pipeline_runs ADD COLUMN IF NOT EXISTS stage_seconds JSONB",
]

_DDL_COLUMN_EVENTS = """
CREATE TABLE IF NOT EXISTS pii_pipeline_column_events (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES pii_pipeline_runs(run_id),
    column_name TEXT NOT NULL,
    detections INTEGER NOT NULL DEFAULT 0,
    entity_counts JSONB,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_DDL_ALERTS = """
CREATE TABLE IF NOT EXISTS pii_pipeline_alerts (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID,
    table_name TEXT,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'error',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_DDL_CONFIG = """
CREATE TABLE IF NOT EXISTS pii_pipeline_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_DDL_COLUMN_EXCLUSIONS = """
CREATE TABLE IF NOT EXISTS pii_column_exclusions (
    table_name TEXT NOT NULL,
    column_name TEXT NOT NULL,
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (table_name, column_name)
)
"""

_DDL_TABLE_TARGETS = """
CREATE TABLE IF NOT EXISTS pii_table_targets (
    id BIGSERIAL PRIMARY KEY,
    source_uri TEXT NOT NULL,
    target_uri TEXT NOT NULL,
    table_name TEXT,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


# ─────────────────────────────────────────────────────────────────────────────
# AuditDB
# ─────────────────────────────────────────────────────────────────────────────

class AuditDB:
    def __init__(self, dsn_or_kwargs: "str | dict") -> None:
        # Accept either a DSN string (tests, DATABASE_URL) or a kwargs dict
        # (individual DB_* env vars — avoids URL-encoding special characters).
        if isinstance(dsn_or_kwargs, str):
            self._connect_kwargs: dict = {"dsn": dsn_or_kwargs}
        else:
            self._connect_kwargs = dsn_or_kwargs
        self._init_schema()

    @contextmanager
    def _cursor(self) -> Generator:
        conn = psycopg2.connect(**self._connect_kwargs)
        try:
            with conn.cursor() as cur:
                yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._cursor() as cur:
            cur.execute(_DDL_RUNS)
            cur.execute(_DDL_COLUMN_EVENTS)
            cur.execute(_DDL_ALERTS)
            cur.execute(_DDL_CONFIG)
            cur.execute(_DDL_COLUMN_EXCLUSIONS)
            cur.execute(_DDL_TABLE_TARGETS)
            for migration in _DDL_RUNS_MIGRATIONS:
                cur.execute(migration)

    def open_run(self, run_id: str, started_at: datetime, mapping) -> None:
        # Pre-populate error_msg so that rows left in status='running' by an
        # OOM kill (SIGKILL) — which bypasses all Python finally-blocks — are
        # immediately identifiable.  close_run() overwrites this on normal exit.
        sql = """
            INSERT INTO pii_pipeline_runs
                (run_id, pipeline_version, started_at, table_name, source_uri, target_uri,
                 status, error_msg)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._cursor() as cur:
            cur.execute(sql, (
                run_id, PIPELINE_VERSION, started_at,
                mapping.table_name, mapping.source_uri, mapping.target_uri,
                "running",
                "process killed before run completed (OOM/SIGKILL — close_run never called)",
            ))

    def record_columns(self, run_id: str, column_stats: list) -> None:
        rows = [
            (run_id, s["column"], s["detections"], json.dumps(s["entity_counts"]))
            for s in column_stats
        ]
        sql = """
            INSERT INTO pii_pipeline_column_events
                (run_id, column_name, detections, entity_counts)
            VALUES %s
        """
        with self._cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, rows)

    def load_runtime_config(self) -> dict[str, str]:
        """Return all key/value pairs from ``pii_pipeline_config``.

        These values override the corresponding environment variables for every
        runtime-tunable parameter.  Secrets and connectivity settings
        (``DATABASE_URL``, Azure credentials, Key Vault, source/target URIs,
        ``HASH_SALT``) must remain in the environment — they are intentionally
        excluded from this table.
        """
        with self._cursor() as cur:
            cur.execute("SELECT key, value FROM pii_pipeline_config")
            return {row[0]: row[1] for row in cur.fetchall()}

    def load_column_exclusions(self) -> dict[str, frozenset[str]]:
        """Return ``{table_name: frozenset(column_names)}`` from ``pii_column_exclusions``.

        Columns in this mapping are removed from the anonymization policy
        before any masking action is applied — they pass through untouched.
        Table names are lowercased so lookups are case-insensitive.
        """
        result: dict[str, list[str]] = {}
        with self._cursor() as cur:
            cur.execute("SELECT table_name, column_name FROM pii_column_exclusions")
            for table, column in cur.fetchall():
                result.setdefault(table.lower(), []).append(column)
        return {t: frozenset(cols) for t, cols in result.items()}

    def load_table_targets(self) -> list[TableMapping]:
        """Return enabled rows from ``pii_table_targets`` as ``TableMapping`` objects.

        When this list is non-empty, ``resolve_table_mappings`` uses it directly
        and skips auto-discovery under ``SOURCE_BASE_ABFSS_URI``.
        """
        with self._cursor() as cur:
            cur.execute(
                "SELECT source_uri, target_uri, table_name"
                " FROM pii_table_targets WHERE enabled = TRUE"
            )
            return [TableMapping(row[0], row[1], row[2]) for row in cur.fetchall()]

    def record_alert(
        self,
        run_id: str | None,
        table_name: str | None,
        subject: str,
        body: str,
        severity: str = "error",
    ) -> None:
        sql = """
            INSERT INTO pii_pipeline_alerts (run_id, table_name, subject, body, severity)
            VALUES (%s, %s, %s, %s, %s)
        """
        with self._cursor() as cur:
            cur.execute(sql, (run_id, table_name, subject, body, severity))

    def close_run(self, run_id: str, audit: dict) -> None:
        sql = """
            UPDATE pii_pipeline_runs SET
                finished_at = %s,
                total_rows = %s,
                total_columns = %s,
                columns_scanned = %s,
                columns_hit = %s,
                entities_total = %s,
                entity_counts = %s,
                unique_entities = %s,
                free_text_cols = %s,
                k_anonymity_k = %s,
                quasi_columns = %s,
                suppressed_rows = %s,
                residual_pii = %s,
                column_renames = %s,
                gps_columns = %s,
                timestamp_cols_binned = %s,
                numeric_cols_binned = %s,
                hashed_columns = %s,
                key_vault_key_version = %s,
                stage_seconds = %s,
                purview_ok = %s,
                purview_flagged = %s,
                purview_diffs = %s,
                output_type = %s,
                aggregate_cells = %s,
                status = %s,
                error_msg = %s
            WHERE run_id = %s
        """
        with self._cursor() as cur:
            cur.execute(sql, (
                audit.get("pipeline_end_ts"),
                audit.get("total_rows_processed"),
                audit.get("total_columns_in_table"),
                audit.get("total_columns_scanned"),
                json.dumps(audit.get("columns_anonymized", [])),
                audit.get("total_entities_detected"),
                json.dumps(audit.get("entity_counts", {})),
                json.dumps(audit.get("unique_entities", {})),
                json.dumps(audit.get("free_text_columns", [])),
                audit.get("k_anonymity_k"),
                json.dumps(audit.get("quasi_columns", [])),
                audit.get("suppressed_rows", 0),
                audit.get("residual_pii_count", 0),
                json.dumps(audit.get("column_renames", {})),
                json.dumps(audit.get("gps_columns_anonymized", [])),
                json.dumps(audit.get("timestamp_columns_binned", [])),
                json.dumps(audit.get("numeric_columns_binned", [])),
                json.dumps(audit.get("hashed_columns", [])),
                audit.get("key_vault_key_version"),
                json.dumps(audit.get("stage_seconds", {})),
                audit.get("purview_available", False),
                json.dumps(audit.get("purview_flagged_columns", [])),
                json.dumps(audit.get("purview_discrepancies", [])),
                audit.get("output_type", "anonymized_rows"),
                audit.get("aggregate_cells"),
                audit.get("status"),
                audit.get("error_message"),
                run_id,
            ))


def _build_db_connect_kwargs(database_url: str | None) -> "dict | None":
    """Return psycopg2 connect kwargs.

    Individual ``DB_*`` env vars take priority over ``database_url`` because
    they are passed as keyword arguments to ``psycopg2.connect()`` — no URL
    encoding is needed, so passwords with special characters work as-is.
    Falls back to ``database_url`` (the old ``DATABASE_URL`` string) when the
    individual vars are not configured.
    """
    import os as _os
    host = _os.environ.get("DB_HOST")
    if host:
        return {
            "host": host,
            "port": int(_os.environ.get("DB_PORT", "5432")),
            "user": _os.environ.get("DB_USER", ""),
            "password": _os.environ.get("DB_PASSWORD", ""),
            "dbname": _os.environ.get("DB_NAME", ""),
        }
    if database_url:
        return {"dsn": database_url}
    return None


def connect_audit_db(database_url: str | None) -> "AuditDB | None":
    connect_kwargs = _build_db_connect_kwargs(database_url)
    if not connect_kwargs:
        logger.info("DATABASE_URL not set and DB_HOST not configured; audit DB disabled.")
        return None
    try:
        return AuditDB(connect_kwargs)
    except Exception as exc:
        logger.warning("Audit DB connection failed (non-fatal): %s", exc)
        return None
