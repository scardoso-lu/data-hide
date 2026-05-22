"""
Unit tests for AuditDB PostgreSQL persistence.

psycopg2.connect is fully mocked so no real database is needed.
"""

import json
from datetime import datetime, timezone

import pytest

from main import AuditDB, PIPELINE_VERSION, TableMapping, connect_audit_db

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

RUN_ID      = "aaaaaaaa-0000-0000-0000-000000000001"
STARTED_AT  = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
FINISHED_AT = datetime(2024, 1, 15, 10, 5, 30, tzinfo=timezone.utc)
SOURCE_URI  = "abfss://ws@onelake.dfs.fabric.microsoft.com/Src.Lakehouse/Tables/raw"
TARGET_URI  = "abfss://ws@onelake.dfs.fabric.microsoft.com/Tgt.Lakehouse/Tables/clean"
MAPPING     = TableMapping(SOURCE_URI, TARGET_URI, "test_table")

SUCCESS_AUDIT = {
    "pipeline_end_ts":         FINISHED_AT.isoformat(),
    "total_rows_processed":    500,
    "total_columns_in_table":  8,
    "total_columns_scanned":   5,
    "columns_anonymized":      ["email", "full_name"],
    "total_entities_detected": 312,
    "entity_counts":           {"EMAIL_ADDRESS": 200, "PERSON": 112},
    "unique_entities":         {"EMAIL_ADDRESS": 150, "PERSON": 80},
    "free_text_columns":       ["notes", "description"],
    "k_anonymity_k":           5,
    "quasi_columns":           ["age", "gender"],
    "suppressed_rows":         3,
    "residual_pii_count":      0,
    "column_renames":          {},
    "hashed_columns":          ["employee_id"],
    "key_vault_key_version":   "2025-05-22-resolved-version",
    "purview_available":       True,
    "purview_flagged_columns": ["email"],
    "purview_discrepancies":   [],
    "status":                  "success",
    "error_message":           None,
}

FAILURE_AUDIT = {
    **SUCCESS_AUDIT,
    "status":        "failure",
    "error_message": "Connection refused",
}


@pytest.fixture
def mock_psycopg2(mocker):
    """
    Mock psycopg2.connect and return (mock_connect, mock_conn, mock_cursor)
    where mock_cursor is the object yielded by `with conn.cursor() as cur:`.
    """
    mock_cursor = mocker.MagicMock()

    mock_cursor_cm = mocker.MagicMock()
    mock_cursor_cm.__enter__ = mocker.MagicMock(return_value=mock_cursor)
    mock_cursor_cm.__exit__  = mocker.MagicMock(return_value=False)

    mock_conn = mocker.MagicMock()
    mock_conn.cursor.return_value = mock_cursor_cm

    mock_connect = mocker.patch("main.psycopg2.connect", return_value=mock_conn)
    return mock_connect, mock_conn, mock_cursor


# ─────────────────────────────────────────────────────────────────────────────
# Schema initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaInit:

    def test_creates_runs_table(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        AuditDB("postgresql://test")
        sqls = [c.args[0] for c in cur.execute.call_args_list]
        assert any("pii_pipeline_runs" in s for s in sqls)

    def test_creates_column_events_table(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        AuditDB("postgresql://test")
        sqls = [c.args[0] for c in cur.execute.call_args_list]
        assert any("pii_pipeline_column_events" in s for s in sqls)

    def test_expected_ddl_statements(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        AuditDB("postgresql://test")
        assert cur.execute.call_count == 3

    def test_connection_committed(self, mock_psycopg2):
        _, conn, _ = mock_psycopg2
        AuditDB("postgresql://test")
        conn.commit.assert_called()

    def test_runs_table_has_new_columns(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        AuditDB("postgresql://test")
        sqls = " ".join(c.args[0] for c in cur.execute.call_args_list)
        for col in ("unique_entities", "free_text_cols", "k_anonymity_k",
                    "quasi_columns", "suppressed_rows", "residual_pii",
                    "column_renames", "hashed_columns", "key_vault_key_version"):
            assert col in sqls, f"Expected column '{col}' in DDL"


# ─────────────────────────────────────────────────────────────────────────────
# open_run
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenRun:

    def test_inserts_into_runs_table(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        db.open_run(RUN_ID, STARTED_AT, MAPPING)

        cur.execute.assert_called_once()
        sql, params = cur.execute.call_args.args
        assert "INSERT INTO pii_pipeline_runs" in sql

    def test_status_is_running(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        db.open_run(RUN_ID, STARTED_AT, MAPPING)
        _, params = cur.execute.call_args.args
        assert "running" in params

    def test_uris_in_params(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        db.open_run(RUN_ID, STARTED_AT, MAPPING)
        _, params = cur.execute.call_args.args
        assert SOURCE_URI in params
        assert TARGET_URI in params

    def test_pipeline_version_in_params(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        db.open_run(RUN_ID, STARTED_AT, MAPPING)
        _, params = cur.execute.call_args.args
        assert PIPELINE_VERSION in params


# ─────────────────────────────────────────────────────────────────────────────
# record_columns
# ─────────────────────────────────────────────────────────────────────────────

COLUMN_STATS = [
    {"column": "email",     "detections": 10, "entity_counts": {"EMAIL_ADDRESS": 10}},
    {"column": "full_name", "detections": 5,  "entity_counts": {"PERSON": 5}},
    {"column": "notes",     "detections": 0,  "entity_counts": {}},
]


class TestRecordColumns:

    def test_uses_execute_values_for_bulk_insert(self, mock_psycopg2, mocker):
        mock_exec_vals = mocker.patch("main.psycopg2.extras.execute_values")
        db = AuditDB("postgresql://test")

        db.record_columns(RUN_ID, COLUMN_STATS)

        mock_exec_vals.assert_called_once()

    def test_inserts_one_row_per_column(self, mock_psycopg2, mocker):
        mock_exec_vals = mocker.patch("main.psycopg2.extras.execute_values")
        db = AuditDB("postgresql://test")

        db.record_columns(RUN_ID, COLUMN_STATS)

        rows = mock_exec_vals.call_args.args[2]
        assert len(rows) == len(COLUMN_STATS)

    def test_column_names_preserved(self, mock_psycopg2, mocker):
        mock_exec_vals = mocker.patch("main.psycopg2.extras.execute_values")
        db = AuditDB("postgresql://test")

        db.record_columns(RUN_ID, COLUMN_STATS)

        rows = mock_exec_vals.call_args.args[2]
        inserted_cols = [r[1] for r in rows]
        assert inserted_cols == ["email", "full_name", "notes"]

    def test_detections_count_preserved(self, mock_psycopg2, mocker):
        mock_exec_vals = mocker.patch("main.psycopg2.extras.execute_values")
        db = AuditDB("postgresql://test")

        db.record_columns(RUN_ID, COLUMN_STATS)

        rows = mock_exec_vals.call_args.args[2]
        assert rows[0][2] == 10   # email
        assert rows[1][2] == 5    # full_name
        assert rows[2][2] == 0    # notes

    def test_entity_counts_json_encoded(self, mock_psycopg2, mocker):
        mock_exec_vals = mocker.patch("main.psycopg2.extras.execute_values")
        db = AuditDB("postgresql://test")

        db.record_columns(RUN_ID, [
            {"column": "email", "detections": 3, "entity_counts": {"EMAIL_ADDRESS": 3}},
        ])

        rows = mock_exec_vals.call_args.args[2]
        entity_json = rows[0][3]
        parsed = json.loads(entity_json)
        assert parsed["EMAIL_ADDRESS"] == 3

    def test_empty_stats_list_still_calls_execute_values(self, mock_psycopg2, mocker):
        mock_exec_vals = mocker.patch("main.psycopg2.extras.execute_values")
        db = AuditDB("postgresql://test")

        db.record_columns(RUN_ID, [])

        mock_exec_vals.assert_called_once()
        rows = mock_exec_vals.call_args.args[2]
        assert rows == []


# ─────────────────────────────────────────────────────────────────────────────
# close_run
# ─────────────────────────────────────────────────────────────────────────────

class TestCloseRun:

    def test_updates_runs_table(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        db.close_run(RUN_ID, SUCCESS_AUDIT)

        cur.execute.assert_called_once()
        sql, _ = cur.execute.call_args.args
        assert "UPDATE pii_pipeline_runs" in sql

    def test_success_status_in_params(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        db.close_run(RUN_ID, SUCCESS_AUDIT)
        _, params = cur.execute.call_args.args
        assert "success" in params

    def test_failure_status_and_error_message(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        db.close_run(RUN_ID, FAILURE_AUDIT)
        _, params = cur.execute.call_args.args
        assert "failure" in params
        assert "Connection refused" in params

    def test_null_error_message_on_success(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        db.close_run(RUN_ID, SUCCESS_AUDIT)
        _, params = cur.execute.call_args.args
        assert None in params  # error_message is NULL

    def test_row_counts_in_params(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        db.close_run(RUN_ID, SUCCESS_AUDIT)
        _, params = cur.execute.call_args.args
        assert 500 in params   # total_rows_processed
        assert 312 in params   # total_entities_detected

    def test_columns_anonymized_json_encoded(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        db.close_run(RUN_ID, SUCCESS_AUDIT)
        _, params = cur.execute.call_args.args
        json_params = [p for p in params if isinstance(p, str) and "email" in p]
        assert json_params, "No JSON-encoded columns_anonymized found in params"
        parsed = json.loads(json_params[0])
        assert "email" in parsed

    def test_unique_entities_json_encoded(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        db.close_run(RUN_ID, SUCCESS_AUDIT)
        _, params = cur.execute.call_args.args
        # unique_entities has EMAIL_ADDRESS: 150 (entity_counts has 200 — must find the right one)
        matching = [
            p for p in params
            if isinstance(p, str)
            and "EMAIL_ADDRESS" in p
            and json.loads(p).get("EMAIL_ADDRESS") == 150
        ]
        assert matching, "No JSON-encoded unique_entities with EMAIL_ADDRESS: 150 found in params"

    def test_suppressed_rows_in_params(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        db.close_run(RUN_ID, SUCCESS_AUDIT)
        _, params = cur.execute.call_args.args
        assert 3 in params  # suppressed_rows

    def test_residual_pii_count_in_params(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        db.close_run(RUN_ID, SUCCESS_AUDIT)
        _, params = cur.execute.call_args.args
        assert 0 in params  # residual_pii_count

    def test_column_renames_empty_json_encoded(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        db.close_run(RUN_ID, SUCCESS_AUDIT)
        _, params = cur.execute.call_args.args
        assert "{}" in params

    def test_k_anonymity_k_in_params(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        db.close_run(RUN_ID, SUCCESS_AUDIT)
        _, params = cur.execute.call_args.args
        assert 5 in params  # k_anonymity_k

    def test_hashed_columns_json_encoded(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        db.close_run(RUN_ID, SUCCESS_AUDIT)
        _, params = cur.execute.call_args.args
        matching = [
            p for p in params
            if isinstance(p, str)
            and "employee_id" in p
        ]
        assert matching, "No JSON-encoded hashed_columns found in params"
        parsed = json.loads(matching[0])
        assert "employee_id" in parsed

    def test_key_vault_key_version_in_params(self, mock_psycopg2):
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        db.close_run(RUN_ID, SUCCESS_AUDIT)
        _, params = cur.execute.call_args.args
        assert "2025-05-22-resolved-version" in params

    def test_key_vault_key_version_null_when_unset(self, mock_psycopg2):
        """Tables with no identifier columns produce audit rows with NULL key version."""
        _, _, cur = mock_psycopg2
        db = AuditDB("postgresql://test")
        cur.execute.reset_mock()

        audit_without_kv = {**SUCCESS_AUDIT, "key_vault_key_version": None}
        db.close_run(RUN_ID, audit_without_kv)
        _, params = cur.execute.call_args.args
        # NULL must be passed as Python None so psycopg2 maps it to SQL NULL.
        assert None in params


# ─────────────────────────────────────────────────────────────────────────────
# connect_audit_db factory
# ─────────────────────────────────────────────────────────────────────────────

class TestConnectAuditDB:

    def test_returns_none_when_url_is_none(self):
        assert connect_audit_db(None) is None

    def test_returns_none_on_connection_failure(self, mocker):
        mocker.patch("main.psycopg2.connect", side_effect=Exception("refused"))
        result = connect_audit_db("postgresql://bad-host/db")
        assert result is None

    def test_returns_audit_db_instance_on_success(self, mock_psycopg2):
        result = connect_audit_db("postgresql://pipeline:pw@db:5432/pii_audit")
        assert isinstance(result, AuditDB)

    def test_connection_failure_is_non_fatal(self, mocker):
        mocker.patch("main.psycopg2.connect", side_effect=OSError("host not found"))
        result = connect_audit_db("postgresql://unreachable/db")
        assert result is None
