"""Tests for PostgreSQL-backed alert records."""

from app.application.pipeline import record_alert


class TestRecordAlert:

    def test_no_op_when_db_missing(self):
        record_alert(None, "run-1", None, "Subject", "Body")

    def test_persists_alert_to_db(self, mocker):
        db = mocker.MagicMock()
        mapping = mocker.MagicMock(table_name="customers")

        record_alert(db, "run-1", mapping, "Pipeline FAILED", "error: boom")

        db.record_alert.assert_called_once_with(
            "run-1",
            "customers",
            "Pipeline FAILED",
            "error: boom",
        )

    def test_db_error_is_non_fatal(self, mocker):
        db = mocker.MagicMock()
        db.record_alert.side_effect = RuntimeError("db unavailable")

        record_alert(db, "run-1", None, "Subject", "Body")
