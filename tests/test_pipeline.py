"""
Integration-style tests for the main() pipeline orchestration.

All external I/O is mocked:
  - delta-rs (DeltaTable / write_deltalake)
  - Azure identity (DefaultAzureCredential)
  - PostgreSQL (AuditDB via connect_audit_db)
  - HTTP webhook (requests.post)

anonymize_dataframe is also mocked in most tests so the spaCy model is not
loaded on every orchestration test; test_anonymization.py covers correctness.
One integration test re-uses the session-scoped Presidio fixture to verify
the full chain end-to-end.
"""

import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Shared test data
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_URI = "abfss://ws@onelake.dfs.fabric.microsoft.com/Src.Lakehouse/Tables/raw"
TARGET_URI = "abfss://ws@onelake.dfs.fabric.microsoft.com/Tgt.Lakehouse/Tables/clean"

REQUIRED_ENV = {
    "SOURCE_ABFSS_URI": SOURCE_URI,
    "TARGET_ABFSS_URI": TARGET_URI,
}

RAW_DF = pd.DataFrame({
    "name":  ["Alice Smith", "Bob Jones"],
    "email": ["alice@example.com", "bob@example.com"],
    "score": [10, 20],
})

ANON_DF = RAW_DF.copy()
ANON_DF["name"]  = ["***", "***"]
ANON_DF["email"] = ["***", "***"]

MOCK_STATS = {
    "text_columns_scanned":    ["name", "email"],
    "columns_with_detections": ["name", "email"],
    "entity_counts":           {"PERSON": 2, "EMAIL_ADDRESS": 2},
    "total_entities_detected": 4,
    "column_stats": [
        {"column": "name",  "detections": 2, "entity_counts": {"PERSON": 2}},
        {"column": "email", "detections": 2, "entity_counts": {"EMAIL_ADDRESS": 2}},
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def env(monkeypatch):
    """Set the minimum required env vars; remove optional ones."""
    for k, v in REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    for opt in ("DATABASE_URL", "PURVIEW_ACCOUNT_NAME", "ALERT_WEBHOOK_URL"):
        monkeypatch.delenv(opt, raising=False)


@pytest.fixture()
def mock_delta(mocker):
    mock_read = mocker.patch("main.DeltaTable")
    mock_read.return_value.to_pandas.return_value = RAW_DF.copy()
    mock_write = mocker.patch("main.write_deltalake")
    return mock_read, mock_write


@pytest.fixture()
def mock_auth(mocker):
    cred = mocker.patch("main.DefaultAzureCredential")
    cred.return_value.get_token.return_value = mocker.MagicMock(token="fake-bearer-token")
    return cred


@pytest.fixture()
def mock_anonymize(mocker):
    return mocker.patch(
        "main.anonymize_dataframe",
        return_value=(ANON_DF.copy(), MOCK_STATS),
    )


@pytest.fixture()
def mock_db(mocker):
    db = mocker.MagicMock()
    mocker.patch("main.connect_audit_db", return_value=db)
    return db


# ─────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineSuccess:

    def test_runs_without_exception(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db
    ):
        from main import main
        main()  # must not raise

    def test_write_is_called(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db
    ):
        from main import main
        _, mock_write = mock_delta
        main()
        assert mock_write.called

    def test_write_receives_anonymized_dataframe(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db
    ):
        from main import main
        _, mock_write = mock_delta
        main()

        written_df: pd.DataFrame = mock_write.call_args.args[1]
        assert list(written_df["name"])  == ["***", "***"]
        assert list(written_df["email"]) == ["***", "***"]

    def test_numeric_column_unchanged_after_write(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db
    ):
        from main import main
        _, mock_write = mock_delta
        main()

        written_df: pd.DataFrame = mock_write.call_args.args[1]
        assert list(written_df["score"]) == [10, 20]

    def test_write_target_uri_is_target(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db
    ):
        from main import main
        _, mock_write = mock_delta
        main()

        write_uri = mock_write.call_args.args[0]
        assert write_uri == TARGET_URI

    def test_audit_open_run_called(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db
    ):
        from main import main
        main()
        mock_db.open_run.assert_called_once()

    def test_audit_close_run_called_with_success(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db
    ):
        from main import main
        main()

        mock_db.close_run.assert_called_once()
        audit = mock_db.close_run.call_args.args[0]
        assert audit["status"] == "success"
        assert audit["error_message"] is None

    def test_audit_records_entity_counts(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db
    ):
        from main import main
        main()

        audit = mock_db.close_run.call_args.args[0]
        assert audit["total_entities_detected"] == 4
        assert audit["entity_counts"]["EMAIL_ADDRESS"] == 2

    def test_no_alert_sent_on_success(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db, mocker
    ):
        mock_alert = mocker.patch("main.send_alert")
        from main import main
        main()
        mock_alert.assert_not_called()

    def test_column_events_recorded_in_db(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db
    ):
        from main import main
        main()
        mock_db.record_columns.assert_called_once_with(MOCK_STATS["column_stats"])


# ─────────────────────────────────────────────────────────────────────────────
# Failure path
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineFailure:

    def test_exception_propagates(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db
    ):
        _, mock_write = mock_delta
        mock_write.side_effect = RuntimeError("storage unavailable")

        from main import main
        with pytest.raises(RuntimeError, match="storage unavailable"):
            main()

    def test_close_run_still_called_on_failure(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db
    ):
        _, mock_write = mock_delta
        mock_write.side_effect = RuntimeError("storage unavailable")

        from main import main
        with pytest.raises(RuntimeError):
            main()

        mock_db.close_run.assert_called_once()

    def test_close_run_status_is_failure(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db
    ):
        _, mock_write = mock_delta
        mock_write.side_effect = RuntimeError("disk full")

        from main import main
        with pytest.raises(RuntimeError):
            main()

        audit = mock_db.close_run.call_args.args[0]
        assert audit["status"] == "failure"

    def test_error_message_captured_in_audit(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db
    ):
        _, mock_write = mock_delta
        mock_write.side_effect = RuntimeError("disk full")

        from main import main
        with pytest.raises(RuntimeError):
            main()

        audit = mock_db.close_run.call_args.args[0]
        assert "disk full" in audit["error_message"]

    def test_alert_sent_when_webhook_configured(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db, mocker, monkeypatch
    ):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hook.example.com/abc")
        mock_alert = mocker.patch("main.send_alert")
        _, mock_write = mock_delta
        mock_write.side_effect = RuntimeError("timeout")

        from main import main
        with pytest.raises(RuntimeError):
            main()

        mock_alert.assert_called_once()
        subject = mock_alert.call_args.args[0]
        assert "FAILED" in subject

    def test_audit_close_run_called_even_when_db_write_fails(
        self, env, mock_delta, mock_auth, mock_anonymize, mocker
    ):
        """If the delta write fails mid-pipeline, close_run must still fire."""
        db = mocker.MagicMock()
        mocker.patch("main.connect_audit_db", return_value=db)
        _, mock_write = mock_delta
        mock_write.side_effect = IOError("write failed")

        from main import main
        with pytest.raises(IOError):
            main()

        db.close_run.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Optional features disabled
# ─────────────────────────────────────────────────────────────────────────────

class TestOptionalFeatures:

    def test_pipeline_runs_without_database_url(
        self, env, mock_delta, mock_auth, mock_anonymize, mocker
    ):
        """No DATABASE_URL → connect_audit_db returns None → pipeline continues."""
        mocker.patch("main.connect_audit_db", return_value=None)
        from main import main
        main()  # must not raise

    def test_pipeline_runs_without_webhook(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db
    ):
        from main import main
        main()  # no ALERT_WEBHOOK_URL set — must not raise

    def test_audit_db_failure_does_not_abort_pipeline(
        self, env, mock_delta, mock_auth, mock_anonymize, mocker
    ):
        db = mocker.MagicMock()
        db.open_run.side_effect = Exception("DB down")
        mocker.patch("main.connect_audit_db", return_value=db)

        from main import main
        main()  # pipeline must complete even if audit DB is down

    def test_purview_check_skipped_when_not_configured(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db, mocker
    ):
        mock_purview = mocker.patch("main.run_purview_check")
        from main import main
        main()
        # PURVIEW_ACCOUNT_NAME not set → run_purview_check is still called
        # but with purview_account=None (the function handles the skip internally)
        mock_purview.assert_called_once()
        assert mock_purview.call_args.args[2] is None


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end with real Presidio (uses session-scoped engine fixture)
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndAnonymization:

    def test_emails_anonymized_in_written_dataframe(
        self, env, mock_delta, mock_auth, mock_db, presidio_engines, mocker
    ):
        """Full chain: real Presidio engines, mocked storage + DB."""
        mocker.patch("main.build_engines", return_value=presidio_engines)
        _, mock_write = mock_delta

        from main import main
        main()

        written_df: pd.DataFrame = mock_write.call_args.args[1]
        for val in written_df["email"]:
            assert "example.com" not in val, f"Email not anonymized: {val!r}"
            assert "@" not in val, f"Email not fully masked: {val!r}"

    def test_score_column_untouched_in_written_dataframe(
        self, env, mock_delta, mock_auth, mock_db, presidio_engines, mocker
    ):
        mocker.patch("main.build_engines", return_value=presidio_engines)
        _, mock_write = mock_delta

        from main import main
        main()

        written_df: pd.DataFrame = mock_write.call_args.args[1]
        assert list(written_df["score"]) == [10, 20]
