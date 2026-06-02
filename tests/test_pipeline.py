"""
Integration-style tests for the main() pipeline orchestration.

All external I/O is mocked:
  - Delta read adapter / Delta output adapter
  - Azure identity (DefaultAzureCredential)
  - PostgreSQL (AuditDB via connect_audit_db)
  - HTTP webhook (requests.post)

anonymize_dataframe, build_engines, validate_residual_pii,
and classify_columns are also mocked in most
tests so the spaCy model is not loaded on every orchestration test.
test_anonymization.py covers correctness.
One integration test re-uses the session-scoped analyzer fixture to verify
the full chain end-to-end.
"""

import pandas as pd
import pytest

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Shared test data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

SOURCE_URI = "abfss://ws@onelake.dfs.fabric.microsoft.com/Src.Lakehouse/Tables/raw"
TARGET_URI = "abfss://ws@onelake.dfs.fabric.microsoft.com/Tgt.Lakehouse/Tables/clean"
RAW_TARGET = "abfss://ws@onelake.dfs.fabric.microsoft.com/Tgt.Lakehouse/Tables/dbo/jaffle_raw_customers"

BASE_SOURCE_URI = "abfss://ws@onelake.dfs.fabric.microsoft.com/Src.Lakehouse"
BASE_TARGET_URI = "abfss://ws@onelake.dfs.fabric.microsoft.com/Tgt.Lakehouse/Tables"

REQUIRED_ENV = {
    "SOURCE_BASE_ABFSS_URI": BASE_SOURCE_URI,
    "TARGET_BASE_ABFSS_URI": BASE_TARGET_URI,
}

RAW_DF = pd.DataFrame({
    "name":  ["Alice Smith", "Bob Jones"],
    "email": ["alice@example.com", "bob@example.com"],
    "score": [10, 20],
})

ANON_DF = RAW_DF.copy()
ANON_DF["name"]  = ["PERSON_0", "PERSON_1"]
ANON_DF["email"] = ["EMAIL_ADDRESS_0", "EMAIL_ADDRESS_1"]

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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Fixtures
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@pytest.fixture()
def env(monkeypatch, mocker):
    """Set required env vars and default discover mock returning a single table."""
    for k, v in REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    for opt in ("DATABASE_URL", "PURVIEW_ACCOUNT_NAME", "K_ANONYMITY_MIN", "QUASI_IDENTIFIER_COLS"):
        monkeypatch.delenv(opt, raising=False)
    from main import TableMapping
    mocker.patch("main.discover_table_mappings",
                 return_value=[TableMapping(SOURCE_URI, TARGET_URI, "test_table")])


@pytest.fixture()
def mock_delta(mocker):
    mock_read = mocker.patch("main.read_delta", return_value=RAW_DF.copy())
    mock_write = mocker.patch("main.write_delta")
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


@pytest.fixture()
def mock_engines(mocker):
    """Prevent spaCy model load in unit tests."""
    return mocker.patch("main.build_engines", return_value=mocker.MagicMock())


@pytest.fixture()
def mock_validate(mocker):
    """Residual PII validation â€” always clean in orchestration unit tests."""
    return mocker.patch("main.validate_residual_pii", return_value=0)


@pytest.fixture()
def mock_classify(mocker):
    return mocker.patch("main.classify_columns", return_value=[])


@pytest.fixture()
def mock_hash(mocker):
    """Mock pseudonymize_identifier_columns â€” identity transform, no columns pseudonymized."""
    m = mocker.patch("main.pseudonymize_identifier_columns")
    m.side_effect = lambda df, cols, pseudonymizer: (df.copy(), [])
    return m


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Happy path
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestPipelineSuccess:

    def test_runs_without_exception(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        from main import main
        main()  # must not raise

    def test_write_is_called(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        from main import main
        _, mock_write = mock_delta
        main()
        assert mock_write.called

    def test_write_receives_anonymized_dataframe(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        from main import main
        _, mock_write = mock_delta
        main()

        written_df: pd.DataFrame = mock_write.call_args.args[0]
        assert list(written_df["name"])  == ["PERSON_0", "PERSON_1"]
        assert list(written_df["email"]) == ["EMAIL_ADDRESS_0", "EMAIL_ADDRESS_1"]

    def test_numeric_column_unchanged_after_write(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        from main import main
        _, mock_write = mock_delta
        main()

        written_df: pd.DataFrame = mock_write.call_args.args[0]
        assert list(written_df["score"]) == [10, 20]

    def test_write_target_uri_is_target(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        from main import main
        _, mock_write = mock_delta
        main()

        write_uri = mock_write.call_args.args[1]
        assert write_uri == TARGET_URI

    def test_audit_open_run_called(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        from main import main
        main()
        mock_db.open_run.assert_called_once()

    def test_audit_close_run_called_with_success(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        from main import main
        main()

        mock_db.close_run.assert_called_once()
        audit = mock_db.close_run.call_args.args[1]
        assert audit["status"] == "success"
        assert audit["error_message"] is None

    def test_audit_records_entity_counts(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        from main import main
        main()

        audit = mock_db.close_run.call_args.args[1]
        assert audit["total_entities_detected"] == 4
        assert audit["entity_counts"]["EMAIL_ADDRESS"] == 2

    def test_no_alert_recorded_on_success(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        from main import main
        main()
        mock_db.record_alert.assert_not_called()

    def test_column_events_recorded_in_db(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        from main import main
        main()
        mock_db.record_columns.assert_called_once()
        assert mock_db.record_columns.call_args.args[1] == MOCK_STATS["column_stats"]

    def test_residual_validation_called(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        from main import main
        main()
        mock_validate.assert_called_once()

    def test_column_classification_called(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        from main import main
        main()
        mock_classify.assert_called_once()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Failure path
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestPipelineFailure:

    def test_refuses_same_source_and_target(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db, mocker,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        from main import TableMapping
        mocker.patch("main.discover_table_mappings",
                     return_value=[TableMapping(SOURCE_URI, SOURCE_URI, "test_table")])

        from main import main
        with pytest.raises(RuntimeError, match="Source and target table URIs are identical"):
            main()

    def test_allows_raw_named_target_when_writing_parquet_file(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db, mocker,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        from main import TableMapping
        mocker.patch("main.discover_table_mappings",
                     return_value=[TableMapping(SOURCE_URI, RAW_TARGET, "test_table")])

        from main import main
        main()

    def test_exception_propagates(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        _, mock_write = mock_delta
        mock_write.side_effect = RuntimeError("storage unavailable")

        from main import main
        with pytest.raises(RuntimeError, match="storage unavailable"):
            main()

    def test_close_run_still_called_on_failure(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        _, mock_write = mock_delta
        mock_write.side_effect = RuntimeError("storage unavailable")

        from main import main
        with pytest.raises(RuntimeError):
            main()

        mock_db.close_run.assert_called_once()

    def test_close_run_status_is_failure(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        _, mock_write = mock_delta
        mock_write.side_effect = RuntimeError("disk full")

        from main import main
        with pytest.raises(RuntimeError):
            main()

        audit = mock_db.close_run.call_args.args[1]
        assert audit["status"] == "failure"

    def test_error_message_captured_in_audit(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        _, mock_write = mock_delta
        mock_write.side_effect = RuntimeError("disk full")

        from main import main
        with pytest.raises(RuntimeError):
            main()

        audit = mock_db.close_run.call_args.args[1]
        assert "disk full" in audit["error_message"]

    def test_alert_recorded_on_failure(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        _, mock_write = mock_delta
        mock_write.side_effect = RuntimeError("timeout")

        from main import main
        with pytest.raises(RuntimeError):
            main()

        mock_db.record_alert.assert_called_once()
        subject = mock_db.record_alert.call_args.args[2]
        assert "FAILED" in subject

    def test_audit_close_run_called_even_when_db_write_fails(
        self, env, mock_delta, mock_auth, mock_anonymize, mocker,
        mock_engines, mock_validate, mock_classify, mock_hash,
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

    def test_residual_pii_aborts_pipeline(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_classify, mock_hash, mocker,
    ):
        """validate_residual_pii raising RuntimeError must abort before write."""
        mocker.patch(
            "main.validate_residual_pii",
            side_effect=RuntimeError("Residual PII detected after anonymization: 2 finding(s)"),
        )
        _, mock_write = mock_delta

        from main import main
        with pytest.raises(RuntimeError, match="Residual PII"):
            main()

        mock_write.assert_not_called()

    def test_invalid_read_mode_aborts_before_read(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash, mocker,
    ):
        from main import TableMapping

        mocker.patch(
            "main.discover_table_mappings",
            return_value=[TableMapping(SOURCE_URI, TARGET_URI, "test_table", read_mode="csv")],
        )
        mock_read, mock_write = mock_delta

        from main import main
        with pytest.raises(RuntimeError, match="Unsupported read_mode"):
            main()

        mock_read.assert_not_called()
        mock_write.assert_not_called()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Optional features disabled
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestOptionalFeatures:

    def test_pipeline_runs_without_database_url(
        self, env, mock_delta, mock_auth, mock_anonymize, mocker,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        """No DATABASE_URL â†’ connect_audit_db returns None â†’ pipeline continues."""
        mocker.patch("main.connect_audit_db", return_value=None)
        from main import main
        main()  # must not raise

    def test_pipeline_runs_without_alert_webhook(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        from main import main
        main()  # alert webhooks are not part of the runtime contract

    def test_audit_db_failure_does_not_abort_pipeline(
        self, env, mock_delta, mock_auth, mock_anonymize, mocker,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        db = mocker.MagicMock()
        db.open_run.side_effect = Exception("DB down")
        mocker.patch("main.connect_audit_db", return_value=db)

        from main import main
        main()  # pipeline must complete even if audit DB is down

    def test_purview_check_skipped_when_not_configured(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db, mocker,
        mock_engines, mock_validate, mock_classify, mock_hash,
    ):
        mock_purview = mocker.patch("main.run_purview_check")
        mock_purview.return_value = {
            "available": False, "flagged_columns": [],
            "column_labels": {}, "discrepancies": [],
        }
        from main import main
        main()
        # PURVIEW_ACCOUNT_NAME not set â†’ run_purview_check is still called
        # but with purview_account=None (the function handles the skip internally)
        mock_purview.assert_called_once()
        assert mock_purview.call_args.args[2] is None

    def test_k_anonymity_skipped_when_no_quasi_cols(
        self, env, mock_delta, mock_auth, mock_anonymize, mock_db,
        mock_engines, mock_validate, mock_classify, mocker,
    ):
        """No quasi-identifier category means enforce_k_anonymity is never called."""
        mock_k = mocker.patch("main.enforce_k_anonymity")

        from main import main
        main()

        mock_k.assert_not_called()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# End-to-end with real Presidio (uses session-scoped engine fixture)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestPipelineConfigValidation:

    def test_rejects_non_positive_max_table_workers(self, monkeypatch):
        from main import PipelineConfig

        monkeypatch.setenv("MAX_TABLE_WORKERS", "0")

        with pytest.raises(ValueError, match="MAX_TABLE_WORKERS"):
            PipelineConfig.from_env()


# ─────────────────────────────────────────────────────────────────────────────
# Runtime table targets (pii_table_targets)
# ─────────────────────────────────────────────────────────────────────────────

SOURCE2 = "abfss://ws@onelake.dfs.fabric.microsoft.com/A.Lakehouse/Tables/raw"
TARGET2 = "abfss://ws@onelake.dfs.fabric.microsoft.com/B.Lakehouse/Tables/clean"


class TestRuntimeTableTargets:

    def test_resolve_uses_db_targets_when_present(self, monkeypatch):
        from main import PipelineConfig, TableMapping, resolve_table_mappings

        monkeypatch.delenv("SOURCE_BASE_ABFSS_URI", raising=False)
        monkeypatch.delenv("TARGET_BASE_ABFSS_URI", raising=False)
        t1 = TableMapping(SOURCE_URI, TARGET_URI, "orders")
        t2 = TableMapping(SOURCE2, TARGET2, "customers")
        config = PipelineConfig.from_env(table_targets=(t1, t2))

        result = resolve_table_mappings(config)

        assert result == [t1, t2]

    def test_resolve_skips_auto_discovery_when_targets_present(self, monkeypatch, mocker):
        from main import PipelineConfig, TableMapping, resolve_table_mappings

        monkeypatch.setenv("SOURCE_BASE_ABFSS_URI", BASE_SOURCE_URI)
        monkeypatch.setenv("TARGET_BASE_ABFSS_URI", BASE_TARGET_URI)
        mock_discover = mocker.patch("main.discover_table_mappings")
        config = PipelineConfig.from_env(
            table_targets=(TableMapping(SOURCE_URI, TARGET_URI, "t"),)
        )

        resolve_table_mappings(config)

        mock_discover.assert_not_called()

    def test_resolve_falls_back_to_discovery_when_targets_empty(self, monkeypatch, mocker):
        from main import PipelineConfig, TableMapping, resolve_table_mappings

        monkeypatch.setenv("SOURCE_BASE_ABFSS_URI", BASE_SOURCE_URI)
        monkeypatch.setenv("TARGET_BASE_ABFSS_URI", BASE_TARGET_URI)
        expected = [TableMapping(SOURCE_URI, TARGET_URI, "t")]
        # patch inside the pipeline module where resolve_table_mappings lives
        mock_discover = mocker.patch(
            "app.application.pipeline.discover_table_mappings", return_value=expected
        )
        config = PipelineConfig.from_env()

        result = resolve_table_mappings(config)

        mock_discover.assert_called_once()
        assert result == expected

    def test_resolve_raises_without_base_uris_and_no_targets(self, monkeypatch):
        from main import PipelineConfig, resolve_table_mappings

        monkeypatch.delenv("SOURCE_BASE_ABFSS_URI", raising=False)
        monkeypatch.delenv("TARGET_BASE_ABFSS_URI", raising=False)
        config = PipelineConfig.from_env()

        with pytest.raises(RuntimeError, match="pii_table_targets"):
            resolve_table_mappings(config)

    def test_from_env_and_db_loads_targets(self, mocker):
        from main import PipelineConfig, TableMapping

        t = TableMapping(SOURCE_URI, TARGET_URI, "orders")
        db = mocker.MagicMock()
        db.load_runtime_config.return_value = {}
        db.load_column_exclusions.return_value = {}
        db.load_table_targets.return_value = [t]

        config = PipelineConfig.from_env_and_db(db)

        assert config.table_targets == (t,)

    def test_from_env_and_db_targets_empty_on_db_error(self, mocker):
        from main import PipelineConfig

        db = mocker.MagicMock()
        db.load_runtime_config.return_value = {}
        db.load_column_exclusions.return_value = {}
        db.load_table_targets.side_effect = Exception("table does not exist")

        config = PipelineConfig.from_env_and_db(db)

        assert config.table_targets == ()

    def test_pipeline_runs_with_db_targets_no_base_uris(
        self, monkeypatch, mock_delta, mock_auth, mock_anonymize,
        mock_engines, mock_validate, mock_classify, mock_hash, mocker,
    ):
        monkeypatch.delenv("SOURCE_BASE_ABFSS_URI", raising=False)
        monkeypatch.delenv("TARGET_BASE_ABFSS_URI", raising=False)
        from main import TableMapping

        db = mocker.MagicMock()
        db.load_runtime_config.return_value = {}
        db.load_column_exclusions.return_value = {}
        db.load_table_targets.return_value = [TableMapping(SOURCE_URI, TARGET_URI, "orders")]
        mocker.patch("main.connect_audit_db", return_value=db)

        from main import main
        main()


class TestEndToEndAnonymization:

    def test_emails_anonymized_in_written_dataframe(
        self, env, mock_delta, mock_auth, mock_db, analyzer, mocker,
        mock_classify,
    ):
        """Full chain: real Presidio engine, mocked storage + DB."""
        mocker.patch("main.build_engines", return_value=analyzer)
        mocker.patch("main.validate_residual_pii", return_value=0)
        _, mock_write = mock_delta

        from main import main
        main()

        written_df: pd.DataFrame = mock_write.call_args.args[0]
        for val in written_df["email"]:
            assert "example.com" not in val, f"Email not anonymized: {val!r}"
            assert "@" not in val, f"Email not fully masked: {val!r}"

    def test_score_column_untouched_in_written_dataframe(
        self, env, mock_delta, mock_auth, mock_db, analyzer, mocker,
        mock_classify,
    ):
        mocker.patch("main.build_engines", return_value=analyzer)
        mocker.patch("main.validate_residual_pii", return_value=0)
        _, mock_write = mock_delta

        from main import main
        main()

        written_df: pd.DataFrame = mock_write.call_args.args[0]
        assert list(written_df["score"]) == [10, 20]

    def test_entity_tokens_in_output(
        self, env, mock_delta, mock_auth, mock_db, analyzer, mocker,
        mock_classify,
    ):
        """After anonymization, columns should contain ENTITY_TYPE_N tokens."""
        mocker.patch("main.build_engines", return_value=analyzer)
        mocker.patch("main.validate_residual_pii", return_value=0)
        _, mock_write = mock_delta

        from main import main
        main()

        written_df: pd.DataFrame = mock_write.call_args.args[0]
        for val in written_df["email"]:
            assert "EMAIL_ADDRESS_" in val, f"Expected EMAIL_ADDRESS token, got: {val!r}"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Dynamic 1-to-1 table discovery via SOURCE_BASE_ABFSS_URI
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestDynamicDiscovery:

    @pytest.fixture()
    def base_env(self, monkeypatch):
        monkeypatch.setenv("SOURCE_BASE_ABFSS_URI", BASE_SOURCE_URI)
        monkeypatch.setenv("TARGET_BASE_ABFSS_URI", BASE_TARGET_URI)
        for opt in ("SOURCE_ABFSS_URI", "TARGET_ABFSS_URI", "DATABASE_URL",
                    "PURVIEW_ACCOUNT_NAME", "K_ANONYMITY_MIN", "QUASI_IDENTIFIER_COLS"):
            monkeypatch.delenv(opt, raising=False)

    def _two_mappings(self):
        from main import TableMapping
        return [
            TableMapping(f"{BASE_SOURCE_URI}/customers", f"{BASE_TARGET_URI}/customers", "customers"),
            TableMapping(f"{BASE_SOURCE_URI}/orders",    f"{BASE_TARGET_URI}/orders",    "orders"),
        ]

    def _std_mocks(self, mocker, mappings):
        mocker.patch("main.discover_table_mappings", return_value=mappings)
        mocker.patch("main.read_delta", return_value=RAW_DF.copy())
        mock_write = mocker.patch("main.write_delta")
        mocker.patch("main.connect_audit_db", return_value=None)
        mocker.patch("main.anonymize_dataframe", return_value=(ANON_DF.copy(), MOCK_STATS))
        return mock_write

    def test_discover_is_called(
        self, base_env, mock_auth, mock_engines, mock_validate,
        mock_classify, mock_hash, mocker,
    ):
        mock_discover = mocker.patch("main.discover_table_mappings", return_value=self._two_mappings())
        mocker.patch("main.read_delta", return_value=RAW_DF.copy())
        mocker.patch("main.write_delta")
        mocker.patch("main.connect_audit_db", return_value=None)
        mocker.patch("main.anonymize_dataframe", return_value=(ANON_DF.copy(), MOCK_STATS))

        from main import main
        main()
        mock_discover.assert_called_once_with(BASE_SOURCE_URI, BASE_TARGET_URI, sql_endpoint=None, sql_database=None)

    def test_write_called_once_per_table(
        self, base_env, mock_auth, mock_engines, mock_validate,
        mock_classify, mock_hash, mocker,
    ):
        mock_write = self._std_mocks(mocker, self._two_mappings())

        from main import main
        main()
        assert mock_write.call_count == 2

    def test_target_uris_match_source_table_names(
        self, base_env, mock_auth, mock_engines, mock_validate,
        mock_classify, mock_hash, mocker,
    ):
        mock_write = self._std_mocks(mocker, self._two_mappings())

        from main import main
        main()

        written_uris = {call.args[1] for call in mock_write.call_args_list}
        assert f"{BASE_TARGET_URI}/customers" in written_uris
        assert f"{BASE_TARGET_URI}/orders" in written_uris

    def test_empty_discovery_raises(
        self, base_env, mock_auth, mock_engines, mock_validate,
        mock_classify, mock_hash, mocker,
    ):
        mocker.patch("main.discover_table_mappings", return_value=[])
        mocker.patch("main.connect_audit_db", return_value=None)

        from main import main
        with pytest.raises(RuntimeError, match="No tables found"):
            main()

    def test_parallel_tables_use_process_workers(self, base_env, monkeypatch):
        import app.application.pipeline as service
        from main import TableMapping

        mappings = self._two_mappings()
        submitted = []
        executor_max_workers = []

        class _Future:
            def __init__(self, value):
                self._value = value

            def result(self):
                return self._value

        class _Executor:
            def __init__(self, max_workers):
                executor_max_workers.append(max_workers)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, func, config, mapping):
                submitted.append((func, mapping))
                return _Future({"table": mapping.table_name})

        monkeypatch.setattr(service, "connect_audit_db", lambda database_url: object())
        monkeypatch.setattr(service, "resolve_table_mappings", lambda config: mappings)
        monkeypatch.setattr(service, "ProcessPoolExecutor", _Executor)

        config = service.PipelineConfig(
            database_url=None,
            purview_account_name=None,
            k_anonymity_min=5,
            source_base_uri=BASE_SOURCE_URI,
            target_base_uri=BASE_TARGET_URI,
            max_table_workers=4,
        )

        result = service.run_pipeline(config)

        assert executor_max_workers == [2]
        assert [mapping.table_name for _, mapping in submitted] == ["customers", "orders"]
        assert all(func is service._run_table_worker for func, _ in submitted)
        assert result == [{"table": "customers"}, {"table": "orders"}]

    def test_table_worker_opens_its_own_audit_connection(self, monkeypatch):
        import app.application.pipeline as service
        from main import TableMapping

        config = service.PipelineConfig(
            database_url="postgresql://audit",
            purview_account_name=None,
            k_anonymity_min=5,
            source_base_uri=BASE_SOURCE_URI,
            target_base_uri=BASE_TARGET_URI,
        )
        mapping = TableMapping(f"{BASE_SOURCE_URI}/customers", f"{BASE_TARGET_URI}/customers", "customers")
        db = object()

        monkeypatch.setattr(service, "connect_audit_db", lambda database_url: db)

        calls = []

        def _run_table(config_arg, mapping_arg, db_arg):
            calls.append((config_arg, mapping_arg, db_arg))
            return {"table": mapping_arg.table_name}

        monkeypatch.setattr(service, "run_table", _run_table)

        assert service._run_table_worker(config, mapping) == {"table": "customers"}
        assert calls == [(config, mapping, db)]


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# discover_table_mappings â€” _delta_log filtering (unit-level, mocks ADLS)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestDiscoverTableMappingsFiltering:
    """Verify that only directories containing _delta_log are returned."""

    def _make_path_item(self, mocker, name: str, is_directory: bool):
        item = mocker.MagicMock()
        item.name = name
        item.is_directory = is_directory
        return item

    def _make_fs_client(self, mocker, path_items, delta_log_dirs):
        """Return a mock FileSystemClient whose get_paths yields path_items
        and whose get_directory_client().exists() returns True only for paths
        listed in delta_log_dirs."""
        fs_client = mocker.MagicMock()
        fs_client.get_paths.return_value = path_items

        def _dir_client(path):
            dc = mocker.MagicMock()
            dc.exists.return_value = path in delta_log_dirs
            return dc

        fs_client.get_directory_client.side_effect = _dir_client
        return fs_client

    def test_delta_tables_included(self, mocker):
        from app.infrastructure.repository import discover_table_mappings, DataLakeServiceClient as _orig
        items = [self._make_path_item(mocker, "Tables/customers", True)]
        fs_client = self._make_fs_client(mocker, items, {"Tables/customers/_delta_log"})
        svc = mocker.MagicMock()
        svc.get_file_system_client.return_value = fs_client
        mocker.patch("app.infrastructure.repository.DataLakeServiceClient", return_value=svc)
        mocker.patch("app.infrastructure.repository._credential_instance", return_value=mocker.MagicMock())

        result = discover_table_mappings(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/Src.Lakehouse",
            "abfss://ws@onelake.dfs.fabric.microsoft.com/Tgt.Lakehouse/Tables",
        )
        assert len(result) == 1
        assert result[0].table_name == "customers"

    def test_files_target_base_rejected(self, mocker):
        from app.infrastructure.repository import discover_table_mappings
        items = [self._make_path_item(mocker, "Tables/customers", True)]
        fs_client = self._make_fs_client(mocker, items, {"Tables/customers/_delta_log"})
        svc = mocker.MagicMock()
        svc.get_file_system_client.return_value = fs_client
        mocker.patch("app.infrastructure.repository.DataLakeServiceClient", return_value=svc)
        mocker.patch("app.infrastructure.repository._credential_instance", return_value=mocker.MagicMock())

        with pytest.raises(ValueError, match="Lakehouse Files"):
            discover_table_mappings(
                "abfss://ws@onelake.dfs.fabric.microsoft.com/Src.Lakehouse",
                "abfss://ws@onelake.dfs.fabric.microsoft.com/Tgt.Lakehouse/Files/anonymized",
            )

    def test_onelake_https_base_uri_is_supported(self, mocker):
        from app.infrastructure.repository import discover_table_mappings
        mocker.patch("app.infrastructure.repository._fabric_workspace_guid_for_name", return_value=None)
        mocker.patch("app.infrastructure.repository._fabric_item_display_name", return_value=None)
        items = [self._make_path_item(mocker, "f96c5a4c-7777-4fda-aeb9-eb239ed1731c/Tables/customers", True)]
        fs_client = self._make_fs_client(
            mocker,
            items,
            {"f96c5a4c-7777-4fda-aeb9-eb239ed1731c/Tables/customers/_delta_log"},
        )
        svc = mocker.MagicMock()
        svc.get_file_system_client.return_value = fs_client
        mocker.patch("app.infrastructure.repository.DataLakeServiceClient", return_value=svc)
        mocker.patch("app.infrastructure.repository._credential_instance", return_value=mocker.MagicMock())

        result = discover_table_mappings(
            "https://onelake.dfs.fabric.microsoft.com/ffb5e061-3824-486b-ab7c-aaef61221403/f96c5a4c-7777-4fda-aeb9-eb239ed1731c/Tables",
            "https://onelake.dfs.fabric.microsoft.com/ffb5e061-3824-486b-ab7c-aaef61221403/target-lakehouse-id/Tables",
        )

        assert len(result) == 1
        assert result[0].source_uri == (
            "abfss://ffb5e061-3824-486b-ab7c-aaef61221403@onelake.dfs.fabric.microsoft.com/"
            "f96c5a4c-7777-4fda-aeb9-eb239ed1731c/Tables/customers"
        )
        assert result[0].target_uri == (
            "abfss://ffb5e061-3824-486b-ab7c-aaef61221403@onelake.dfs.fabric.microsoft.com/"
            "target-lakehouse-id/Tables/customers"
        )

    def test_onelake_item_id_path_resolves_to_lakehouse_name(self, mocker):
        from app.infrastructure.repository import discover_table_mappings
        items = [self._make_path_item(mocker, "SourceLakehouse.Lakehouse/Tables/customers", True)]
        fs_client = self._make_fs_client(
            mocker,
            items,
            {"SourceLakehouse.Lakehouse/Tables/customers/_delta_log"},
        )
        svc = mocker.MagicMock()
        svc.get_file_system_client.return_value = fs_client
        mocker.patch("app.infrastructure.repository.DataLakeServiceClient", return_value=svc)
        mocker.patch("app.infrastructure.repository._credential_instance", return_value=mocker.MagicMock())
        mocker.patch(
            "app.infrastructure.repository._fabric_workspace_guid_for_name",
            return_value="ffb5e061-3824-486b-ab7c-aaef61221403",
        )
        mocker.patch("app.infrastructure.repository._fabric_item_display_name", return_value="SourceLakehouse")

        result = discover_table_mappings(
            "abfss://VIBECODING@onelake.dfs.fabric.microsoft.com/f96c5a4c-7777-4fda-aeb9-eb239ed1731c/Tables",
            "abfss://VIBECODING@onelake.dfs.fabric.microsoft.com/DATALAKE.Lakehouse/Tables",
        )

        fs_client.get_paths.assert_called_once_with(
            path="SourceLakehouse.Lakehouse/Tables",
            recursive=False,
        )
        assert len(result) == 1
        assert result[0].source_uri == (
            "abfss://VIBECODING@onelake.dfs.fabric.microsoft.com/"
            "SourceLakehouse.Lakehouse/Tables/customers"
        )

    def test_guid_workspace_keeps_guid_lakehouse_path(self, mocker):
        from app.infrastructure.repository import discover_table_mappings
        items = [self._make_path_item(mocker, "f96c5a4c-7777-4fda-aeb9-eb239ed1731c/Tables/customers", True)]
        fs_client = self._make_fs_client(
            mocker,
            items,
            {"f96c5a4c-7777-4fda-aeb9-eb239ed1731c/Tables/customers/_delta_log"},
        )
        svc = mocker.MagicMock()
        svc.get_file_system_client.return_value = fs_client
        mocker.patch("app.infrastructure.repository.DataLakeServiceClient", return_value=svc)
        mocker.patch("app.infrastructure.repository._credential_instance", return_value=mocker.MagicMock())
        ws_resolver = mocker.patch("app.infrastructure.repository._fabric_workspace_guid_for_name", return_value="some-guid")
        resolver = mocker.patch("app.infrastructure.repository._fabric_item_display_name", return_value="SourceLakehouse")

        result = discover_table_mappings(
            "abfss://ffb5e061-3824-486b-ab7c-aaef61221403@onelake.dfs.fabric.microsoft.com/f96c5a4c-7777-4fda-aeb9-eb239ed1731c/Tables",
            "abfss://VIBECODING@onelake.dfs.fabric.microsoft.com/DATALAKE.Lakehouse/Tables",
        )

        ws_resolver.assert_not_called()
        resolver.assert_not_called()
        fs_client.get_paths.assert_any_call(
            path="f96c5a4c-7777-4fda-aeb9-eb239ed1731c/Tables",
            recursive=False,
        )
        assert len(result) == 1

    def test_friendly_workspace_guid_lakehouse_api_success_resolves_name(self, mocker):
        """Happy path: friendly workspace + GUID lakehouse, Fabric API resolves the name."""
        from app.infrastructure.repository import discover_table_mappings
        lakehouse_guid = "f96c5a4c-7777-4fda-aeb9-eb239ed1731c"
        workspace_guid = "ffb5e061-3824-486b-ab7c-aaef61221403"
        items = [self._make_path_item(mocker, "SourceLakehouse.Lakehouse/Tables/customers", True)]
        fs_client = self._make_fs_client(
            mocker,
            items,
            {"SourceLakehouse.Lakehouse/Tables/customers/_delta_log"},
        )
        svc = mocker.MagicMock()
        svc.get_file_system_client.return_value = fs_client
        mocker.patch("app.infrastructure.repository.DataLakeServiceClient", return_value=svc)
        mocker.patch("app.infrastructure.repository._credential_instance", return_value=mocker.MagicMock())
        mocker.patch(
            "app.infrastructure.repository._fabric_workspace_guid_for_name",
            return_value=workspace_guid,
        )
        mocker.patch(
            "app.infrastructure.repository._fabric_item_display_name",
            return_value="SourceLakehouse",
        )

        result = discover_table_mappings(
            f"abfss://MyWorkspace@onelake.dfs.fabric.microsoft.com/{lakehouse_guid}/Tables",
            "abfss://MyWorkspace@onelake.dfs.fabric.microsoft.com/DATALAKE.Lakehouse/Tables",
        )

        fs_client.get_paths.assert_called_once_with(
            path="SourceLakehouse.Lakehouse/Tables",
            recursive=False,
        )
        assert len(result) == 1
        assert "SourceLakehouse.Lakehouse" in result[0].source_uri

    def test_friendly_workspace_guid_lakehouse_api_failure_falls_back_to_root(self, mocker):
        """Regression: when Fabric API is inaccessible (SP has only storage RBAC),
        fall back to workspace-root scan instead of failing with FriendlyNameSupportDisabled."""
        from app.infrastructure.repository import discover_table_mappings
        lakehouse_guid = "f96c5a4c-7777-4fda-aeb9-eb239ed1731c"
        # The workspace root listing returns lakehouses by friendly name.
        items = [self._make_path_item(mocker, "SourceLakehouse.Lakehouse/Tables/customers", True)]
        fs_client = self._make_fs_client(
            mocker,
            items,
            {"SourceLakehouse.Lakehouse/Tables/customers/_delta_log"},
        )
        svc = mocker.MagicMock()
        svc.get_file_system_client.return_value = fs_client
        mocker.patch("app.infrastructure.repository.DataLakeServiceClient", return_value=svc)
        mocker.patch("app.infrastructure.repository._credential_instance", return_value=mocker.MagicMock())
        # Both Fabric API passes fail.
        mocker.patch(
            "app.infrastructure.repository._fabric_item_display_name",
            side_effect=Exception("403 Forbidden"),
        )
        mocker.patch("app.infrastructure.repository._fabric_workspace_guid_for_name", return_value=None)

        result = discover_table_mappings(
            f"abfss://MyWorkspace@onelake.dfs.fabric.microsoft.com/{lakehouse_guid}/Tables",
            "abfss://MyWorkspace@onelake.dfs.fabric.microsoft.com/DATALAKE.Lakehouse/Tables",
        )

        # Must have scanned workspace root (path=""), not the GUID path.
        fs_client.get_paths.assert_called_once_with(path="", recursive=False)
        assert len(result) == 1

    def test_recursive_delta_log_discovery_when_immediate_listing_has_no_tables(self, mocker):
        from app.infrastructure.repository import discover_table_mappings
        delta_log = self._make_path_item(mocker, "Lakehouse.Lakehouse/Tables/customers/_delta_log", True)
        fs_client = self._make_fs_client(mocker, [], set())
        fs_client.get_paths.side_effect = [
            [],
            [delta_log],
        ]
        svc = mocker.MagicMock()
        svc.get_file_system_client.return_value = fs_client
        mocker.patch("app.infrastructure.repository.DataLakeServiceClient", return_value=svc)
        mocker.patch("app.infrastructure.repository._credential_instance", return_value=mocker.MagicMock())

        result = discover_table_mappings(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/Lakehouse.Lakehouse/Tables",
            "abfss://ws@onelake.dfs.fabric.microsoft.com/Tgt.Lakehouse/Tables",
        )

        fs_client.get_paths.assert_any_call(path="Lakehouse.Lakehouse/Tables", recursive=True)
        assert len(result) == 1
        assert result[0].table_name == "customers"

    def test_recursive_delta_log_discovery_when_immediate_listing_is_large(self, mocker):
        from app.infrastructure.repository import discover_table_mappings

        items = [self._make_path_item(mocker, f"Tables/folder_{idx}", True) for idx in range(25)]
        delta_log = self._make_path_item(mocker, "Tables/customers/_delta_log", True)
        fs_client = self._make_fs_client(mocker, items, set())
        fs_client.get_paths.side_effect = [
            items,
            [delta_log],
        ]
        svc = mocker.MagicMock()
        svc.get_file_system_client.return_value = fs_client
        mocker.patch("app.infrastructure.repository.DataLakeServiceClient", return_value=svc)
        mocker.patch("app.infrastructure.repository._credential_instance", return_value=mocker.MagicMock())

        result = discover_table_mappings(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/Src.Lakehouse/Tables",
            "abfss://ws@onelake.dfs.fabric.microsoft.com/Tgt.Lakehouse/Tables",
        )

        fs_client.get_paths.assert_any_call(path="Src.Lakehouse/Tables", recursive=True)
        fs_client.get_directory_client.assert_not_called()
        assert len(result) == 1
        assert result[0].table_name == "customers"

    def test_non_delta_directories_excluded(self, mocker):
        from app.infrastructure.repository import discover_table_mappings
        items = [
            self._make_path_item(mocker, "Tables/customers", True),
            self._make_path_item(mocker, "Tables/_schemas", True),   # schema folder
            self._make_path_item(mocker, "Tables/tmp_import", True), # helper folder
        ]
        fs_client = self._make_fs_client(mocker, items, {"Tables/customers/_delta_log"})
        svc = mocker.MagicMock()
        svc.get_file_system_client.return_value = fs_client
        mocker.patch("app.infrastructure.repository.DataLakeServiceClient", return_value=svc)
        mocker.patch("app.infrastructure.repository._credential_instance", return_value=mocker.MagicMock())

        result = discover_table_mappings(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/Src.Lakehouse",
            "abfss://ws@onelake.dfs.fabric.microsoft.com/Tgt.Lakehouse/Tables",
        )
        assert len(result) == 1
        assert result[0].table_name == "customers"

    def test_files_excluded(self, mocker):
        from app.infrastructure.repository import discover_table_mappings
        items = [
            self._make_path_item(mocker, "Tables/customers", True),
            self._make_path_item(mocker, "Tables/readme.md", False),  # file, not dir
        ]
        fs_client = self._make_fs_client(mocker, items, {"Tables/customers/_delta_log"})
        svc = mocker.MagicMock()
        svc.get_file_system_client.return_value = fs_client
        mocker.patch("app.infrastructure.repository.DataLakeServiceClient", return_value=svc)
        mocker.patch("app.infrastructure.repository._credential_instance", return_value=mocker.MagicMock())

        result = discover_table_mappings(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/Src.Lakehouse",
            "abfss://ws@onelake.dfs.fabric.microsoft.com/Tgt.Lakehouse/Tables",
        )
        assert len(result) == 1

    def _adls_setup(self, mocker, delta_names):
        """Shared ADLS mock: given list of Delta table names, wire up the ADLS mocks."""
        items = [self._make_path_item(mocker, f"Tables/{n}", True) for n in delta_names]
        delta_log_dirs = {f"Tables/{n}/_delta_log" for n in delta_names}
        fs_client = self._make_fs_client(mocker, items, delta_log_dirs)
        svc = mocker.MagicMock()
        svc.get_file_system_client.return_value = fs_client
        mocker.patch("app.infrastructure.repository.DataLakeServiceClient", return_value=svc)
        mocker.patch("app.infrastructure.repository._credential_instance", return_value=mocker.MagicMock())

    def test_sql_shortcuts_included(self, mocker):
        """SQL tables not present in ADLS get read_mode='sql' mappings; schema
        is preserved in both source and target paths so a schema-enabled
        Fabric lakehouse registers the destination table."""
        from app.infrastructure.repository import discover_table_mappings
        self._adls_setup(mocker, ["customers"])
        mocker.patch(
            "app.infrastructure.repository._discover_sql_table_names",
            return_value=[("dbo", "customers"), ("dbo", "customer_view")],
        )

        result = discover_table_mappings(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/Src.Lakehouse",
            "abfss://ws@onelake.dfs.fabric.microsoft.com/Tgt.Lakehouse/Tables",
            sql_endpoint="ws.datawarehouse.fabric.microsoft.com",
            sql_database="SourceLakehouse",
        )

        assert len(result) == 2
        delta_m = next(m for m in result if m.table_name == "customers")
        sql_m = next(m for m in result if m.table_name == "customer_view")
        assert delta_m.read_mode == "delta"
        assert sql_m.read_mode == "sql"
        assert sql_m.source_uri.startswith("sql://")
        assert sql_m.schema == "dbo"
        assert sql_m.target_uri.endswith("/dbo/customer_view")

    def test_delta_table_not_duplicated_by_sql(self, mocker):
        """A schema-less Delta source (Tables/<name>) is the same physical
        table as the SQL spec (dbo.<name>) â€” dedup keeps the Delta side."""
        from app.infrastructure.repository import discover_table_mappings
        self._adls_setup(mocker, ["customers", "orders"])
        mocker.patch(
            "app.infrastructure.repository._discover_sql_table_names",
            return_value=[("dbo", "customers"), ("dbo", "orders")],
        )

        result = discover_table_mappings(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/Src.Lakehouse",
            "abfss://ws@onelake.dfs.fabric.microsoft.com/Tgt.Lakehouse/Tables",
            sql_endpoint="ws.datawarehouse.fabric.microsoft.com",
            sql_database="SourceLakehouse",
        )

        assert len(result) == 2
        assert all(m.read_mode == "delta" for m in result)

    def test_sql_discovery_failure_is_non_fatal(self, mocker):
        """A SQL endpoint error is logged and does not abort ADLS discovery."""
        from app.infrastructure.repository import discover_table_mappings
        self._adls_setup(mocker, ["customers"])
        mocker.patch(
            "app.infrastructure.repository._discover_sql_table_names",
            side_effect=Exception("connection refused"),
        )

        result = discover_table_mappings(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/Src.Lakehouse",
            "abfss://ws@onelake.dfs.fabric.microsoft.com/Tgt.Lakehouse/Tables",
            sql_endpoint="ws.datawarehouse.fabric.microsoft.com",
            sql_database="SourceLakehouse",
        )

        assert len(result) == 1
        assert result[0].table_name == "customers"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SQL shortcut read routing
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestSQLShortcutRouting:
    """Verify that run_table reads SQL mappings via read_sql_table, not Delta read."""

    @pytest.fixture()
    def sql_env(self, monkeypatch, mocker):
        """Env fixture with SQL_ENDPOINT_URL and SQL_DATABASE set."""
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("SQL_ENDPOINT_URL", "ws.datawarehouse.fabric.microsoft.com")
        monkeypatch.setenv("SQL_DATABASE", "SourceLakehouse")
        for opt in ("DATABASE_URL", "PURVIEW_ACCOUNT_NAME", "K_ANONYMITY_MIN", "QUASI_IDENTIFIER_COLS"):
            monkeypatch.delenv(opt, raising=False)
        from main import TableMapping
        mocker.patch(
            "main.discover_table_mappings",
            return_value=[TableMapping(
                source_uri="sql://ws.datawarehouse.fabric.microsoft.com/SourceLakehouse/shortcuts",
                target_uri=TARGET_URI,
                table_name="shortcuts",
                read_mode="sql",
            )],
        )

    def test_read_sql_table_called_for_sql_mapping(
        self, sql_env, mock_auth, mock_anonymize, mock_engines, mock_validate,
        mock_classify, mock_hash, mocker,
    ):
        mock_sql_read = mocker.patch("main.read_sql_table", return_value=RAW_DF.copy())
        mocker.patch("main.write_delta")
        mocker.patch("main.connect_audit_db", return_value=None)

        from main import main
        main()

        mock_sql_read.assert_called_once()
        assert mock_sql_read.call_args.args[0] == "shortcuts"

    def test_delta_table_not_called_for_sql_mapping(
        self, sql_env, mock_auth, mock_anonymize, mock_engines, mock_validate,
        mock_classify, mock_hash, mocker,
    ):
        mocker.patch("main.read_sql_table", return_value=RAW_DF.copy())
        mocker.patch("main.write_delta")
        mocker.patch("main.connect_audit_db", return_value=None)
        mock_delta_read = mocker.patch("main.read_delta")

        from main import main
        main()

        mock_delta_read.assert_not_called()
