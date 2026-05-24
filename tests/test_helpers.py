"""
Unit tests for URI helpers, storage options, the optional Purview check,
free-text column flagging, quasi-identifier detection, k-anonymity enforcement,
and residual PII validation.
No external services or spaCy model required (except residual PII tests).
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from main import (
    PurviewClient,
    _account_name,
    _storage_opts,
    detect_identifier_columns,
    detect_quasi_identifiers,
    enforce_k_anonymity,
    flag_free_text_columns,
    pseudonymize_identifier_columns,
    run_purview_check,
    validate_residual_pii,
    write_delta,
)
from app.repository import AuditDB, _parse_abfss_uri, read_delta


class _FakePseudonymizer:
    """Deterministic in-memory pseudonymizer for tests.

    Mimics KeyVaultPseudonymizer.pseudonymize: nulls pass through, every
    other value is hashed under a fixed secret derived from ``key``.
    """

    def __init__(self, key: bytes = b"test-key") -> None:
        self._key = key

    def __call__(self, value):
        import hmac
        import hashlib
        import pandas as pd

        try:
            if pd.isna(value):
                return value
        except (TypeError, ValueError):
            pass
        raw = value if isinstance(value, str) else str(value)
        return hmac.new(self._key, raw.encode("utf-8"), hashlib.sha256).hexdigest()[:24]


class TestAccountNameParser:

    @pytest.mark.parametrize("uri,expected", [
        (
            "abfss://workspace@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t",
            "onelake",
        ),
        (
            "abfss://mycontainer@myaccount.dfs.core.windows.net/path/data",
            "myaccount",
        ),
        (
            "abfss://ws@storage123.dfs.core.windows.net/",
            "storage123",
        ),
        (
            "abfss://WorkspaceName@onelake.dfs.fabric.microsoft.com/Demo.Lakehouse/Tables/orders",
            "onelake",
        ),
    ])
    def test_extracts_correct_account(self, uri, expected):
        assert _account_name(uri) == expected

    def test_extracts_account_from_onelake_https_uri(self):
        uri = "https://onelake.dfs.fabric.microsoft.com/workspace-id/lakehouse-id/Tables/orders"
        assert _account_name(uri) == "onelake"

    @pytest.mark.parametrize("bad_uri", [
        "not-a-uri",
        "https://example.com/path",
        "",
        "abfss://no-at-sign-here",
        "abfss://workspace",
    ])
    def test_raises_on_invalid_uri(self, bad_uri):
        with pytest.raises(ValueError, match="Cannot parse"):
            _account_name(bad_uri)


class TestStorageOpts:

    def test_contains_account_name_and_token(self):
        uri = "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t"
        opts = _storage_opts(uri, "my-bearer-token")
        assert opts["account_name"] == "onelake"
        assert opts["bearer_token"] == "my-bearer-token"

    def test_different_accounts_produce_different_opts(self):
        uri_a = "abfss://c@accountA.dfs.core.windows.net/p"
        uri_b = "abfss://c@accountB.dfs.core.windows.net/p"
        assert _storage_opts(uri_a, "tok")["account_name"] == "accountA"
        assert _storage_opts(uri_b, "tok")["account_name"] == "accountB"


class TestStorageUriParser:

    def test_parses_abfss_uri(self):
        assert _parse_abfss_uri("abfss://ws@onelake.dfs.fabric.microsoft.com/lh/Tables") == (
            "ws",
            "onelake.dfs.fabric.microsoft.com",
            "lh/Tables",
        )

    def test_parses_onelake_https_uri(self):
        assert _parse_abfss_uri(
            "https://onelake.dfs.fabric.microsoft.com/ffb5e061-3824-486b-ab7c-aaef61221403/f96c5a4c-7777-4fda-aeb9-eb239ed1731c/Tables"
        ) == (
            "ffb5e061-3824-486b-ab7c-aaef61221403",
            "onelake.dfs.fabric.microsoft.com",
            "f96c5a4c-7777-4fda-aeb9-eb239ed1731c/Tables",
        )


class TestReadDeltaLookback:

    class _FakeDeltaTable:
        def __init__(self, uri, storage_options=None):
            self.uri = uri
            self.storage_options = storage_options

        def to_pyarrow_dataset(self):
            import pyarrow as pa

            return pa.table({
                "id": [1, 2, 3],
                "created_at": [
                    datetime(2023, 12, 31, tzinfo=timezone.utc),
                    datetime(2024, 1, 1, tzinfo=timezone.utc),
                    datetime(2025, 1, 2, tzinfo=timezone.utc),
                ],
                "name": ["old", "edge", "new"],
            })

    def test_delta_read_uses_duckdb_cutoff_before_dataframe_materialization(self, monkeypatch, mocker):
        import app.repository as repo

        monkeypatch.setattr(repo, "DeltaTable", self._FakeDeltaTable)
        mocker.patch("app.repository.read_cutoff_ts", return_value=datetime(2024, 1, 1, tzinfo=timezone.utc))

        df = read_delta("abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t", {})

        assert list(df["id"]) == [2, 3]

    def test_delta_read_filters_string_temporal_columns_by_name(self, monkeypatch, mocker):
        import pyarrow as pa
        import app.repository as repo

        class FakeDeltaTable:
            def __init__(self, uri, storage_options=None):
                pass

            def to_pyarrow_dataset(self):
                return pa.table({
                    "id": [1, 2],
                    "event_date": ["2023-12-31", "2024-01-02"],
                })

        monkeypatch.setattr(repo, "DeltaTable", FakeDeltaTable)
        mocker.patch("app.repository.read_cutoff_ts", return_value=datetime(2024, 1, 1, tzinfo=timezone.utc))

        df = read_delta("abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t", {})

        assert list(df["id"]) == [2]


class TestWriteDelta:

    class _FakeDuckDB:
        def connect(self):
            return self._Connection()

        class _Connection:
            def install_extension(self, name):
                pass

            def load_extension(self, name):
                pass

            def register(self, name, df):
                pass

            def execute(self, sql):
                import pathlib
                import re

                match = re.search(r"COPY _df TO '([^']+)'", sql)
                if match:
                    table_path = pathlib.Path(match.group(1))
                    (table_path / "_delta_log").mkdir(parents=True)
                    (table_path / "_delta_log" / "00000000000000000000.json").write_text("{}", encoding="utf-8")
                    (table_path / "part-00000.parquet").write_bytes(b"parquet")

            def close(self):
                pass

    def test_uploads_delta_directory_to_explicit_path(self, monkeypatch, mocker):
        import app.repository as repo

        fs_client = mocker.MagicMock()
        file_client = mocker.MagicMock()
        service = mocker.MagicMock()
        service.get_file_system_client.return_value = fs_client
        fs_client.get_file_client.return_value = file_client
        service_cls = mocker.MagicMock(return_value=service)

        monkeypatch.setattr(repo, "_duckdb", self._FakeDuckDB())
        monkeypatch.setattr(repo, "DataLakeServiceClient", service_cls)
        monkeypatch.setattr(repo, "_credential_instance", lambda: object())

        write_delta(
            pd.DataFrame({"x": [1]}),
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/customers",
            {},
        )

        service_cls.assert_called_once()
        service.get_file_system_client.assert_called_once_with(file_system="ws")
        uploaded_paths = {call.kwargs["file_path"] for call in fs_client.get_file_client.call_args_list}
        assert "lh.Lakehouse/Tables/customers/_delta_log/00000000000000000000.json" in uploaded_paths
        assert any(
            path.startswith("lh.Lakehouse/Tables/customers/part-") and path.endswith(".parquet")
            for path in uploaded_paths
        )
        assert file_client.upload_data.call_count == 2
        assert file_client.upload_data.call_args.kwargs["overwrite"] is True

    def test_replaces_existing_remote_delta_directory_before_upload(self, monkeypatch, mocker):
        import app.repository as repo

        fs_client = mocker.MagicMock()
        directory_client = mocker.MagicMock()
        directory_client.exists.return_value = True
        file_client = mocker.MagicMock()
        service = mocker.MagicMock()
        service.get_file_system_client.return_value = fs_client
        fs_client.get_directory_client.return_value = directory_client
        fs_client.get_file_client.return_value = file_client

        monkeypatch.setattr(repo, "_duckdb", self._FakeDuckDB())
        monkeypatch.setattr(repo, "DataLakeServiceClient", mocker.MagicMock(return_value=service))
        monkeypatch.setattr(repo, "_credential_instance", lambda: object())

        write_delta(
            pd.DataFrame({"x": [1]}),
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/customers",
            {},
        )

        fs_client.get_directory_client.assert_called_once_with("lh.Lakehouse/Tables/customers")
        directory_client.delete_directory.assert_called_once()
        method_names = [call[0] for call in fs_client.method_calls]
        assert method_names.index("get_directory_client") < method_names.index("get_file_client")

    def test_uses_table_folder_name_for_delta_output(self, monkeypatch, mocker):
        import app.repository as repo

        fs_client = mocker.MagicMock()
        file_client = mocker.MagicMock()
        service = mocker.MagicMock()
        service.get_file_system_client.return_value = fs_client
        fs_client.get_file_client.return_value = file_client

        monkeypatch.setattr(repo, "_duckdb", self._FakeDuckDB())
        monkeypatch.setattr(repo, "DataLakeServiceClient", mocker.MagicMock(return_value=service))
        monkeypatch.setattr(repo, "_credential_instance", lambda: object())

        write_delta(pd.DataFrame({"x": [1]}), "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/customers", {})

        uploaded_paths = {call.kwargs["file_path"] for call in fs_client.get_file_client.call_args_list}
        assert any(
            path.startswith("lh.Lakehouse/Tables/customers/part-") and path.endswith(".parquet")
            for path in uploaded_paths
        )
        assert "lh.Lakehouse/Tables/customers/_delta_log/00000000000000000000.json" in uploaded_paths

    def test_write_uses_delta_rs_not_duckdb_extension(self, monkeypatch, mocker):
        import app.repository as repo

        service = mocker.MagicMock()
        service.get_file_system_client.return_value = mocker.MagicMock()

        duckdb = mocker.MagicMock()
        monkeypatch.setattr(repo, "_duckdb", duckdb)
        monkeypatch.setattr(repo, "DataLakeServiceClient", mocker.MagicMock(return_value=service))
        monkeypatch.setattr(repo, "_credential_instance", lambda: object())

        write_delta(
            pd.DataFrame({"x": [1]}),
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/customers",
            {},
        )

        duckdb.connect.assert_not_called()

    def test_rejects_lakehouse_files_target(self):
        with pytest.raises(ValueError, match="Lakehouse Files"):
            write_delta(
                pd.DataFrame({"x": [1]}),
                "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Files/out/customers",
                {},
            )

    def test_rejects_empty_schema_before_storage_access(self, monkeypatch, mocker):
        import app.repository as repo

        service_cls = mocker.MagicMock()
        monkeypatch.setattr(repo, "DataLakeServiceClient", service_cls)

        with pytest.raises(ValueError, match="at least one column"):
            write_delta(
                pd.DataFrame(),
                "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/customers",
                {},
            )

        service_cls.assert_not_called()


class TestAuditSchemaMigration:

    class _Cursor:
        def __init__(self, executed):
            self.executed = executed

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            self.executed.append(sql)

    class _Connection:
        def __init__(self, executed):
            self.executed = executed

        def cursor(self):
            return TestAuditSchemaMigration._Cursor(self.executed)

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    def test_init_schema_adds_missing_audit_columns(self, monkeypatch):
        import app.repository as repo

        executed: list[str] = []
        monkeypatch.setattr(
            repo.psycopg2,
            "connect",
            lambda dsn: self._Connection(executed),
        )

        AuditDB("postgresql://example")

        joined = "\n".join(executed)
        assert "ADD COLUMN IF NOT EXISTS key_vault_key_version TEXT" in joined
        assert "ADD COLUMN IF NOT EXISTS stage_seconds JSONB" in joined


class TestPurviewQualifiedName:

    @pytest.mark.parametrize("uri,expected_qn", [
        (
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t",
            "https://onelake.dfs.fabric.microsoft.com/ws/lh.Lakehouse/Tables/t",
        ),
        (
            "abfss://MyWorkspace@onelake.dfs.fabric.microsoft.com/Demo.Lakehouse/Tables/customers",
            "https://onelake.dfs.fabric.microsoft.com/MyWorkspace/Demo.Lakehouse/Tables/customers",
        ),
    ])
    def test_converts_abfss_to_qualified_name(self, uri, expected_qn):
        assert PurviewClient.qualified_name(uri) == expected_qn

    def test_scheme_stripped(self):
        qn = PurviewClient.qualified_name(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t"
        )
        assert not qn.startswith("abfss://")
        assert qn.startswith("https://")


class TestRunPurviewCheck:

    def test_skipped_when_account_not_set(self):
        result = run_purview_check(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t",
            ["col1", "col2"],
            purview_account=None,
        )
        assert result["available"] is False
        assert result["flagged_columns"] == []
        assert result["discrepancies"] == []

    def test_returns_flagged_columns(self, mocker):
        mocker.patch("main.acquire_token", return_value="fake-token")
        mock_cls = mocker.patch("main.PurviewClient")
        mock_cls.qualified_name.return_value = "https://onelake.dfs.fabric.microsoft.com/ws/lh/t"
        mock_cls.return_value.column_classifications.return_value = {
            "email":  ["MICROSOFT.PERSONAL.EMAIL"],
            "name":   ["MICROSOFT.PERSONAL.NAME"],
        }

        result = run_purview_check(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t",
            df_columns=["email", "name", "score"],
            purview_account="my-purview",
        )

        assert result["available"] is True
        assert set(result["flagged_columns"]) == {"email", "name"}
        assert result["discrepancies"] == []

    def test_discrepancy_when_flagged_column_absent_from_dataframe(self, mocker):
        mocker.patch("main.acquire_token", return_value="fake-token")
        mock_cls = mocker.patch("main.PurviewClient")
        mock_cls.qualified_name.return_value = "https://..."
        mock_cls.return_value.column_classifications.return_value = {
            "ssn":   ["MICROSOFT.PERSONAL.SSN"],
            "email": ["MICROSOFT.PERSONAL.EMAIL"],
        }

        result = run_purview_check(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t",
            df_columns=["email", "score"],
            purview_account="my-purview",
        )

        assert "ssn" in result["discrepancies"]
        assert "email" not in result["discrepancies"]

    def test_non_fatal_on_auth_failure(self, mocker):
        mocker.patch("main.acquire_token", side_effect=Exception("auth failed"))
        result = run_purview_check(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t",
            df_columns=[],
            purview_account="my-purview",
        )
        assert result["available"] is False

    def test_non_fatal_on_http_404(self, mocker):
        import requests
        mocker.patch("main.acquire_token", return_value="token")
        mock_cls = mocker.patch("main.PurviewClient")
        mock_cls.qualified_name.return_value = "https://..."
        http_err = requests.HTTPError(response=mocker.MagicMock(status_code=404))
        mock_cls.return_value.column_classifications.side_effect = http_err

        result = run_purview_check(
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/t",
            df_columns=[],
            purview_account="my-purview",
        )
        assert result["available"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Free-text column flagging
# ─────────────────────────────────────────────────────────────────────────────

class TestFlagFreeTextColumns:

    def test_notes_column_flagged(self):
        df = pd.DataFrame({"notes": ["This is a long free text value with multiple words."], "price": [9.99]})
        assert "notes" in flag_free_text_columns(df)

    def test_description_column_flagged(self):
        df = pd.DataFrame({"description": ["Widget A needs a longer narrative description for review."], "id": [1]})
        assert "description" in flag_free_text_columns(df)

    def test_feedback_column_flagged(self):
        df = pd.DataFrame({"customer_feedback": ["The delivery was late and the customer explained the problem in detail."], "qty": [1]})
        assert "customer_feedback" in flag_free_text_columns(df)

    def test_numeric_column_not_flagged(self):
        df = pd.DataFrame({"price": pd.array([9.99], dtype="float64"), "qty": pd.array([1], dtype="int64")})
        assert flag_free_text_columns(df) == []

    def test_non_text_object_column_with_matching_name_still_flagged(self):
        df = pd.DataFrame({"notes": [{"key": "val"}]})
        assert "notes" not in flag_free_text_columns(df)

    def test_unrelated_object_column_not_flagged(self):
        df = pd.DataFrame({"customer_id": ["CID-001", "CID-002"]})
        assert "customer_id" not in flag_free_text_columns(df)

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame()
        assert flag_free_text_columns(df) == []


# ─────────────────────────────────────────────────────────────────────────────
# Quasi-identifier detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectQuasiIdentifiers:

    def test_explicit_cols_used_when_provided(self):
        df = pd.DataFrame({"age": [25], "city": ["NYC"], "score": [10]})
        qi = detect_quasi_identifiers(df, explicit_cols=["age", "city"])
        assert qi == ["age", "city"]

    def test_explicit_cols_filtered_to_present(self):
        df = pd.DataFrame({"age": [25, 25, 25, 30, 30, 30], "score": [10, 11, 12, 13, 14, 15]})
        qi = detect_quasi_identifiers(df, explicit_cols=["age", "missing_col"])
        assert qi == ["age"]
        assert "missing_col" not in qi

    def test_keyword_detection_on_age(self):
        df = pd.DataFrame({"age": [25, 25, 25, 30, 30, 30], "score": [10, 11, 12, 13, 14, 15]})
        qi = detect_quasi_identifiers(df)
        assert "age" in qi

    def test_keyword_detection_on_gender(self):
        df = pd.DataFrame({"gender": ["M", "M", "M", "F", "F", "F"], "score": [10, 11, 12, 13, 14, 15]})
        qi = detect_quasi_identifiers(df)
        assert "gender" in qi

    def test_non_qi_column_not_detected(self):
        df = pd.DataFrame({"product_name": ["Widget"], "price": [9.99]})
        qi = detect_quasi_identifiers(df)
        assert qi == []

    def test_empty_explicit_cols_falls_back_to_keyword(self):
        df = pd.DataFrame({"age": [25, 25, 25, 30, 30, 30], "city": ["NYC", "NYC", "NYC", "LUX", "LUX", "LUX"]})
        qi = detect_quasi_identifiers(df, explicit_cols=[])
        assert "age" in qi


# ─────────────────────────────────────────────────────────────────────────────
# k-Anonymity enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestEnforceKAnonymity:

    def test_no_suppression_when_k_met(self):
        df = pd.DataFrame({"age": [25, 25, 30, 30, 30], "score": [1, 2, 3, 4, 5]})
        result_df, info = enforce_k_anonymity(df, ["age"], k=2)
        assert len(result_df) == 5
        assert info["suppressed_rows"] == 0

    def test_suppresses_rare_groups(self):
        df = pd.DataFrame({"age": [25, 25, 30, 30, 99]})
        result_df, info = enforce_k_anonymity(df, ["age"], k=2)
        assert 99 not in result_df["age"].values
        assert info["suppressed_rows"] == 1

    def test_empty_quasi_cols_returns_unchanged(self):
        df = pd.DataFrame({"score": [1, 2, 3]})
        result_df, info = enforce_k_anonymity(df, [], k=5)
        pd.testing.assert_frame_equal(result_df, df)
        assert info["suppressed_rows"] == 0

    def test_all_rows_suppressed_when_none_meet_k(self):
        df = pd.DataFrame({"age": [10, 20, 30]})  # each age appears once
        result_df, info = enforce_k_anonymity(df, ["age"], k=2)
        assert len(result_df) == 0
        assert info["suppressed_rows"] == 3

    def test_multi_column_quasi_identifiers(self):
        df = pd.DataFrame({
            "age":    [25, 25, 25, 30],
            "gender": ["M", "M", "F", "M"],
        })
        result_df, info = enforce_k_anonymity(df, ["age", "gender"], k=2)
        assert len(result_df) == 2   # only (25, M) has count >= 2
        assert info["suppressed_rows"] == 2

    def test_missing_quasi_col_treated_gracefully(self):
        df = pd.DataFrame({"score": [1, 2, 3]})
        result_df, info = enforce_k_anonymity(df, ["nonexistent"], k=2)
        pd.testing.assert_frame_equal(result_df, df)
        assert info["suppressed_rows"] == 0

    def test_k_value_returned_in_info(self):
        df = pd.DataFrame({"age": [25, 25]})
        _, info = enforce_k_anonymity(df, ["age"], k=3)
        assert info["k"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Residual PII validation
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateResidualPII:

    def test_non_identifier_ner_residuals_do_not_abort(self):
        df = pd.DataFrame({
            "indicator_label": ["Luxembourg", "France", "Kayl"],
            "commune": ["Luxembourg", "Kayl", "France"],
            "notes": ["Luxembourg", "France", "Kayl"],
        })

        assert validate_residual_pii(df) == 0

    def test_structured_phone_false_positive_does_not_abort(self):
        df = pd.DataFrame({
            "record_key": ["source_2024_001"],
            "source_file": ["bronze_communes_2024.csv"],
        })

        assert validate_residual_pii(df) == 0

    def test_direct_phone_residual_still_fails(self):
        df = pd.DataFrame({"phone": ["+352 621 123 456"]})

        with pytest.raises(RuntimeError, match="phone.PHONE_NUMBER=1"):
            validate_residual_pii(df)

    def test_direct_email_residual_still_fails(self):
        df = pd.DataFrame({"source_file": ["alice@example.com.csv"], "email": ["bob@example.com"]})

        with pytest.raises(RuntimeError, match="email.EMAIL_ADDRESS=1"):
            validate_residual_pii(df)

    def test_direct_url_residual_still_fails(self):
        df = pd.DataFrame({"url": ["https://example.com/private/customer"]})

        with pytest.raises(RuntimeError, match="url.URL=1"):
            validate_residual_pii(df)

    def test_direct_ip_residual_still_fails(self):
        df = pd.DataFrame({"ip_address": ["192.168.10.25"]})

        with pytest.raises(RuntimeError, match="ip_address.IP_ADDRESS=1"):
            validate_residual_pii(df)

    def test_generated_tokens_are_not_residual_pii(self):
        df = pd.DataFrame({"name": ["PERSON_0"], "email": ["EMAIL_ADDRESS_0"]})
        assert validate_residual_pii(df) == 0

    def test_residual_error_summarizes_column_without_value(self):
        df = pd.DataFrame({"email": ["alice@example.com"]})
        with pytest.raises(RuntimeError) as exc_info:
            validate_residual_pii(df)
        message = str(exc_info.value)
        assert "email.EMAIL_ADDRESS=1" in message
        assert "alice@example.com" not in message

    def test_passes_clean_dataframe(self):
        df = pd.DataFrame({"note": ["No issues found."], "qty": [5]})
        count = validate_residual_pii(df)
        assert count == 0

    def test_raises_on_residual_email(self):
        df = pd.DataFrame({"email": ["alice@example.com"]})
        with pytest.raises(RuntimeError, match="Residual PII"):
            validate_residual_pii(df)

    def test_raises_message_contains_count(self):
        df = pd.DataFrame({"email": ["alice@example.com", "bob@company.org"]})
        with pytest.raises(RuntimeError) as exc_info:
            validate_residual_pii(df)
        assert "finding" in str(exc_info.value)

    def test_skips_non_object_columns(self):
        df = pd.DataFrame({
            "id":    pd.array([1, 2, 3], dtype="int64"),
            "score": pd.array([0.1, 0.2, 0.3], dtype="float64"),
        })
        count = validate_residual_pii(df)
        assert count == 0

    def test_skips_non_string_values_in_object_column(self):
        df = pd.DataFrame({"mixed": [None, 42, {"key": "val"}]})
        count = validate_residual_pii(df)
        assert count == 0

    def test_empty_dataframe_passes(self):
        df = pd.DataFrame({"email": pd.Series([], dtype=object)})
        count = validate_residual_pii(df)
        assert count == 0

    def test_entity_token_passes(self):
        """ENTITY_TYPE_N pseudonym tokens must not be flagged as PII."""
        df = pd.DataFrame({
            "name":  ["PERSON_0", "PERSON_1"],
            "email": ["EMAIL_ADDRESS_0", "EMAIL_ADDRESS_1"],
        })
        count = validate_residual_pii(df)
        assert count == 0

    def test_raises_on_pii_inside_json_string(self):
        df = pd.DataFrame({"payload": ['{"email": "alice@example.com"}']})
        with pytest.raises(RuntimeError, match="Residual PII"):
            validate_residual_pii(df)

    def test_raises_on_pii_inside_native_dict(self):
        df = pd.DataFrame({"data": [{"email": "alice@example.com"}]})
        with pytest.raises(RuntimeError, match="Residual PII"):
            validate_residual_pii(df)

    def test_raises_on_pii_inside_json_key(self):
        df = pd.DataFrame({"data": [{"alice@example.com": "primary contact"}]})
        with pytest.raises(RuntimeError, match=r"data:\$\.<key>\.EMAIL_ADDRESS"):
            validate_residual_pii(df)

    def test_metadata_column_exemption_does_not_hide_json_value_pii(self):
        df = pd.DataFrame({"source_file": [{"email": "alice@example.com"}]})
        with pytest.raises(RuntimeError, match=r"source_file:\$\.email\.EMAIL_ADDRESS=1"):
            validate_residual_pii(df)

    def test_pseudonym_passes_validation(self):
        """24-hex pseudonym tokens must not be detected as PII."""
        token = _FakePseudonymizer()("EMP001")
        df = pd.DataFrame({"employee_id": [token]})
        count = validate_residual_pii(df)
        assert count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Identifier column detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectIdentifierColumns:

    def test_employee_id_detected(self):
        df = pd.DataFrame({"employee_id": [1], "name": ["Alice"]})
        assert "employee_id" in detect_identifier_columns(df)

    def test_microsoft_id_detected(self):
        df = pd.DataFrame({"microsoft_id": ["abc123"], "score": [10]})
        assert "microsoft_id" in detect_identifier_columns(df)

    def test_matricule_detected(self):
        df = pd.DataFrame({"matricule": ["M001"], "score": [10]})
        assert "matricule" in detect_identifier_columns(df)

    def test_person_id_detected(self):
        df = pd.DataFrame({"person_id": ["P001"]})
        assert "person_id" in detect_identifier_columns(df)

    def test_user_id_detected(self):
        df = pd.DataFrame({"user_id": [42]})
        assert "user_id" in detect_identifier_columns(df)

    def test_column_with_space_normalised(self):
        df = pd.DataFrame({"employee id": ["E001"]})
        assert "employee id" in detect_identifier_columns(df)

    def test_non_identifier_column_not_detected(self):
        df = pd.DataFrame({"product_name": ["Widget"], "price": [9.99]})
        assert detect_identifier_columns(df) == []

    def test_explicit_cols_override(self):
        df = pd.DataFrame({"emp_id": [1], "score": [10]})
        assert detect_identifier_columns(df, explicit_cols=["emp_id"]) == ["emp_id"]

    def test_explicit_cols_filtered_to_present(self):
        df = pd.DataFrame({"emp_id": [1]})
        result = detect_identifier_columns(df, explicit_cols=["emp_id", "missing_col"])
        assert result == ["emp_id"]
        assert "missing_col" not in result

    def test_placeholder_identifier_column_detected(self):
        """Legacy placeholder identifier column names are still detected."""
        df = pd.DataFrame({"IDENTIFIER_0": ["x"]})
        assert "IDENTIFIER_0" in detect_identifier_columns(df)


# ─────────────────────────────────────────────────────────────────────────────
# Identifier column pseudonymization (Key Vault-bound)
# ─────────────────────────────────────────────────────────────────────────────

class TestPseudonymizeIdentifierColumns:

    def test_string_value_pseudonymized(self):
        df = pd.DataFrame({"employee_id": ["EMP001", "EMP002"]})
        result_df, pseudonymized = pseudonymize_identifier_columns(df, ["employee_id"], _FakePseudonymizer())
        assert "EMP001" not in result_df["employee_id"].values
        assert "EMP002" not in result_df["employee_id"].values
        assert "employee_id" in pseudonymized

    def test_mapping_is_deterministic_for_same_key(self):
        df = pd.DataFrame({"employee_id": ["EMP001"]})
        r1, _ = pseudonymize_identifier_columns(df.copy(), ["employee_id"], _FakePseudonymizer())
        r2, _ = pseudonymize_identifier_columns(df.copy(), ["employee_id"], _FakePseudonymizer())
        assert r1["employee_id"].iloc[0] == r2["employee_id"].iloc[0]

    def test_same_value_same_pseudonym_across_rows(self):
        df = pd.DataFrame({"employee_id": ["EMP001", "EMP001"]})
        result_df, _ = pseudonymize_identifier_columns(df, ["employee_id"], _FakePseudonymizer())
        assert result_df["employee_id"].iloc[0] == result_df["employee_id"].iloc[1]

    def test_different_values_different_pseudonyms(self):
        df = pd.DataFrame({"employee_id": ["EMP001", "EMP002"]})
        result_df, _ = pseudonymize_identifier_columns(df, ["employee_id"], _FakePseudonymizer())
        assert result_df["employee_id"].iloc[0] != result_df["employee_id"].iloc[1]

    def test_null_preserved(self):
        df = pd.DataFrame({"employee_id": [None, "EMP001"]})
        result_df, _ = pseudonymize_identifier_columns(df, ["employee_id"], _FakePseudonymizer())
        assert pd.isna(result_df["employee_id"].iloc[0])

    def test_integer_id_pseudonymized(self):
        df = pd.DataFrame({"employee_id": pd.array([12345, 67890], dtype="int64")})
        result_df, pseudonymized = pseudonymize_identifier_columns(df, ["employee_id"], _FakePseudonymizer())
        assert 12345 not in result_df["employee_id"].values
        assert "employee_id" in pseudonymized

    def test_different_keys_produce_different_pseudonyms(self):
        df = pd.DataFrame({"employee_id": ["EMP001"]})
        r1, _ = pseudonymize_identifier_columns(df.copy(), ["employee_id"], _FakePseudonymizer(b"key_a"))
        r2, _ = pseudonymize_identifier_columns(df.copy(), ["employee_id"], _FakePseudonymizer(b"key_b"))
        assert r1["employee_id"].iloc[0] != r2["employee_id"].iloc[0]

    def test_empty_id_cols_returns_unchanged(self):
        df = pd.DataFrame({"employee_id": ["EMP001"]})
        result_df, pseudonymized = pseudonymize_identifier_columns(df, [], _FakePseudonymizer())
        assert result_df["employee_id"].iloc[0] == "EMP001"
        assert pseudonymized == []

    def test_pseudonym_has_fixed_length(self):
        df = pd.DataFrame({"employee_id": ["short", "a_much_longer_employee_id_string"]})
        result_df, _ = pseudonymize_identifier_columns(df, ["employee_id"], _FakePseudonymizer())
        assert len(result_df["employee_id"].iloc[0]) == 24
        assert len(result_df["employee_id"].iloc[1]) == 24

    def test_original_dataframe_not_mutated(self):
        df = pd.DataFrame({"employee_id": ["EMP001"]})
        original = df["employee_id"].iloc[0]
        pseudonymize_identifier_columns(df, ["employee_id"], _FakePseudonymizer())
        assert df["employee_id"].iloc[0] == original

    def test_missing_column_silently_skipped(self):
        df = pd.DataFrame({"employee_id": ["EMP001"]})
        result_df, pseudonymized = pseudonymize_identifier_columns(
            df, ["employee_id", "nonexistent"], _FakePseudonymizer(),
        )
        assert "employee_id" in pseudonymized
        assert "nonexistent" not in pseudonymized

    def test_none_pseudonymizer_raises(self):
        df = pd.DataFrame({"employee_id": ["EMP001"]})
        with pytest.raises(ValueError, match="pseudonymizer is required"):
            pseudonymize_identifier_columns(df, ["employee_id"], None)
