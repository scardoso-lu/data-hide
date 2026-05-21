"""
Unit tests for URI helpers, storage options, the optional Purview check,
free-text column flagging, quasi-identifier detection, k-anonymity enforcement,
column name sanitization, and residual PII validation.
No external services or spaCy model required (except residual PII tests).
"""

import pandas as pd
import pytest

from main import (
    PurviewClient,
    _account_name,
    _scan_json_for_pii,
    _storage_opts,
    detect_identifier_columns,
    detect_quasi_identifiers,
    enforce_k_anonymity,
    flag_free_text_columns,
    hash_identifier_columns,
    run_purview_check,
    sanitize_column_names,
    validate_residual_pii,
    write_delta,
)


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


class TestWriteDelta:

    def test_uploads_parquet_file_to_explicit_path(self, mocker):
        import app.repository as repo

        file_client = mocker.MagicMock()
        service = mocker.MagicMock()
        service.get_file_client.return_value = file_client
        service_cls = mocker.MagicMock(return_value=service)

        mocker.patch_object(repo, "DataLakeServiceClient", service_cls)
        mocker.patch_object(repo, "_credential_instance", lambda: object())

        write_delta(
            pd.DataFrame({"x": [1]}),
            "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Files/out/customers.parquet",
            {},
        )

        service_cls.assert_called_once()
        service.get_file_client.assert_called_once_with(
            file_system="ws",
            file_path="lh.Lakehouse/Files/out/customers.parquet",
        )
        file_client.upload_data.assert_called_once()
        assert file_client.upload_data.call_args.kwargs["overwrite"] is True

    def test_appends_default_parquet_name_for_folder_path(self, mocker):
        import app.repository as repo

        file_client = mocker.MagicMock()
        service = mocker.MagicMock()
        service.get_file_client.return_value = file_client

        mocker.patch_object(repo, "DataLakeServiceClient", mocker.MagicMock(return_value=service))
        mocker.patch_object(repo, "_credential_instance", lambda: object())

        write_delta(pd.DataFrame({"x": [1]}), "abfss://ws@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Files/out/customers", {})

        service.get_file_client.assert_called_once_with(
            file_system="ws",
            file_path="lh.Lakehouse/Files/out/customers/part-00000.parquet",
        )


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
# Column name sanitization
# ─────────────────────────────────────────────────────────────────────────────

class TestSanitizeColumnNames:

    def test_identifier_column_name_preserved(self):
        df = pd.DataFrame({"employee_id": ["EMP-001"], "name": ["Alice"]})
        result_df, renames = sanitize_column_names(df)
        assert list(result_df.columns) == ["employee_id", "name"]
        assert renames == {}

    def test_non_sensitive_columns_unchanged(self):
        df = pd.DataFrame({"score": [10], "category": ["A"], "email": ["a@b.com"]})
        result_df, renames = sanitize_column_names(df)
        assert "score" in result_df.columns
        assert "category" in result_df.columns
        assert renames == {}

    def test_identifier_column_does_not_get_category_prefix(self):
        df = pd.DataFrame({"employee_id": ["EMP-001"]})
        result_df, renames = sanitize_column_names(df)
        assert list(result_df.columns) == ["employee_id"]
        assert renames == {}

    def test_multiple_sensitive_columns_preserved(self):
        df = pd.DataFrame({"employee_id": ["EMP-001"], "person_id": ["P-001"], "score": [1]})
        result_df, renames = sanitize_column_names(df)
        assert list(result_df.columns) == ["employee_id", "person_id", "score"]
        assert renames == {}

    def test_sensitive_column_name_preserved(self):
        df = pd.DataFrame({"risk_band": ["high", "high", "high", "low", "low", "low"]})
        result_df, renames = sanitize_column_names(df)
        assert list(result_df.columns) == ["risk_band"]
        assert renames == {}

    def test_no_renames_returns_empty_dict(self):
        df = pd.DataFrame({"product": ["Widget"], "price": [9.99]})
        _, renames = sanitize_column_names(df)
        assert renames == {}

    def test_original_dataframe_not_mutated(self):
        df = pd.DataFrame({"ssn": ["123"]})
        original_cols = list(df.columns)
        sanitize_column_names(df)
        assert list(df.columns) == original_cols


# ─────────────────────────────────────────────────────────────────────────────
# Residual PII validation
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateResidualPII:

    class FakeAnalyzer:
        def analyze(self, text, entities=None, language=None):
            from types import SimpleNamespace

            findings = []
            for token in ("PERSON_0", "EMAIL_ADDRESS_0"):
                start = text.find(token)
                if start >= 0:
                    findings.append(SimpleNamespace(start=start, end=start + len(token), entity_type="PERSON"))
            email = "alice@example.com"
            start = text.find(email)
            if start >= 0:
                findings.append(SimpleNamespace(start=start, end=start + len(email), entity_type="EMAIL_ADDRESS"))
            return findings

    def test_generated_tokens_are_not_residual_pii(self):
        df = pd.DataFrame({"name": ["PERSON_0"], "email": ["EMAIL_ADDRESS_0"]})
        assert validate_residual_pii(df, self.FakeAnalyzer()) == 0

    def test_residual_error_summarizes_column_without_value(self):
        df = pd.DataFrame({"email": ["alice@example.com"]})
        with pytest.raises(RuntimeError) as exc_info:
            validate_residual_pii(df, self.FakeAnalyzer())
        message = str(exc_info.value)
        assert "email.EMAIL_ADDRESS=1" in message
        assert "alice@example.com" not in message

    def test_passes_clean_dataframe(self, analyzer):
        df = pd.DataFrame({"note": ["No issues found."], "qty": [5]})
        count = validate_residual_pii(df, analyzer)
        assert count == 0

    def test_raises_on_residual_email(self, analyzer):
        df = pd.DataFrame({"email": ["alice@example.com"]})
        with pytest.raises(RuntimeError, match="Residual PII"):
            validate_residual_pii(df, analyzer)

    def test_raises_message_contains_count(self, analyzer):
        df = pd.DataFrame({"email": ["alice@example.com", "bob@company.org"]})
        with pytest.raises(RuntimeError) as exc_info:
            validate_residual_pii(df, analyzer)
        assert "finding" in str(exc_info.value)

    def test_skips_non_object_columns(self, analyzer):
        df = pd.DataFrame({
            "id":    pd.array([1, 2, 3], dtype="int64"),
            "score": pd.array([0.1, 0.2, 0.3], dtype="float64"),
        })
        count = validate_residual_pii(df, analyzer)
        assert count == 0

    def test_skips_non_string_values_in_object_column(self, analyzer):
        df = pd.DataFrame({"mixed": [None, 42, {"key": "val"}]})
        count = validate_residual_pii(df, analyzer)
        assert count == 0

    def test_empty_dataframe_passes(self, analyzer):
        df = pd.DataFrame({"email": pd.Series([], dtype=object)})
        count = validate_residual_pii(df, analyzer)
        assert count == 0

    def test_entity_token_passes(self, analyzer):
        """ENTITY_TYPE_N pseudonym tokens must not be flagged as PII."""
        df = pd.DataFrame({
            "name":  ["PERSON_0", "PERSON_1"],
            "email": ["EMAIL_ADDRESS_0", "EMAIL_ADDRESS_1"],
        })
        count = validate_residual_pii(df, analyzer)
        assert count == 0

    def test_raises_on_pii_inside_json_string(self, analyzer):
        df = pd.DataFrame({"payload": ['{"email": "alice@example.com"}']})
        with pytest.raises(RuntimeError, match="Residual PII"):
            validate_residual_pii(df, analyzer)

    def test_raises_on_pii_inside_native_dict(self, analyzer):
        df = pd.DataFrame({"data": [{"email": "alice@example.com"}]})
        with pytest.raises(RuntimeError, match="Residual PII"):
            validate_residual_pii(df, analyzer)

    def test_hash_passes_validation(self, analyzer):
        """SHA-256 hex digest must not be detected as PII."""
        import hashlib
        h = hashlib.sha256(b"EMP001").hexdigest()[:24]
        df = pd.DataFrame({"employee_id": [h]})
        count = validate_residual_pii(df, analyzer)
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

    def test_sanitized_identifier_column_detected(self):
        """Columns renamed to IDENTIFIER_N by sanitize_column_names are also detected."""
        df = pd.DataFrame({"IDENTIFIER_0": ["x"]})
        assert "IDENTIFIER_0" in detect_identifier_columns(df)


# ─────────────────────────────────────────────────────────────────────────────
# Identifier column hashing
# ─────────────────────────────────────────────────────────────────────────────

class TestHashIdentifierColumns:

    def test_string_value_hashed(self):
        df = pd.DataFrame({"employee_id": ["EMP001", "EMP002"]})
        result_df, hashed = hash_identifier_columns(df, ["employee_id"])
        assert "EMP001" not in result_df["employee_id"].values
        assert "EMP002" not in result_df["employee_id"].values
        assert "employee_id" in hashed

    def test_hash_is_deterministic(self):
        df = pd.DataFrame({"employee_id": ["EMP001"]})
        r1, _ = hash_identifier_columns(df.copy(), ["employee_id"])
        r2, _ = hash_identifier_columns(df.copy(), ["employee_id"])
        assert r1["employee_id"].iloc[0] == r2["employee_id"].iloc[0]

    def test_same_value_same_hash_across_rows(self):
        df = pd.DataFrame({"employee_id": ["EMP001", "EMP001"]})
        result_df, _ = hash_identifier_columns(df, ["employee_id"])
        assert result_df["employee_id"].iloc[0] == result_df["employee_id"].iloc[1]

    def test_different_values_different_hashes(self):
        df = pd.DataFrame({"employee_id": ["EMP001", "EMP002"]})
        result_df, _ = hash_identifier_columns(df, ["employee_id"])
        assert result_df["employee_id"].iloc[0] != result_df["employee_id"].iloc[1]

    def test_null_preserved(self):
        df = pd.DataFrame({"employee_id": [None, "EMP001"]})
        result_df, _ = hash_identifier_columns(df, ["employee_id"])
        assert pd.isna(result_df["employee_id"].iloc[0])

    def test_integer_id_hashed(self):
        df = pd.DataFrame({"employee_id": pd.array([12345, 67890], dtype="int64")})
        result_df, hashed = hash_identifier_columns(df, ["employee_id"])
        assert 12345 not in result_df["employee_id"].values
        assert "employee_id" in hashed

    def test_salt_changes_hash(self):
        df = pd.DataFrame({"employee_id": ["EMP001"]})
        r1, _ = hash_identifier_columns(df.copy(), ["employee_id"], salt="salt_a")
        r2, _ = hash_identifier_columns(df.copy(), ["employee_id"], salt="salt_b")
        assert r1["employee_id"].iloc[0] != r2["employee_id"].iloc[0]

    def test_empty_id_cols_returns_unchanged(self):
        df = pd.DataFrame({"employee_id": ["EMP001"]})
        result_df, hashed = hash_identifier_columns(df, [])
        assert result_df["employee_id"].iloc[0] == "EMP001"
        assert hashed == []

    def test_hash_has_fixed_length(self):
        df = pd.DataFrame({"employee_id": ["short", "a_much_longer_employee_id_string"]})
        result_df, _ = hash_identifier_columns(df, ["employee_id"])
        assert len(result_df["employee_id"].iloc[0]) == 24
        assert len(result_df["employee_id"].iloc[1]) == 24

    def test_original_dataframe_not_mutated(self):
        df = pd.DataFrame({"employee_id": ["EMP001"]})
        original = df["employee_id"].iloc[0]
        hash_identifier_columns(df, ["employee_id"])
        assert df["employee_id"].iloc[0] == original

    def test_missing_column_silently_skipped(self):
        df = pd.DataFrame({"employee_id": ["EMP001"]})
        result_df, hashed = hash_identifier_columns(df, ["employee_id", "nonexistent"])
        assert "employee_id" in hashed
        assert "nonexistent" not in hashed
