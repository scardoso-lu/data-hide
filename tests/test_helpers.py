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
    _storage_opts,
    detect_quasi_identifiers,
    enforce_k_anonymity,
    flag_free_text_columns,
    run_purview_check,
    sanitize_column_names,
    validate_residual_pii,
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
        df = pd.DataFrame({"notes": ["some text"], "price": [9.99]})
        assert "notes" in flag_free_text_columns(df)

    def test_description_column_flagged(self):
        df = pd.DataFrame({"description": ["Widget A"], "id": [1]})
        assert "description" in flag_free_text_columns(df)

    def test_feedback_column_flagged(self):
        df = pd.DataFrame({"customer_feedback": ["great!"], "qty": [1]})
        assert "customer_feedback" in flag_free_text_columns(df)

    def test_numeric_column_not_flagged(self):
        df = pd.DataFrame({"price": pd.array([9.99], dtype="float64"), "qty": pd.array([1], dtype="int64")})
        assert flag_free_text_columns(df) == []

    def test_non_text_object_column_with_matching_name_still_flagged(self):
        df = pd.DataFrame({"notes": [{"key": "val"}]})
        assert "notes" in flag_free_text_columns(df)

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
        df = pd.DataFrame({"age": [25], "score": [10]})
        qi = detect_quasi_identifiers(df, explicit_cols=["age", "missing_col"])
        assert qi == ["age"]
        assert "missing_col" not in qi

    def test_keyword_detection_on_age(self):
        df = pd.DataFrame({"age": [25], "score": [10]})
        qi = detect_quasi_identifiers(df)
        assert "age" in qi

    def test_keyword_detection_on_gender(self):
        df = pd.DataFrame({"gender": ["M"], "score": [10]})
        qi = detect_quasi_identifiers(df)
        assert "gender" in qi

    def test_non_qi_column_not_detected(self):
        df = pd.DataFrame({"product_name": ["Widget"], "price": [9.99]})
        qi = detect_quasi_identifiers(df)
        assert qi == []

    def test_empty_explicit_cols_falls_back_to_keyword(self):
        df = pd.DataFrame({"age": [25], "city": ["NYC"]})
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

    def test_ssn_column_renamed(self):
        df = pd.DataFrame({"ssn": ["123-45-6789"], "name": ["Alice"]})
        result_df, renames = sanitize_column_names(df)
        assert "ssn" not in result_df.columns
        assert "ssn" in renames

    def test_non_sensitive_columns_unchanged(self):
        df = pd.DataFrame({"score": [10], "category": ["A"], "email": ["a@b.com"]})
        result_df, renames = sanitize_column_names(df)
        assert "score" in result_df.columns
        assert "category" in result_df.columns
        assert renames == {}

    def test_renamed_column_gets_category_prefix(self):
        df = pd.DataFrame({"ssn": ["123"]})
        result_df, renames = sanitize_column_names(df)
        new_name = renames["ssn"]
        assert new_name.startswith("IDENTIFIER_")

    def test_multiple_sensitive_columns_all_renamed(self):
        df = pd.DataFrame({"ssn": ["x"], "passport": ["y"], "score": [1]})
        result_df, renames = sanitize_column_names(df)
        assert "ssn" not in result_df.columns
        assert "passport" not in result_df.columns
        assert "score" in result_df.columns

    def test_health_column_renamed_as_sensitive(self):
        df = pd.DataFrame({"health_status": ["ok"]})
        result_df, renames = sanitize_column_names(df)
        assert "health_status" not in result_df.columns
        new_name = renames["health_status"]
        assert new_name.startswith("SENSITIVE_")

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
