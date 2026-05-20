"""
Unit tests for URI helpers, storage options, and the optional Purview check.
No external services or spaCy model required.
"""

import pytest

from main import (
    PurviewClient,
    _account_name,
    _storage_opts,
    run_purview_check,
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
        assert result["discrepancies"] == []     # both present in df_columns

    def test_discrepancy_when_flagged_column_absent_from_dataframe(self, mocker):
        mocker.patch("main.acquire_token", return_value="fake-token")
        mock_cls = mocker.patch("main.PurviewClient")
        mock_cls.qualified_name.return_value = "https://..."
        mock_cls.return_value.column_classifications.return_value = {
            "ssn":   ["MICROSOFT.PERSONAL.SSN"],    # NOT in df
            "email": ["MICROSOFT.PERSONAL.EMAIL"],   # is in df
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
