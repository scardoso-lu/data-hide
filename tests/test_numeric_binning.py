"""Tests for numeric quasi-identifier binning."""

import pandas as pd
import pytest

from app.anonymization import bin_numeric_columns


class TestBinNumericColumns:

    def test_returns_dataframe_and_list(self):
        df = pd.DataFrame({"hours": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        result, binned = bin_numeric_columns(df, ["hours"])
        assert isinstance(result, pd.DataFrame)
        assert isinstance(binned, list)

    def test_column_included_in_binned_list(self):
        df = pd.DataFrame({"hours": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        _, binned = bin_numeric_columns(df, ["hours"])
        assert "hours" in binned

    def test_values_replaced_with_range_strings(self):
        df = pd.DataFrame({"price": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]})
        result, _ = bin_numeric_columns(df, ["price"])
        assert result["price"].dtype == object
        for val in result["price"].dropna():
            assert "–" in str(val)

    def test_original_dataframe_not_mutated(self):
        df = pd.DataFrame({"hours": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        original_vals = df["hours"].tolist()
        bin_numeric_columns(df, ["hours"])
        assert df["hours"].tolist() == original_vals

    def test_nulls_preserved(self):
        df = pd.DataFrame({"hours": [10.0, None, 30.0, 40.0, 50.0, 60.0]})
        result, _ = bin_numeric_columns(df, ["hours"])
        assert result["hours"].isna().sum() == 1

    def test_non_numeric_column_skipped(self):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Carol"]})
        result, binned = bin_numeric_columns(df, ["name"])
        assert "name" not in binned
        assert result["name"].tolist() == ["Alice", "Bob", "Carol"]

    def test_column_with_single_unique_value_skipped(self):
        df = pd.DataFrame({"score": [42.0, 42.0, 42.0]})
        result, binned = bin_numeric_columns(df, ["score"])
        assert "score" not in binned
        assert result["score"].tolist() == [42.0, 42.0, 42.0]

    def test_empty_cols_list_returns_unchanged(self):
        df = pd.DataFrame({"hours": [10.0, 20.0, 30.0]})
        result, binned = bin_numeric_columns(df, [])
        assert binned == []
        assert result["hours"].tolist() == [10.0, 20.0, 30.0]

    def test_multiple_columns_binned(self):
        df = pd.DataFrame({
            "hours":    [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "turnover": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0],
        })
        _, binned = bin_numeric_columns(df, ["hours", "turnover"])
        assert "hours" in binned
        assert "turnover" in binned

    def test_integer_like_values_have_no_decimal_in_label(self):
        df = pd.DataFrame({"units": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        result, _ = bin_numeric_columns(df, ["units"])
        for val in result["units"].dropna():
            assert "." not in val or val.replace("–", "").replace(".", "").isnumeric() is False

    def test_missing_column_name_silently_ignored(self):
        df = pd.DataFrame({"hours": [10.0, 20.0, 30.0, 40.0]})
        result, binned = bin_numeric_columns(df, ["hours", "nonexistent"])
        assert "nonexistent" not in binned
        assert "hours" in binned

    def test_same_row_values_differ_in_output(self):
        """Values that were different before binning end up in consistent bins."""
        df = pd.DataFrame({"hours": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
        result, _ = bin_numeric_columns(df, ["hours"])
        # All values should be assigned a bin (no NaN introduced beyond original)
        assert result["hours"].notna().all()

    def test_fewer_unique_values_than_bins_still_works(self):
        df = pd.DataFrame({"score": [10.0, 10.0, 20.0, 20.0, 30.0, 30.0]})
        result, binned = bin_numeric_columns(df, ["score"])
        assert "score" in binned
        assert result["score"].dtype == object
