"""Tests for GPS coordinate detection and spatial rounding anonymization."""

from decimal import Decimal
from datetime import datetime, timezone

import pandas as pd
import pytest

from app.classification import detect_gps_columns, detect_timestamp_columns
from app.anonymization import anonymize_gps_columns, bin_timestamp_columns, _round_wkt


# ─────────────────────────────────────────────────────────────────────────────
# detect_gps_columns
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectGpsColumns:

    def test_detects_numeric_latitude_column(self):
        df = pd.DataFrame({"latitude": [49.6112, 49.6200, 49.5900]})
        assert "latitude" in detect_gps_columns(df)

    def test_detects_numeric_longitude_column(self):
        df = pd.DataFrame({"longitude": [6.1319, 6.1400, 6.1200]})
        assert "longitude" in detect_gps_columns(df)

    def test_detects_abbreviated_lat_lon(self):
        df = pd.DataFrame({"lat": [49.6, 49.7], "lon": [6.1, 6.2]})
        cols = detect_gps_columns(df)
        assert "lat" in cols
        assert "lon" in cols

    def test_detects_string_decimal_lat_lon_by_name(self):
        df = pd.DataFrame({"latitude": ["49.611234", "49.620000"], "longitude": ["6.131987", "6.140000"]})
        cols = detect_gps_columns(df)
        assert set(cols) == {"latitude", "longitude"}

    def test_detects_decimal_object_lat_lon_by_name(self):
        df = pd.DataFrame({
            "lat": [Decimal("49.611234"), Decimal("49.620000")],
            "lon": [Decimal("6.131987"), Decimal("6.140000")],
        })
        cols = detect_gps_columns(df)
        assert set(cols) == {"lat", "lon"}

    def test_detects_lng_abbreviation(self):
        df = pd.DataFrame({"lng": [6.1319, 6.1400]})
        assert "lng" in detect_gps_columns(df)

    def test_detects_wkt_point_column(self):
        df = pd.DataFrame({"geom": [
            "POINT(6.1319 49.6112)",
            "POINT(6.1400 49.6200)",
        ]})
        assert "geom" in detect_gps_columns(df)

    def test_detects_wkt_point_case_insensitive(self):
        df = pd.DataFrame({"geometry": [
            "point(6.1319 49.6112)",
            "point(6.1400 49.6200)",
        ]})
        assert "geometry" in detect_gps_columns(df)

    def test_non_gps_numeric_column_excluded(self):
        df = pd.DataFrame({"age": [25, 30, 45], "score": [88, 72, 95]})
        assert detect_gps_columns(df) == []

    def test_non_gps_string_column_excluded(self):
        df = pd.DataFrame({"notes": ["hello world", "another note"]})
        assert detect_gps_columns(df) == []

    def test_decimal_values_without_gps_name_excluded(self):
        df = pd.DataFrame({"amount": ["49.611234", "49.620000"]})
        assert detect_gps_columns(df) == []

    def test_out_of_range_values_not_detected(self):
        # Values > 180 cannot be valid coordinates
        df = pd.DataFrame({"lat": [200.0, 350.0, 999.0]})
        assert detect_gps_columns(df) == []

    def test_null_values_tolerated(self):
        df = pd.DataFrame({"latitude": [49.6112, None, 49.5900]})
        assert "latitude" in detect_gps_columns(df)

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame({"latitude": pd.Series([], dtype="float64")})
        assert detect_gps_columns(df) == []

    def test_mixed_gps_and_non_gps_columns(self):
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "lat": [49.6, 49.7],
            "lon": [6.1, 6.2],
            "age": [25, 30],
        })
        cols = detect_gps_columns(df)
        assert set(cols) == {"lat", "lon"}


# ─────────────────────────────────────────────────────────────────────────────
# _round_wkt
# ─────────────────────────────────────────────────────────────────────────────

class TestRoundWkt:

    def test_rounds_point_coordinates(self):
        result = _round_wkt("POINT(6.131900 49.611200)", precision=2)
        assert result == "POINT(6.13 49.61)"

    def test_handles_negative_coordinates(self):
        result = _round_wkt("POINT(-73.935242 40.730610)", precision=2)
        assert result == "POINT(-73.94 40.73)"

    def test_precision_zero_gives_integer_coordinates(self):
        result = _round_wkt("POINT(6.131900 49.611200)", precision=0)
        assert result == "POINT(6.0 50.0)"

    def test_non_wkt_string_unchanged(self):
        result = _round_wkt("hello world", precision=2)
        assert result == "hello world"


# ─────────────────────────────────────────────────────────────────────────────
# anonymize_gps_columns
# ─────────────────────────────────────────────────────────────────────────────

class TestAnonymizeGpsColumns:

    def test_rounds_numeric_columns(self):
        df = pd.DataFrame({"lat": [49.6112345, 49.7654321], "lon": [6.1319876, 6.2345678]})
        result, anonymized = anonymize_gps_columns(df, ["lat", "lon"], precision=2)
        assert list(result["lat"]) == [49.61, 49.77]
        assert list(result["lon"]) == [6.13, 6.23]
        assert set(anonymized) == {"lat", "lon"}

    def test_default_precision_is_about_one_kilometer(self):
        df = pd.DataFrame({"lat": [49.6112345], "lon": [6.1319876]})
        result, _ = anonymize_gps_columns(df, ["lat", "lon"])
        assert list(result["lat"]) == [49.61]
        assert list(result["lon"]) == [6.13]

    def test_string_decimal_coordinates_are_rounded(self):
        df = pd.DataFrame({"latitude": ["49.611234"], "longitude": ["6.131987"]})
        result, _ = anonymize_gps_columns(df, ["latitude", "longitude"], precision=2)
        assert list(result["latitude"]) == [49.61]
        assert list(result["longitude"]) == [6.13]

    def test_decimal_object_coordinates_are_rounded(self):
        df = pd.DataFrame({"lat": [Decimal("49.611234")], "lon": [Decimal("6.131987")]})
        result, _ = anonymize_gps_columns(df, ["lat", "lon"], precision=2)
        assert list(result["lat"]) == [49.61]
        assert list(result["lon"]) == [6.13]

    def test_precision_3_decimal_places(self):
        df = pd.DataFrame({"latitude": [49.611234]})
        result, _ = anonymize_gps_columns(df, ["latitude"], precision=3)
        assert list(result["latitude"]) == [49.611]

    def test_precision_0_rounds_to_nearest_degree(self):
        df = pd.DataFrame({"lat": [49.6112]})
        result, _ = anonymize_gps_columns(df, ["lat"], precision=0)
        assert list(result["lat"]) == [50.0]

    def test_wkt_string_column_rounded(self):
        df = pd.DataFrame({"geom": ["POINT(6.131900 49.611200)", "POINT(6.140000 49.620000)"]})
        result, anonymized = anonymize_gps_columns(df, ["geom"], precision=2)
        assert list(result["geom"]) == ["POINT(6.13 49.61)", "POINT(6.14 49.62)"]
        assert anonymized == ["geom"]

    def test_null_values_preserved(self):
        df = pd.DataFrame({"lat": [49.6112, None]})
        result, _ = anonymize_gps_columns(df, ["lat"], precision=2)
        assert result["lat"].isna().sum() == 1
        assert result["lat"].iloc[0] == pytest.approx(49.61)

    def test_empty_gps_cols_returns_unchanged_df(self):
        df = pd.DataFrame({"lat": [49.6112]})
        result, anonymized = anonymize_gps_columns(df, [], precision=2)
        assert list(result["lat"]) == [49.6112]
        assert anonymized == []

    def test_original_dataframe_not_mutated(self):
        df = pd.DataFrame({"lat": [49.6112345]})
        anonymize_gps_columns(df, ["lat"], precision=2)
        assert df["lat"].iloc[0] == pytest.approx(49.6112345)

    def test_non_gps_columns_untouched(self):
        df = pd.DataFrame({"lat": [49.6112], "name": ["Alice"], "score": [88]})
        result, _ = anonymize_gps_columns(df, ["lat"], precision=2)
        assert list(result["name"]) == ["Alice"]
        assert list(result["score"]) == [88]

    def test_unknown_column_silently_skipped(self):
        df = pd.DataFrame({"lat": [49.6112]})
        result, anonymized = anonymize_gps_columns(df, ["lat", "nonexistent"], precision=2)
        assert anonymized == ["lat"]


# ─────────────────────────────────────────────────────────────────────────────
# detect_timestamp_columns
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectTimestampColumns:

    def test_detects_datetime64_column(self):
        df = pd.DataFrame({"recorded_at": pd.to_datetime(["2024-01-15 08:30:00", "2024-01-15 09:00:00"])})
        assert "recorded_at" in detect_timestamp_columns(df)

    def test_detects_string_column_with_timestamp_name(self):
        df = pd.DataFrame({"timestamp": ["2024-01-15 08:30:00", "2024-01-15 09:00:00"]})
        assert "timestamp" in detect_timestamp_columns(df)

    def test_detects_created_at_name(self):
        df = pd.DataFrame({"created_at": pd.to_datetime(["2024-01-15", "2024-01-16"])})
        assert "created_at" in detect_timestamp_columns(df)

    def test_non_timestamp_string_column_excluded(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        assert detect_timestamp_columns(df) == []

    def test_numeric_column_excluded(self):
        df = pd.DataFrame({"score": [88, 72]})
        assert detect_timestamp_columns(df) == []

    def test_returns_empty_for_no_timestamp_columns(self):
        df = pd.DataFrame({"lat": [49.6], "lon": [6.1], "value": [42]})
        assert detect_timestamp_columns(df) == []

    def test_mixed_table_returns_only_timestamp_cols(self):
        df = pd.DataFrame({
            "lat": [49.6],
            "lon": [6.1],
            "recorded_at": pd.to_datetime(["2024-01-15 08:30:00"]),
            "name": ["Alice"],
        })
        ts_cols = detect_timestamp_columns(df)
        assert ts_cols == ["recorded_at"]


# ─────────────────────────────────────────────────────────────────────────────
# bin_timestamp_columns
# ─────────────────────────────────────────────────────────────────────────────

class TestBinTimestampColumns:

    def test_floors_datetime64_to_midnight(self):
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-15 08:30:00", "2024-01-15 22:45:00"])})
        result, binned = bin_timestamp_columns(df, ["ts"])
        assert binned == ["ts"]
        assert result["ts"].iloc[0] == pd.Timestamp("2024-01-15")
        assert result["ts"].iloc[1] == pd.Timestamp("2024-01-15")

    def test_two_different_days_stay_distinct(self):
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-15 08:30:00", "2024-01-16 09:00:00"])})
        result, _ = bin_timestamp_columns(df, ["ts"])
        assert result["ts"].iloc[0] != result["ts"].iloc[1]

    def test_null_values_preserved(self):
        df = pd.DataFrame({"ts": pd.to_datetime([None, "2024-01-15 08:30:00"])})
        result, binned = bin_timestamp_columns(df, ["ts"])
        assert binned == ["ts"]
        assert pd.isna(result["ts"].iloc[0])
        assert result["ts"].iloc[1] == pd.Timestamp("2024-01-15")

    def test_string_column_not_binned(self):
        df = pd.DataFrame({"ts": ["2024-01-15 08:30:00", "2024-01-15 09:00:00"]})
        result, binned = bin_timestamp_columns(df, ["ts"])
        assert binned == []
        assert list(result["ts"]) == ["2024-01-15 08:30:00", "2024-01-15 09:00:00"]

    def test_original_dataframe_not_mutated(self):
        original_val = pd.Timestamp("2024-01-15 08:30:00")
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-15 08:30:00"])})
        bin_timestamp_columns(df, ["ts"])
        assert df["ts"].iloc[0] == original_val

    def test_empty_ts_cols_returns_unchanged_df(self):
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-15 08:30:00"])})
        result, binned = bin_timestamp_columns(df, [])
        assert binned == []
        assert result["ts"].iloc[0] == pd.Timestamp("2024-01-15 08:30:00")


# ─────────────────────────────────────────────────────────────────────────────
# Compound quasi-identifier: GPS + timestamp → k-anonymity
# ─────────────────────────────────────────────────────────────────────────────

class TestGpsTimestampKAnonymity:
    """Verify GPS spatial rounding + timestamp binning together feed k-anonymity."""

    def test_rounded_gps_reduces_cardinality_for_grouping(self):
        df = pd.DataFrame({
            "lat": [49.61234, 49.61567, 49.61890],
            "lon": [6.13456, 6.13123, 6.13789],
        })
        result, _ = anonymize_gps_columns(df, ["lat", "lon"], precision=1)
        assert result["lat"].nunique() == 1
        assert result["lon"].nunique() == 1

    def test_floored_timestamps_form_equal_groups_within_day(self):
        df = pd.DataFrame({"ts": pd.to_datetime([
            "2024-01-15 08:00:00",
            "2024-01-15 13:00:00",
            "2024-01-15 22:00:00",
        ])})
        result, binned = bin_timestamp_columns(df, ["ts"])
        assert binned == ["ts"]
        assert result["ts"].nunique() == 1

    def test_different_days_stay_in_separate_groups(self):
        df = pd.DataFrame({"ts": pd.to_datetime([
            "2024-01-15 08:00:00",
            "2024-01-16 08:00:00",
        ])})
        result, _ = bin_timestamp_columns(df, ["ts"])
        assert result["ts"].nunique() == 2
