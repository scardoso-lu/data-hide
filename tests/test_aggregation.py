"""Tests for GPS trajectory aggregation into spatial-temporal statistics."""

import pandas as pd
import pytest

from app.domain.aggregation import detect_speed_column, aggregate_gps_table


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# detect_speed_column
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestDetectSpeedColumn:

    def test_detects_speed_column_by_name(self):
        df = pd.DataFrame({"speed": [30.0, 50.0, 0.0]})
        assert detect_speed_column(df) == "speed"

    def test_detects_kmh_name(self):
        df = pd.DataFrame({"speed_kmh": [30.0, 50.0]})
        assert detect_speed_column(df) == "speed_kmh"

    def test_detects_velocity(self):
        df = pd.DataFrame({"velocity": [10.0, 20.0]})
        assert detect_speed_column(df) == "velocity"

    def test_non_speed_numeric_column_excluded(self):
        df = pd.DataFrame({"score": [88.0, 72.0], "age": [25.0, 30.0]})
        assert detect_speed_column(df) is None

    def test_out_of_range_values_excluded(self):
        df = pd.DataFrame({"speed": [400.0, 500.0]})
        assert detect_speed_column(df) is None

    def test_string_column_with_speed_name_excluded(self):
        df = pd.DataFrame({"speed": ["fast", "slow"]})
        assert detect_speed_column(df) is None

    def test_returns_none_for_empty_dataframe(self):
        df = pd.DataFrame({"speed": pd.Series([], dtype="float64")})
        assert detect_speed_column(df) is None


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# aggregate_gps_table
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _trajectory_df():
    """Minimal GPS trajectory: 3 pings in one cell on Monday morning, 2 in another."""
    return pd.DataFrame({
        "lat":       [49.6, 49.6, 49.6, 49.7, 49.7],
        "lon":       [6.1,  6.1,  6.1,  6.2,  6.2],
        "ts":        pd.to_datetime([
            "2024-01-15 08:10:00",
            "2024-01-15 08:25:00",
            "2024-01-15 08:40:00",
            "2024-01-15 09:05:00",
            "2024-01-15 09:20:00",
        ]),
        "speed":     [30.0, 45.0, 50.0, 20.0, 25.0],
        "driver_id": ["drv_abc", "drv_abc", "drv_abc", "drv_def", "drv_def"],
        "address":   ["LOCATION_0", "LOCATION_0", "LOCATION_0", "LOCATION_1", "LOCATION_1"],
    })


class TestAggregateGpsTable:

    def test_returns_dataframe(self):
        df, _ = aggregate_gps_table(_trajectory_df(), ["lat", "lon"], "speed", "ts", k=1)
        assert isinstance(df, pd.DataFrame)

    def test_group_keys_present(self):
        df, _ = aggregate_gps_table(_trajectory_df(), ["lat", "lon"], "speed", "ts", k=1)
        assert "lat" in df.columns
        assert "lon" in df.columns
        assert "hour_of_day" in df.columns
        assert "day_of_week" in df.columns

    def test_speed_metrics_present(self):
        df, _ = aggregate_gps_table(_trajectory_df(), ["lat", "lon"], "speed", "ts", k=1)
        assert "ping_count" in df.columns
        assert "avg_speed_kmh" in df.columns
        assert "p50_speed_kmh" in df.columns
        assert "p85_speed_kmh" in df.columns

    def test_individual_columns_dropped(self):
        df, _ = aggregate_gps_table(_trajectory_df(), ["lat", "lon"], "speed", "ts", k=1)
        assert "driver_id" not in df.columns
        assert "address" not in df.columns
        assert "ts" not in df.columns

    def test_hour_extracted_correctly(self):
        df, _ = aggregate_gps_table(_trajectory_df(), ["lat", "lon"], "speed", "ts", k=1)
        hours = set(df["hour_of_day"].tolist())
        assert hours == {8, 9}

    def test_day_of_week_extracted(self):
        df, _ = aggregate_gps_table(_trajectory_df(), ["lat", "lon"], "speed", "ts", k=1)
        assert df["day_of_week"].iloc[0] == "Monday"

    def test_ping_count_correct(self):
        df, _ = aggregate_gps_table(_trajectory_df(), ["lat", "lon"], "speed", "ts", k=1)
        cell_49_6 = df[df["lat"] == 49.6]
        assert cell_49_6["ping_count"].iloc[0] == 3

    def test_avg_speed_correct(self):
        df, _ = aggregate_gps_table(_trajectory_df(), ["lat", "lon"], "speed", "ts", k=1)
        cell_49_6 = df[df["lat"] == 49.6]
        assert cell_49_6["avg_speed_kmh"].iloc[0] == pytest.approx((30.0 + 45.0 + 50.0) / 3)

    def test_k_suppression_removes_small_cells(self):
        """With k=3, the cell with only 2 pings (49.7) should be suppressed."""
        df, stats = aggregate_gps_table(_trajectory_df(), ["lat", "lon"], "speed", "ts", k=3)
        assert len(df) == 1
        assert df["lat"].iloc[0] == 49.6
        assert stats["cells_retained"] == 1
        assert stats["cells_total"] == 2

    def test_suppressed_pings_counted_correctly(self):
        """The 2 pings in the suppressed cell should be counted."""
        _, stats = aggregate_gps_table(_trajectory_df(), ["lat", "lon"], "speed", "ts", k=3)
        assert stats["pings_suppressed"] == 2

    def test_k_equals_total_keeps_all(self):
        df, stats = aggregate_gps_table(_trajectory_df(), ["lat", "lon"], "speed", "ts", k=1)
        assert stats["pings_suppressed"] == 0
        assert len(df) == 2

    def test_original_dataframe_not_mutated(self):
        original = _trajectory_df()
        original_cols = list(original.columns)
        aggregate_gps_table(original, ["lat", "lon"], "speed", "ts", k=1)
        assert list(original.columns) == original_cols
        assert "hour_of_day" not in original.columns

    def test_two_different_hours_give_two_cells_for_same_location(self):
        df = pd.DataFrame({
            "lat":   [49.6, 49.6, 49.6, 49.6, 49.6, 49.6],
            "lon":   [6.1,  6.1,  6.1,  6.1,  6.1,  6.1],
            "ts":    pd.to_datetime([
                "2024-01-15 08:00:00", "2024-01-15 08:10:00", "2024-01-15 08:20:00",
                "2024-01-15 09:00:00", "2024-01-15 09:10:00", "2024-01-15 09:20:00",
            ]),
            "speed": [30.0, 35.0, 40.0, 50.0, 55.0, 60.0],
        })
        result, _ = aggregate_gps_table(df, ["lat", "lon"], "speed", "ts", k=1)
        assert len(result) == 2
        assert set(result["hour_of_day"]) == {8, 9}
