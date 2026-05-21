"""GPS trajectory aggregation for GDPR-safe business analytics."""

from __future__ import annotations

import logging

import pandas as pd

from .classification import _tokens

logger = logging.getLogger(__name__)

_SPEED_NAME_TOKENS = frozenset({"speed", "velocity", "kmh", "mph", "knots", "spd"})


def detect_speed_column(df: pd.DataFrame) -> str | None:
    """Return the first numeric column whose name and value range suggest speed.

    Accepts values in [0, 300] km/h (covers walking → high-speed rail).
    Returns None when no candidate column is found.
    """
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col].dtype):
            continue
        if not (set(_tokens(str(col))) & _SPEED_NAME_TOKENS):
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        if non_null.between(0, 300).mean() >= 0.8:
            return col
    return None


def aggregate_gps_table(
    df: pd.DataFrame,
    gps_cols: list[str],
    speed_col: str,
    ts_col: str,
    k: int,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate a GPS trajectory table into spatial-temporal statistics.

    Groups rows by (rounded GPS cell × hour_of_day × day_of_week) and
    computes ping count and speed percentiles.  Cells with fewer than k
    pings are suppressed so no rare location-time combination survives.

    The resulting DataFrame contains only aggregate metrics — no individual
    rows, no vehicle identifiers, no addresses — and is safe to pass to
    external LLMs or business consumers.

    Returns (aggregate_df, stats) where stats reports cells retained and
    pings suppressed for the audit record.
    """
    df = df.copy()

    ts = pd.to_datetime(df[ts_col], errors="coerce")
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.day_name()

    group_keys = gps_cols + ["hour_of_day", "day_of_week"]
    total_pings = len(df)

    agg = (
        df.groupby(group_keys, dropna=False)
        .agg(
            ping_count=(speed_col, "count"),
            avg_speed_kmh=(speed_col, "mean"),
            p50_speed_kmh=(speed_col, lambda x: x.quantile(0.5)),
            p85_speed_kmh=(speed_col, lambda x: x.quantile(0.85)),
        )
        .reset_index()
    )

    kept = agg[agg["ping_count"] >= k].reset_index(drop=True)
    pings_suppressed = total_pings - int(kept["ping_count"].sum())

    logger.info(
        "GPS aggregate: %d cells retained (of %d), %d pings suppressed (k=%d)",
        len(kept), len(agg), pings_suppressed, k,
    )
    return kept, {
        "cells_retained": len(kept),
        "cells_total": len(agg),
        "pings_suppressed": pings_suppressed,
    }
