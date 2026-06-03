"""GPS trajectory aggregation for GDPR-safe business analytics (Polars-native)."""

from __future__ import annotations

import logging

import polars as pl

from .classification import _tokens

logger = logging.getLogger(__name__)

_SPEED_NAME_TOKENS = frozenset({"speed", "velocity", "kmh", "mph", "knots", "spd"})


def detect_speed_column(df: pl.DataFrame) -> str | None:
    """Return the first numeric column whose name and value range suggest speed.

    Accepts values in [0, 300] km/h (covers walking → high-speed rail).
    Returns None when no candidate column is found.
    """
    for col in df.columns:
        series = df[col]
        if not series.dtype.is_numeric():
            continue
        if not (set(_tokens(str(col))) & _SPEED_NAME_TOKENS):
            continue
        non_null = series.drop_nulls()
        if len(non_null) == 0:
            continue
        in_range = ((non_null >= 0) & (non_null <= 300)).mean()
        if in_range is not None and in_range >= 0.8:
            return col
    return None


def _parsed_timestamp_expr(df: pl.DataFrame, ts_col: str) -> pl.Expr:
    """Expression yielding a Datetime for the timestamp column.

    Temporal columns pass through; string columns are parsed leniently
    (unparseable values become null and group under a null hour/day cell,
    matching the previous coerce semantics).
    """
    dtype = df.schema[ts_col]
    if isinstance(dtype, pl.Datetime):
        return pl.col(ts_col)
    if dtype == pl.Date:
        return pl.col(ts_col).cast(pl.Datetime)
    if dtype == pl.String:
        return pl.col(ts_col).str.to_datetime(strict=False)
    return pl.col(ts_col).cast(pl.Datetime, strict=False)


def aggregate_gps_table(
    df: pl.DataFrame,
    gps_cols: list[str],
    speed_col: str,
    ts_col: str,
    k: int,
) -> tuple[pl.DataFrame, dict]:
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
    total_pings = len(df)
    ts = _parsed_timestamp_expr(df, ts_col)

    # Only the columns the aggregation needs are materialised — the wide
    # source frame is never copied.
    cells = df.select([
        *gps_cols,
        speed_col,
        ts.dt.hour().alias("hour_of_day"),
        ts.dt.strftime("%A").alias("day_of_week"),
    ])

    group_keys = gps_cols + ["hour_of_day", "day_of_week"]
    agg = cells.group_by(group_keys).agg([
        pl.col(speed_col).count().alias("ping_count"),
        pl.col(speed_col).mean().alias("avg_speed_kmh"),
        pl.col(speed_col).quantile(0.5, interpolation="linear").alias("p50_speed_kmh"),
        pl.col(speed_col).quantile(0.85, interpolation="linear").alias("p85_speed_kmh"),
    ])

    kept = agg.filter(pl.col("ping_count") >= k)
    pings_suppressed = total_pings - int(kept["ping_count"].sum() or 0)

    logger.info(
        "GPS aggregate: %d cells retained (of %d), %d pings suppressed (k=%d)",
        len(kept), len(agg), pings_suppressed, k,
    )
    return kept, {
        "cells_retained": len(kept),
        "cells_total": len(agg),
        "pings_suppressed": pings_suppressed,
    }
