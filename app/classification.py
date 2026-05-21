"""Column classification helpers for GDPR-oriented anonymization."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

import pandas as pd


IDENTIFIER = "IDENTIFIER"
SENSITIVE = "SENSITIVE"
FREE_TEXT = "FREE_TEXT"
QUASI_IDENTIFIER = "QUASI_IDENTIFIER"

_GPS_NAME_TOKENS = frozenset({
    "lat", "latitude", "lon", "lng", "longitude",
    "gps", "coord", "coords", "coordinate", "coordinates",
    "geom", "geometry", "wkt", "point",
})
_WKT_POINT_RE = re.compile(r"POINT\s*\(", re.IGNORECASE)


def _is_text_column(dtype) -> bool:
    return pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9]+", name.lower()) if t]


def _sample(series: pd.Series, limit: int = 100) -> list[object]:
    return [v for v in series.dropna().head(limit).tolist()]


def _unique_ratio(values: Iterable[object]) -> float:
    vals = [str(v) for v in values if not pd.isna(v)]
    if not vals:
        return 0.0
    return len(set(vals)) / len(vals)


def _looks_like_identifier_values(values: list[object]) -> bool:
    if not values:
        return False
    text_values = [str(v).strip() for v in values if str(v).strip()]
    if not text_values:
        return False
    ratio = _unique_ratio(text_values)
    structured = sum(bool(re.fullmatch(r"[A-Za-z0-9_.:@/-]{3,128}", v)) for v in text_values)
    has_identifier_shape = sum(bool(re.search(r"\d", v) and re.search(r"[A-Za-z]", v)) for v in text_values)
    return ratio >= 0.75 and structured / len(text_values) >= 0.8 and has_identifier_shape / len(text_values) >= 0.5


def _looks_like_free_text(values: list[object]) -> bool:
    strings = [v for v in values if isinstance(v, str) and v.strip()]
    if not strings:
        return False
    avg_len = sum(len(v) for v in strings) / len(strings)
    avg_words = sum(len(v.split()) for v in strings) / len(strings)
    jsonish = sum(v.lstrip().startswith(("{", "[")) for v in strings) / len(strings)
    return avg_len >= 32 or avg_words >= 5 or jsonish >= 0.5


def _looks_like_quasi_identifier(series: pd.Series, values: list[object]) -> bool:
    if not values:
        return False
    ratio = _unique_ratio(values)
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return True
    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
        if numeric.empty:
            return False
        return ratio <= 0.4 and numeric.between(0, 130).mean() >= 0.8
    return 0 < ratio <= 0.35


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    categories: tuple[str, ...]


class ColumnClassifier:
    """Classifies columns from names and observed value shape.

    The classifier deliberately avoids project-specific keyword catalogs. It
    uses generic identifier naming conventions, cardinality, type, and sample
    text shape so it can adapt to new enterprise schemas.
    """

    def classify(self, df: pd.DataFrame) -> list[ColumnProfile]:
        profiles: list[ColumnProfile] = []
        for col in df.columns:
            series = df[col]
            values = _sample(series)
            tokens = _tokens(str(col))
            cats: list[str] = []

            if self._is_identifier_column(tokens, values):
                cats.append(IDENTIFIER)
            if _looks_like_quasi_identifier(series, values) and IDENTIFIER not in cats:
                cats.append(QUASI_IDENTIFIER)
            if _is_text_column(series.dtype) and _looks_like_free_text(values):
                cats.append(FREE_TEXT)
            if self._is_sensitive_shape(series, values):
                cats.append(SENSITIVE)

            profiles.append(ColumnProfile(col, tuple(dict.fromkeys(cats))))
        return profiles

    @staticmethod
    def _is_identifier_column(tokens: list[str], values: list[object]) -> bool:
        token_set = set(tokens)
        id_token = bool(token_set & {"id", "ids", "uuid", "guid", "identifier", "key"})
        id_suffix = any(t.endswith("id") and len(t) > 2 for t in tokens)
        return id_token or id_suffix or _looks_like_identifier_values(values)

    @staticmethod
    def _is_sensitive_shape(series: pd.Series, values: list[object]) -> bool:
        if not values:
            return False
        if pd.api.types.is_bool_dtype(series.dtype):
            return False
        non_null = len(values)
        unique = len({str(v) for v in values})
        entropy = 0.0
        counts: dict[str, int] = {}
        for v in values:
            counts[str(v)] = counts.get(str(v), 0) + 1
        for count in counts.values():
            p = count / non_null
            entropy -= p * math.log2(p)
        return unique <= max(8, non_null * 0.2) and entropy > 0 and non_null >= 5


def classify_columns(df: pd.DataFrame) -> list[ColumnProfile]:
    return ColumnClassifier().classify(df)


def columns_by_category(df: pd.DataFrame, category: str) -> list[str]:
    return [p.name for p in classify_columns(df) if category in p.categories]


def flag_free_text_columns(df: pd.DataFrame) -> list[str]:
    return columns_by_category(df, FREE_TEXT)


def detect_quasi_identifiers(df: pd.DataFrame, explicit_cols: list[str] | None = None) -> list[str]:
    if explicit_cols:
        return [c for c in explicit_cols if c in df.columns]
    return columns_by_category(df, QUASI_IDENTIFIER)


def detect_identifier_columns(df: pd.DataFrame, explicit_cols: list[str] | None = None) -> list[str]:
    if explicit_cols:
        return [c for c in explicit_cols if c in df.columns]
    return columns_by_category(df, IDENTIFIER)


def detect_gps_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that contain GPS coordinates (numeric lat/lon or WKT POINT strings).

    A numeric column qualifies when its name contains a GPS keyword and all
    non-null values fall within [-180, 180].  A string column qualifies when
    ≥80 % of its non-null sample values match the WKT POINT(…) pattern.
    """
    result: list[str] = []
    for col in df.columns:
        series = df[col]
        tokens = set(_tokens(str(col)))
        if _is_numeric_gps_column(series, tokens) or _is_wkt_gps_column(series):
            result.append(col)
    return result


def _is_numeric_gps_column(series: pd.Series, tokens: set[str]) -> bool:
    if not pd.api.types.is_numeric_dtype(series.dtype):
        return False
    if not (tokens & _GPS_NAME_TOKENS):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    return bool(non_null.between(-180, 180).all())


def _is_wkt_gps_column(series: pd.Series) -> bool:
    if not _is_text_column(series.dtype):
        return False
    sample = [v for v in series.dropna().head(20).tolist() if isinstance(v, str)]
    if not sample:
        return False
    matches = sum(1 for v in sample if _WKT_POINT_RE.match(v.strip()))
    return matches / len(sample) >= 0.8


def sanitize_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep the source schema contract intact.

    Earlier versions renamed sensitive columns to category placeholders. That
    breaks downstream consumers, so classification now drives value
    transformations only and column names are preserved.
    """
    return df, {}
