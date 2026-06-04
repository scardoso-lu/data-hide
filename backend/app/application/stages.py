"""Pipeline stage helpers — extracted from pipeline.py.

Pure helper functions that implement individual pipeline stages or
aggregate audit/stats structures.  ``run_table`` in ``pipeline.py``
calls these functions; they contain no orchestration logic of their own.
"""

from __future__ import annotations

import polars as pl

from ..domain.anonymization import EntityRegistry
from ..domain.classification import (
    FREE_TEXT,
    IDENTIFIER,
    QUASI_IDENTIFIER,
)
from ..infrastructure.repository import (
    TableMapping,
    _fresh_opts,
    read_delta,
    read_delta_sample,
    read_sql_table,
)
from .config import PipelineConfig

_CLASSIFICATION_SAMPLE_ROWS = 500


def _read_source_table(config: PipelineConfig, mapping: TableMapping) -> pl.DataFrame:
    if mapping.read_mode == "sql":
        return read_sql_table(
            mapping.table_name,
            config.sql_endpoint,
            config.sql_database,
            schema=mapping.schema or "dbo",
        )
    if mapping.read_mode == "delta":
        return read_delta(mapping.source_uri, _fresh_opts(mapping.source_uri))
    raise RuntimeError(f"Unsupported read_mode {mapping.read_mode!r} for table {mapping.table_name!r}")


def _read_source_sample(config: PipelineConfig, mapping: TableMapping) -> pl.DataFrame:
    """Read at most ``_CLASSIFICATION_SAMPLE_ROWS`` rows for Phase 1
    classification — the full table is only materialised later, in Phase 2,
    after every model has been released."""
    if mapping.read_mode == "sql":
        return read_sql_table(
            mapping.table_name,
            config.sql_endpoint,
            config.sql_database,
            schema=mapping.schema or "dbo",
            limit=_CLASSIFICATION_SAMPLE_ROWS,
        )
    if mapping.read_mode == "delta":
        return read_delta_sample(
            mapping.source_uri, _fresh_opts(mapping.source_uri), n=_CLASSIFICATION_SAMPLE_ROWS,
        )
    raise RuntimeError(f"Unsupported read_mode {mapping.read_mode!r} for table {mapping.table_name!r}")


def _apply_purview_audit(audit: dict, purview_result: dict) -> None:
    audit["purview_available"] = purview_result["available"]
    audit["purview_flagged_columns"] = purview_result["flagged_columns"]
    audit["purview_discrepancies"] = purview_result["discrepancies"]


def _apply_anonymization_audit(audit: dict, stats: dict, registry: EntityRegistry) -> None:
    audit["total_columns_scanned"] = len(stats["text_columns_scanned"])
    audit["columns_anonymized"] = stats["columns_with_detections"]
    audit["total_entities_detected"] = stats["total_entities_detected"]
    audit["entity_counts"] = stats["entity_counts"]
    audit["unique_entities"] = registry.unique_counts()


def _column_policy_stats(policies: dict, policy_stats: dict) -> dict:
    """Build the anonymization-audit ``stats`` dict from the column-policy
    layer's results (one "detection" per non-null cell rewritten, grouped by
    the policy's entity type). Shaped to match ``anonymize_dataframe``'s stats
    so the targeted row-scan can be merged in via ``_merge_scan_stats``.
    """
    entity_counts: dict[str, int] = {}
    for col, masked in policy_stats["values_masked"].items():
        etype = policy_stats["entity_types"].get(col) or "UNKNOWN"
        entity_counts[etype] = entity_counts.get(etype, 0) + masked
    return {
        "text_columns_scanned": list(policies.keys()),
        "columns_with_detections": list(policy_stats["columns_processed"]),
        "entity_counts": entity_counts,
        "total_entities_detected": sum(policy_stats["values_masked"].values()),
        "column_stats": [
            {
                "column": col,
                "detections": policy_stats["values_masked"][col],
                "entity_counts": {
                    (policy_stats["entity_types"].get(col) or "UNKNOWN"):
                        policy_stats["values_masked"][col],
                },
            }
            for col in policy_stats["columns_processed"]
        ],
    }


def _merge_scan_stats(stats: dict, scan_stats: dict) -> None:
    """Fold the targeted row-by-row scan's stats into the column-policy stats
    (in place). Scanned column lists are unioned; entity counts and totals are
    summed; per-column events are appended."""
    seen = set(stats["text_columns_scanned"])
    for c in scan_stats.get("text_columns_scanned", []):
        if c not in seen:
            stats["text_columns_scanned"].append(c)
            seen.add(c)
    seen_hit = set(stats["columns_with_detections"])
    for c in scan_stats.get("columns_with_detections", []):
        if c not in seen_hit:
            stats["columns_with_detections"].append(c)
            seen_hit.add(c)
    for etype, n in scan_stats.get("entity_counts", {}).items():
        stats["entity_counts"][etype] = stats["entity_counts"].get(etype, 0) + n
    stats["total_entities_detected"] += scan_stats.get("total_entities_detected", 0)
    stats["column_stats"].extend(scan_stats.get("column_stats", []))


def _profile_columns_by_category(profiles, category: str) -> list[str]:
    return [profile.name for profile in profiles if category in profile.categories]


def _configured_or_profiled_columns(configured: tuple[str, ...], profiles, category: str, df: pl.DataFrame) -> list[str]:
    if configured:
        return [col for col in configured if col in df.columns]
    return _profile_columns_by_category(profiles, category)
