"""Pipeline orchestration service."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from time import perf_counter
import uuid

import pandas as pd

from .aggregation import aggregate_gps_table, detect_speed_column
from .anonymization import (
    EntityRegistry,
    anonymize_dataframe,
    anonymize_gps_columns,
    bin_numeric_columns,
    bin_timestamp_columns,
    build_engines,
    enforce_k_anonymity,
    pseudonymize_identifier_columns,
    validate_residual_pii,
)
from .classification import (
    ACTION_HASH,
    FREE_TEXT,
    IDENTIFIER,
    QUASI_IDENTIFIER,
    apply_column_policies,
    classify_columns,
    classify_pii_columns,
    detect_gps_columns,
    detect_timestamp_columns,
    free_text_columns_from_policies,
)
from .keyvault import build_pseudonymizer_from_env
from .repository import (
    AuditDB,
    TableMapping,
    _fresh_opts,
    connect_audit_db,
    discover_table_mappings,
    read_delta,
    read_sql_table,
    run_purview_check,
    write_delta,
)

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class PipelineConfig:
    database_url: str | None
    purview_account_name: str | None
    k_anonymity_min: int
    key_vault_url: str | None = None
    key_vault_rsa_key_name: str | None = None
    key_vault_enabled: bool = True
    hash_salt: str | None = None
    quasi_identifier_cols: tuple[str, ...] = ()
    identifier_cols: tuple[str, ...] = ()
    source_base_uri: str | None = None
    target_base_uri: str | None = None
    sql_endpoint: str | None = None
    sql_database: str | None = None
    gps_precision: int = 2
    max_table_workers: int = 1

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            database_url=os.environ.get("DATABASE_URL"),
            purview_account_name=os.environ.get("PURVIEW_ACCOUNT_NAME"),
            k_anonymity_min=int(os.environ.get("K_ANONYMITY_MIN", "5")),
            key_vault_url=os.environ.get("KEY_VAULT_URL"),
            key_vault_rsa_key_name=os.environ.get("KEY_VAULT_RSA_KEY_NAME"),
            key_vault_enabled=os.environ.get("ENABLE_KEY_VAULT", "1").strip().lower() in {
                "1", "true", "yes", "on",
            },
            hash_salt=os.environ.get("HASH_SALT"),
            quasi_identifier_cols=_csv(os.environ.get("QUASI_IDENTIFIER_COLS", "")),
            identifier_cols=_csv(os.environ.get("IDENTIFIER_COLS", "")),
            source_base_uri=os.environ.get("SOURCE_BASE_ABFSS_URI"),
            target_base_uri=os.environ.get("TARGET_BASE_ABFSS_URI"),
            sql_endpoint=os.environ.get("SQL_ENDPOINT_URL"),
            sql_database=os.environ.get("SQL_DATABASE"),
            gps_precision=int(os.environ.get("GPS_PRECISION", "2")),
            max_table_workers=_env_int_at_least("MAX_TABLE_WORKERS", 1, 1),
        )


def _csv(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def _env_int_at_least(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be {minimum} or greater")
    return value


def _normalize_uri(uri: str) -> str:
    return uri.rstrip("/").lower()


@contextmanager
def timed_stage(audit: dict, name: str):
    start = perf_counter()
    try:
        yield
    finally:
        audit.setdefault("stage_seconds", {})[name] = round(perf_counter() - start, 6)


def resolve_table_mappings(config: PipelineConfig) -> list[TableMapping]:
    if not config.source_base_uri or not config.target_base_uri:
        raise RuntimeError(
            "SOURCE_BASE_ABFSS_URI and TARGET_BASE_ABFSS_URI must both be set."
        )
    mappings = discover_table_mappings(
        config.source_base_uri,
        config.target_base_uri,
        sql_endpoint=config.sql_endpoint,
        sql_database=config.sql_database,
    )
    if not mappings:
        raise RuntimeError(
            f"No tables found under {config.source_base_uri!r}. "
            "Ensure the path exists and contains at least one table subdirectory, "
            "or configure SQL_ENDPOINT_URL and SQL_DATABASE for shortcut discovery."
        )
    return mappings


def _new_audit(config: PipelineConfig) -> dict:
    return {
        "pipeline_end_ts": None,
        "total_rows_processed": 0,
        "total_columns_in_table": 0,
        "total_columns_scanned": 0,
        "columns_anonymized": [],
        "total_entities_detected": 0,
        "entity_counts": {},
        "unique_entities": {},
        "free_text_columns": [],
        "k_anonymity_k": config.k_anonymity_min,
        "quasi_columns": [],
        "suppressed_rows": 0,
        "residual_pii_count": 0,
        "column_renames": {},
        "gps_columns_anonymized": [],
        "timestamp_columns_binned": [],
        "numeric_columns_binned": [],
        "output_type": "anonymized_rows",
        "aggregate_cells": 0,
        "hashed_columns": [],
        "key_vault_key_version": None,
        "stage_seconds": {},
        "purview_available": False,
        "purview_flagged_columns": [],
        "purview_discrepancies": [],
        "status": "failure",
        "error_message": None,
    }


def record_alert(db: AuditDB | None, run_id: str | None, mapping: TableMapping | None, subject: str, body: str) -> None:
    if db is None:
        logger.warning("Audit DB unavailable; alert not persisted: %s", subject)
        return
    try:
        db.record_alert(run_id, mapping.table_name if mapping else None, subject, body)
    except Exception as exc:
        logger.error("Alert persistence failed: %s", exc)


def _read_source_table(config: PipelineConfig, mapping: TableMapping) -> pd.DataFrame:
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


def _profile_columns_by_category(profiles, category: str) -> list[str]:
    return [profile.name for profile in profiles if category in profile.categories]


def _configured_or_profiled_columns(configured: tuple[str, ...], profiles, category: str, df: pd.DataFrame) -> list[str]:
    if configured:
        return [col for col in configured if col in df.columns]
    return _profile_columns_by_category(profiles, category)


def _close_audit_run(db: AuditDB | None, run_id: str, audit: dict) -> None:
    if db is None:
        return
    try:
        db.close_run(run_id, audit)
    except Exception as exc:
        logger.warning("Audit close_run failed (non-fatal): %s", exc)


def run_table(config: PipelineConfig, mapping: TableMapping, db: AuditDB | None, run_id: str | None = None) -> dict:
    run_id = run_id or str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    audit = _new_audit(config)

    if _normalize_uri(mapping.source_uri) == _normalize_uri(mapping.target_uri):
        raise RuntimeError(
            "Source and target table URIs are identical. Refusing to overwrite the source table. "
            f"table_name={mapping.table_name!r} uri={mapping.source_uri}"
        )

    if db:
        try:
            db.open_run(run_id, started_at, mapping)
        except Exception as exc:
            logger.warning("Audit open_run failed (non-fatal): %s", exc)

    try:
        with timed_stage(audit, "read"):
            df_raw = _read_source_table(config, mapping)
        audit["total_rows_processed"] = len(df_raw)
        audit["total_columns_in_table"] = len(df_raw.columns)

        with timed_stage(audit, "gps_detection_and_transform"):
            gps_anonymized: list[str] = []
            gps_cols = detect_gps_columns(df_raw)
            if gps_cols:
                df_raw, gps_anonymized = anonymize_gps_columns(df_raw, gps_cols, config.gps_precision)
                audit["gps_columns_anonymized"] = gps_anonymized

            # Trajectory tables are GPS + speed + timestamp.
            ts_cols: list[str] = detect_timestamp_columns(df_raw) if gps_anonymized else []
            speed_col: str | None = detect_speed_column(df_raw) if gps_anonymized else None
            is_trajectory = bool(gps_anonymized) and speed_col is not None and bool(ts_cols)

        ts_binned: list[str] = []
        with timed_stage(audit, "timestamp_binning"):
            if not is_trajectory and gps_anonymized and ts_cols:
                df_raw, ts_binned = bin_timestamp_columns(df_raw, ts_cols)
                audit["timestamp_columns_binned"] = ts_binned

        with timed_stage(audit, "column_classification"):
            column_profiles = classify_columns(df_raw)

        pseudonymizer = None
        with timed_stage(audit, "identifier_pseudonymization"):
            id_cols = _configured_or_profiled_columns(config.identifier_cols, column_profiles, IDENTIFIER, df_raw)
            if id_cols:
                pseudonymizer = build_pseudonymizer_from_env(
                    config.key_vault_url,
                    config.key_vault_rsa_key_name,
                    enable_key_vault=config.key_vault_enabled,
                    hash_salt=config.hash_salt,
                )
                if pseudonymizer is None:
                    raise RuntimeError(
                        "Identifier columns detected but Key Vault is not configured. "
                        "Set KEY_VAULT_URL and KEY_VAULT_RSA_KEY_NAME, or set "
                        "ENABLE_KEY_VAULT=0 to use local hashing instead."
                    )
                df_raw, pseudonymized = pseudonymize_identifier_columns(df_raw, id_cols, pseudonymizer)
                audit["key_vault_key_version"] = pseudonymizer.key_version
            else:
                pseudonymized = []
        audit["hashed_columns"] = pseudonymized

        audit["free_text_columns"] = _profile_columns_by_category(column_profiles, FREE_TEXT)

        with timed_stage(audit, "purview_check"):
            pv = run_purview_check(mapping.source_uri, list(df_raw.columns), config.purview_account_name)
        _apply_purview_audit(audit, pv)

        with timed_stage(audit, "k_anonymity"):
            if not is_trajectory:
                qi_cols = _configured_or_profiled_columns(config.quasi_identifier_cols, column_profiles, QUASI_IDENTIFIER, df_raw)
                qi_cols = list(dict.fromkeys(gps_anonymized + ts_binned + qi_cols))
                audit["quasi_columns"] = qi_cols
                numeric_qi = [c for c in qi_cols if pd.api.types.is_numeric_dtype(df_raw[c])]
                if numeric_qi:
                    df_raw, num_binned = bin_numeric_columns(df_raw, numeric_qi)
                    audit["numeric_columns_binned"] = num_binned
                if qi_cols:
                    df_raw, k_info = enforce_k_anonymity(df_raw, qi_cols, config.k_anonymity_min)
                    audit["suppressed_rows"] = k_info["suppressed_rows"]

        with timed_stage(audit, "build_engines"):
            analyzer = build_engines()
        registry = EntityRegistry()

        # ── Column-policy classification (Phase 2/3 of the column-aware
        # PII layer).  Runs Tier A (Purview) → B1 (presidio-structured
        # value sampling) → B2 (spaCy embedding similarity) per column.
        # Hash/tokenise classified columns BEFORE row-by-row Presidio scans.
        # Failures here are non-fatal — the existing per-cell scan remains
        # the backstop.
        with timed_stage(audit, "column_policy_classification"):
            try:
                policies = classify_pii_columns(df_raw, analyzer=analyzer)
            except Exception as exc:
                logger.warning("Column-policy classification failed (non-fatal): %s", exc)
                policies = {}

        policy_needs_hash = any(p.action == ACTION_HASH for p in policies.values())
        policy_pseudonymizer = pseudonymizer if policy_needs_hash else None
        if policy_needs_hash and policy_pseudonymizer is None:
            policy_pseudonymizer = build_pseudonymizer_from_env(
                config.key_vault_url,
                config.key_vault_rsa_key_name,
                enable_key_vault=config.key_vault_enabled,
                hash_salt=config.hash_salt,
            )

        with timed_stage(audit, "column_policy_mask"):
            df_raw, policy_stats = apply_column_policies(
                df_raw, policies,
                registry=registry,
                pseudonymizer=policy_pseudonymizer,
            )
        audit["column_policy"] = {
            "columns_processed": policy_stats["columns_processed"],
            "actions_applied": policy_stats["actions_applied"],
            "entity_types": policy_stats["entity_types"],
            "values_masked": policy_stats["values_masked"],
            "skipped_columns": policy_stats["skipped_columns"],
            "sources": {
                col: pol.source for col, pol in policies.items()
            },
        }
        scan_columns = free_text_columns_from_policies(policies)

        with timed_stage(audit, "anonymization"):
            df_clean, stats = anonymize_dataframe(df_raw, analyzer, registry, scan_columns=scan_columns)
        _apply_anonymization_audit(audit, stats, registry)

        if db and stats["column_stats"]:
            try:
                db.record_columns(run_id, stats["column_stats"])
            except Exception as exc:
                logger.warning("Audit record_columns failed (non-fatal): %s", exc)

        if is_trajectory:
            with timed_stage(audit, "gps_aggregation"):
                df_clean, agg_stats = aggregate_gps_table(
                    df_clean, gps_anonymized, speed_col, ts_cols[0], config.k_anonymity_min,
                )
            audit["output_type"] = "gps_aggregate"
            audit["aggregate_cells"] = agg_stats["cells_retained"]
            audit["suppressed_rows"] = agg_stats["pings_suppressed"]
        else:
            with timed_stage(audit, "residual_validation"):
                audit["residual_pii_count"] = validate_residual_pii(df_clean)

        with timed_stage(audit, "write"):
            write_delta(df_clean, mapping.target_uri, _fresh_opts(mapping.target_uri))

        audit["pipeline_end_ts"] = datetime.now(timezone.utc).isoformat()
        audit["status"] = "success"
        logger.info("Table '%s' stage timings: %s", mapping.table_name or mapping.source_uri, audit["stage_seconds"])
        return audit
    except Exception as exc:
        audit["pipeline_end_ts"] = datetime.now(timezone.utc).isoformat()
        audit["error_message"] = str(exc)
        record_alert(db, run_id, mapping, "Pipeline FAILED", f"source: {mapping.source_uri}\nerror: {exc}")
        raise
    finally:
        _close_audit_run(db, run_id, audit)


def run_pipeline(config: PipelineConfig | None = None) -> list[dict]:
    config = config or PipelineConfig.from_env()
    db = connect_audit_db(config.database_url)
    mappings = resolve_table_mappings(config)
    if config.max_table_workers <= 1 or len(mappings) <= 1:
        return [run_table(config, mapping, db) for mapping in mappings]

    workers = min(config.max_table_workers, len(mappings))
    logger.info("Processing %d table(s) with %d process worker(s)", len(mappings), workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_table_worker, config, mapping) for mapping in mappings]
        return [future.result() for future in futures]


def _run_table_worker(config: PipelineConfig, mapping: TableMapping) -> dict:
    db = connect_audit_db(config.database_url)
    return run_table(config, mapping, db)
