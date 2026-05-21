"""Pipeline orchestration service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
import uuid

from .anonymization import (
    EntityRegistry,
    anonymize_dataframe,
    anonymize_gps_columns,
    build_engines,
    enforce_k_anonymity,
    hash_identifier_columns,
    validate_residual_pii,
)
from .classification import (
    detect_gps_columns,
    detect_identifier_columns,
    detect_quasi_identifiers,
    flag_free_text_columns,
    sanitize_column_names,
)
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
    hash_salt: str
    quasi_identifier_cols: tuple[str, ...] = ()
    identifier_cols: tuple[str, ...] = ()
    source_base_uri: str | None = None
    target_base_uri: str | None = None
    sql_endpoint: str | None = None
    sql_database: str | None = None
    gps_precision: int = 2

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            database_url=os.environ.get("DATABASE_URL"),
            purview_account_name=os.environ.get("PURVIEW_ACCOUNT_NAME"),
            k_anonymity_min=int(os.environ.get("K_ANONYMITY_MIN", "5")),
            hash_salt=os.environ.get("HASH_SALT", ""),
            quasi_identifier_cols=_csv(os.environ.get("QUASI_IDENTIFIER_COLS", "")),
            identifier_cols=_csv(os.environ.get("IDENTIFIER_COLS", "")),
            source_base_uri=os.environ.get("SOURCE_BASE_ABFSS_URI"),
            target_base_uri=os.environ.get("TARGET_BASE_ABFSS_URI"),
            sql_endpoint=os.environ.get("SQL_ENDPOINT_URL"),
            sql_database=os.environ.get("SQL_DATABASE"),
            gps_precision=int(os.environ.get("GPS_PRECISION", "2")),
        )


def _csv(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def _normalize_uri(uri: str) -> str:
    return uri.rstrip("/").lower()


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
        "hashed_columns": [],
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


def run_table(config: PipelineConfig, mapping: TableMapping, db: AuditDB | None, run_id: str | None = None) -> dict:
    run_id = run_id or str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    audit = _new_audit(config)

    if _normalize_uri(mapping.source_uri) == _normalize_uri(mapping.target_uri):
        raise RuntimeError(
            "Source and target table URIs are identical. Refusing to overwrite the source table. "
            f"table_name={mapping.table_name!r} uri={mapping.source_uri}"
        )
    if not config.hash_salt:
        logger.warning("HASH_SALT is not set; identifier hashes are unsalted.")

    if db:
        try:
            db.open_run(run_id, started_at, mapping)
        except Exception as exc:
            logger.warning("Audit open_run failed (non-fatal): %s", exc)

    try:
        if mapping.read_mode == "sql":
            df_raw = read_sql_table(mapping.table_name, config.sql_endpoint, config.sql_database)
        else:
            df_raw = read_delta(mapping.source_uri, _fresh_opts(mapping.source_uri))
        audit["total_rows_processed"] = len(df_raw)
        audit["total_columns_in_table"] = len(df_raw.columns)

        df_raw, col_renames = sanitize_column_names(df_raw)
        audit["column_renames"] = col_renames

        gps_cols = detect_gps_columns(df_raw)
        if gps_cols:
            df_raw, gps_anonymized = anonymize_gps_columns(df_raw, gps_cols, config.gps_precision)
            audit["gps_columns_anonymized"] = gps_anonymized

        id_cols = detect_identifier_columns(df_raw, list(config.identifier_cols))
        df_raw, hashed = hash_identifier_columns(df_raw, id_cols, config.hash_salt)
        audit["hashed_columns"] = hashed

        free_text_cols = flag_free_text_columns(df_raw)
        audit["free_text_columns"] = free_text_cols

        pv = run_purview_check(mapping.source_uri, list(df_raw.columns), config.purview_account_name)
        audit["purview_available"] = pv["available"]
        audit["purview_flagged_columns"] = pv["flagged_columns"]
        audit["purview_discrepancies"] = pv["discrepancies"]

        qi_cols = detect_quasi_identifiers(df_raw, list(config.quasi_identifier_cols))
        audit["quasi_columns"] = qi_cols
        if qi_cols:
            df_raw, k_info = enforce_k_anonymity(df_raw, qi_cols, config.k_anonymity_min)
            audit["suppressed_rows"] = k_info["suppressed_rows"]

        analyzer = build_engines()
        registry = EntityRegistry()
        df_clean, stats = anonymize_dataframe(df_raw, analyzer, registry)
        audit["total_columns_scanned"] = len(stats["text_columns_scanned"])
        audit["columns_anonymized"] = stats["columns_with_detections"]
        audit["total_entities_detected"] = stats["total_entities_detected"]
        audit["entity_counts"] = stats["entity_counts"]
        audit["unique_entities"] = registry.unique_counts()

        if db and stats["column_stats"]:
            try:
                db.record_columns(run_id, stats["column_stats"])
            except Exception as exc:
                logger.warning("Audit record_columns failed (non-fatal): %s", exc)

        audit["residual_pii_count"] = validate_residual_pii(df_clean, analyzer)
        write_delta(df_clean, mapping.target_uri, _fresh_opts(mapping.target_uri))

        audit["pipeline_end_ts"] = datetime.now(timezone.utc).isoformat()
        audit["status"] = "success"
        return audit
    except Exception as exc:
        audit["pipeline_end_ts"] = datetime.now(timezone.utc).isoformat()
        audit["error_message"] = str(exc)
        record_alert(db, run_id, mapping, "Pipeline FAILED", f"source: {mapping.source_uri}\nerror: {exc}")
        raise
    finally:
        if db:
            try:
                db.close_run(run_id, audit)
            except Exception as exc:
                logger.warning("Audit close_run failed (non-fatal): %s", exc)


def run_pipeline(config: PipelineConfig | None = None) -> list[dict]:
    config = config or PipelineConfig.from_env()
    db = connect_audit_db(config.database_url)
    mappings = resolve_table_mappings(config)
    return [run_table(config, mapping, db) for mapping in mappings]
