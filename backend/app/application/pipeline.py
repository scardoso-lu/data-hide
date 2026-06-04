"""Pipeline orchestration — application layer.

Coordinates the domain logic (classification, anonymization, aggregation) with
the infrastructure adapters (repository, key vault) to execute the full
read → classify → anonymize → write pipeline for every discovered table.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import logging
import os
from time import perf_counter
import uuid

import polars as pl

from ..domain.aggregation import aggregate_gps_table, detect_speed_column
from ..domain.anonymization import (
    EntityRegistry,
    anonymize_dataframe,
    anonymize_gps_columns,
    bin_numeric_columns,
    bin_timestamp_columns,
    build_engines,
    enforce_k_anonymity,
    pseudonymize_identifier_columns,
    release_engines,
    summarize_residual_pii,
    validate_residual_pii,
)
from ..domain.classification import (
    ACTION_HASH,
    FREE_TEXT,
    IDENTIFIER,
    QUASI_IDENTIFIER,
    _tier_a_purview,
    apply_column_policies,
    classify_columns,
    classify_pii_columns,
    classify_pii_columns_multi_pass,
    detect_gps_columns,
    detect_timestamp_columns,
    free_text_columns_from_policies,
    release_sequential_model,
)
from ..infrastructure.keyvault import LocalHashPseudonymizer, build_pseudonymizer_from_env
from ..infrastructure.repository import (
    AuditDB,
    TableMapping,
    _clear_caches,
    _fresh_opts,
    connect_audit_db,
    discover_table_mappings,
    process_rss_mb,
    read_delta,
    read_delta_sample,
    read_sql_table,
    run_purview_check,
    write_delta,
)
from .config import (
    PipelineConfig,
    _csv,
    _env_int_at_least,
    _parse_k_anonymity_tables,
    _parse_table_qi_cols,
    _QI_PREFIX,
)
from .stages import (
    _apply_anonymization_audit,
    _apply_purview_audit,
    _column_policy_stats,
    _configured_or_profiled_columns,
    _merge_scan_stats,
    _profile_columns_by_category,
    _read_source_sample,
    _read_source_table,
)

logger = logging.getLogger(__name__)


def _normalize_uri(uri: str) -> str:
    return uri.rstrip("/").lower()


@contextmanager
def timed_stage(audit: dict, name: str):
    start = perf_counter()
    try:
        yield
    finally:
        audit.setdefault("stage_seconds", {})[name] = round(perf_counter() - start, 6)
        # Per-stage RSS so OOM kills can be attributed from the container log:
        # the last "stage_rss" line before exit code 137 names the stage that
        # was running when the limit was hit, and the audit row (pre-populated
        # by open_run) survives even though the kill bypasses finally-blocks
        # in the data path.
        rss = process_rss_mb()
        if rss is not None:
            audit.setdefault("stage_rss_mb", {})[name] = round(rss)
            logger.info("stage_rss: stage=%s rss=%d MB", name, round(rss))


def resolve_table_mappings(config: PipelineConfig) -> list[TableMapping]:
    if config.table_targets:
        return list(config.table_targets)
    if not config.source_base_uri or not config.target_base_uri:
        raise RuntimeError(
            "SOURCE_BASE_ABFSS_URI and TARGET_BASE_ABFSS_URI must both be set "
            "(or populate pii_table_targets in the audit database)."
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


def _close_audit_run(db: AuditDB | None, run_id: str, audit: dict) -> None:
    if db is None:
        return
    try:
        db.close_run(run_id, audit)
    except Exception as exc:
        logger.warning("Audit close_run failed (non-fatal): %s", exc)


def run_table(
    config: PipelineConfig,
    mapping: TableMapping,
    db: AuditDB | None,
    run_id: str | None = None,
    *,
    analyzer=None,
    policies: dict | None = None,
) -> dict:
    """Process one table end to end.

    ``policies`` — pre-computed column policies from the Phase 1
    language-major classification passes (see ``run_pipeline``).  When
    provided, no NLP engine or spaCy model is needed (or loaded) for this
    table: classification already happened on a small sample, and every
    remaining stage (masking, binning, k-anonymity, write) is model-free.
    """
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
        logger.info(
            "Table '%s': read %d row(s), %d column(s)",
            mapping.table_name or mapping.source_uri,
            len(df_raw),
            len(df_raw.columns),
        )

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
                    logger.warning(
                        "Key Vault not configured and ENABLE_KEY_VAULT is not "
                        "explicitly set to '0' — falling back to local HMAC "
                        "hashing for identifier columns.  Set KEY_VAULT_URL + "
                        "KEY_VAULT_RSA_KEY_NAME for HSM-bound pseudonymization, "
                        "or add ENABLE_KEY_VAULT=0 to .env to suppress this warning."
                    )
                    pseudonymizer = LocalHashPseudonymizer(config.hash_salt)
                df_raw, pseudonymized = pseudonymize_identifier_columns(df_raw, id_cols, pseudonymizer, inplace=True)
                audit["key_vault_key_version"] = pseudonymizer.key_version
            else:
                pseudonymized = []
        audit["hashed_columns"] = pseudonymized

        audit["free_text_columns"] = _profile_columns_by_category(column_profiles, FREE_TEXT)

        with timed_stage(audit, "purview_check"):
            pv = run_purview_check(mapping.source_uri, list(df_raw.columns), config.purview_account_name)
        _apply_purview_audit(audit, pv)

        # Lowercase table name used by both k-anonymity and column-exclusion
        # lookups below — computed once here to avoid repetition.
        _table_key = (mapping.table_name or "").lower()

        with timed_stage(audit, "k_anonymity"):
            # K-anonymity runs only for tables explicitly listed in
            # K_ANONYMITY_TABLES.  All other tables are skipped so that
            # business tables (HR, finance, absence, …) are never silently
            # suppressed without an explicit operator decision.
            _table_qi_cols = config.quasi_identifier_cols_by_table.get(_table_key, ())
            if not is_trajectory and _table_key in config.k_anonymity_tables:
                qi_cols = _configured_or_profiled_columns(_table_qi_cols, column_profiles, QUASI_IDENTIFIER, df_raw)
                qi_cols = list(dict.fromkeys(gps_anonymized + ts_binned + qi_cols))
                audit["quasi_columns"] = qi_cols
                numeric_qi = [c for c in qi_cols if df_raw.schema[c].is_numeric()]
                if numeric_qi:
                    df_raw, num_binned = bin_numeric_columns(df_raw, numeric_qi)
                    audit["numeric_columns_binned"] = num_binned
                if qi_cols:
                    df_raw, k_info = enforce_k_anonymity(df_raw, qi_cols, config.k_anonymity_min)
                    audit["suppressed_rows"] = k_info["suppressed_rows"]
            else:
                logger.debug(
                    "Table '%s': skipping k-anonymity (not listed in K_ANONYMITY_TABLES).",
                    mapping.table_name or mapping.source_uri,
                )
                audit["quasi_columns"] = []

        registry = EntityRegistry()

        # ── Column-policy classification (Phase 2/3 of the column-aware
        # PII layer).  Runs Tier A (Purview) → B1 (presidio-structured
        # value sampling) → B2 (spaCy embedding similarity) per column.
        # Hash/tokenise classified columns BEFORE row-by-row Presidio scans.
        # Failures here are non-fatal — the existing per-cell scan remains
        # the backstop.
        #
        # Preferred flow: ``policies`` arrives pre-computed from the Phase 1
        # language-major sample passes in run_pipeline — no model is loaded
        # in this function at all.  The inline branch below remains for
        # direct callers (parallel workers, tests).
        with timed_stage(audit, "column_policy_classification"):
            if policies is not None:
                policies = dict(policies)
                # Apply Purview overrides to pre-computed policies so that
                # columns classified by the multi-pass phase are still
                # subject to operator-level Purview pins.
                _tier_a_purview(
                    df_raw,
                    pv.get("column_labels") or None,
                    policies,
                    purview_must_anonymize_type=config.purview_must_anonymize_type,
                )
            else:
                if analyzer is None:
                    with timed_stage(audit, "build_engines"):
                        # English-only — see the comment in run_pipeline().
                        analyzer = build_engines(("en",))
                try:
                    policies = classify_pii_columns(
                        df_raw,
                        analyzer=analyzer,
                        purview_classifications=pv.get("column_labels") or None,
                        purview_must_anonymize_type=config.purview_must_anonymize_type,
                    )
                except Exception as exc:
                    logger.warning("Column-policy classification failed (non-fatal): %s", exc)
                    policies = {}

        # Columns pseudonymized above already hold deterministic hashes —
        # masking them again (e.g. a Tier B1 PERSON vote from raw sample
        # values) would tokenise the hashes and break join keys.
        if pseudonymized:
            policies = {c: p for c, p in policies.items() if c not in set(pseudonymized)}

        # Drop columns that the operator explicitly excluded from anonymization
        # (pii_column_exclusions table in PostgreSQL).  These columns pass
        # through untouched regardless of what the classification tiers found.
        _excluded = config.excluded_columns_by_table.get(_table_key, frozenset())

        # Purview NOT_PII safelist — columns Purview has verified as non-PII
        # are added to the exclusion set for this run, bypassing both
        # column-policy masking and row-level Presidio.
        if config.purview_not_pii_type and pv.get("column_labels"):
            _not_pii_upper = config.purview_not_pii_type.upper()
            purview_safe = frozenset(
                col for col, types in pv["column_labels"].items()
                if any(
                    t.upper() == _not_pii_upper
                    for t in (types if isinstance(types, list) else [types])
                )
            )
            if purview_safe:
                _excluded = _excluded | purview_safe
        if _excluded:
            before = len(policies)
            policies = {col: pol for col, pol in policies.items() if col not in _excluded}
            logger.info(
                "Table '%s': %d column(s) excluded from anonymization by operator rule: %s",
                mapping.table_name or mapping.source_uri,
                before - len(policies),
                sorted(_excluded),
            )

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
                inplace=True,
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
        # Audit stats start from the column-policy layer (HASH / TOKENIZE /
        # REDACT already applied above → policy_stats); the targeted row-by-row
        # scan below adds to them.
        stats = _column_policy_stats(policies, policy_stats)

        # Targeted row-by-row Presidio scan. Operators opt specific columns into
        # cell-by-cell scanning by listing the (table, column) pair in the
        # apply_row_scan table — loaded into config.row_scan_columns_by_table.
        # This is the "more targeted approach" that replaces the globally-
        # disabled row scan: only listed columns of a matching table are scanned,
        # so the false positives from scanning every text column are avoided.
        # Columns already masked by a column policy are skipped (they no longer
        # hold raw values).
        described = config.row_scan_columns_by_table.get(_table_key, frozenset())
        already_masked = set(policy_stats["columns_processed"])
        row_scan_columns = [
            c for c in described if c in df_raw.columns and c not in already_masked
        ]
        with timed_stage(audit, "anonymization"):
            df_clean = df_raw
            if row_scan_columns:
                if analyzer is None:
                    with timed_stage(audit, "build_engines"):
                        analyzer = build_engines()
                logger.info(
                    "Table '%s': targeted row-by-row scan on %d operator-described "
                    "column(s): %s",
                    mapping.table_name or mapping.source_uri,
                    len(row_scan_columns), sorted(row_scan_columns),
                )
                df_clean, scan_stats = anonymize_dataframe(
                    df_clean, analyzer, registry,
                    scan_columns=row_scan_columns, inplace=True,
                )
                _merge_scan_stats(stats, scan_stats)
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
                # Residual PII is NON-FATAL: record the count, raise an alert,
                # and still write the (partially-anonymized) table. A single
                # over-eager recognizer (e.g. false-positive PHONE_NUMBER hits
                # on a numeric column) must not abort the run or withhold the
                # output. The count + summary are persisted to the audit row.
                residual_count, residual_detail = summarize_residual_pii(df_clean)
                audit["residual_pii_count"] = residual_count
                if residual_count:
                    logger.warning(
                        "Table '%s': %d residual PII finding(s) after anonymization "
                        "(table still written): %s",
                        mapping.table_name or mapping.source_uri,
                        residual_count,
                        residual_detail,
                    )
                    record_alert(
                        db, run_id, mapping, "Residual PII detected",
                        f"source: {mapping.source_uri}\n"
                        f"{residual_count} residual finding(s): {residual_detail}",
                    )

        rows_out = len(df_clean)
        if rows_out == 0:
            logger.warning(
                "Table '%s': 0 rows remain after processing "
                "(source=%d, suppressed=%d) — writing empty table.",
                mapping.table_name or mapping.source_uri,
                audit["total_rows_processed"],
                audit["suppressed_rows"],
            )
        else:
            logger.info(
                "Table '%s': %d → %d row(s) written (%d suppressed by k-anonymity).",
                mapping.table_name or mapping.source_uri,
                audit["total_rows_processed"],
                rows_out,
                audit["suppressed_rows"],
            )

        with timed_stage(audit, "write"):
            write_delta(df_clean, mapping.target_uri, _fresh_opts(mapping.target_uri))

        audit["pipeline_end_ts"] = datetime.now(timezone.utc).isoformat()
        audit["status"] = "success"
        logger.info(
            "Table '%s' OK — %d→%d rows, %d suppressed. Timings: %s",
            mapping.table_name or mapping.source_uri,
            audit["total_rows_processed"],
            rows_out,
            audit["suppressed_rows"],
            audit["stage_seconds"],
        )
        return audit
    except Exception as exc:
        audit["pipeline_end_ts"] = datetime.now(timezone.utc).isoformat()
        audit["error_message"] = str(exc)
        record_alert(db, run_id, mapping, "Pipeline FAILED", f"source: {mapping.source_uri}\nerror: {exc}")
        raise
    finally:
        _close_audit_run(db, run_id, audit)


def _release_between_tables() -> None:
    """Drop refs held by per-table caches and return freed pages to the OS.

    Between Phase 2 tables: collect garbage (the previous table's Polars
    frame, EntityRegistry, and stats are now unreferenced) and trim the glibc
    heap so RSS actually falls instead of plateauing at the high-water mark.
    Also clears the process-wide language-detection cache and all repository
    caches so Azure resource GUIDs don't accumulate across tables.
    """
    import gc
    from ..domain.anonymization import _detect_language
    from ..infrastructure.nlp import _trim_native_heap

    try:
        _detect_language.cache_clear()
    except Exception:
        pass
    _clear_caches()
    gc.collect()
    _trim_native_heap()


def run_pipeline(config: PipelineConfig | None = None) -> list[dict]:
    if config is None:
        # Connect to DB first so runtime config and column exclusions can be
        # loaded before the full PipelineConfig is built.  DATABASE_URL must
        # come from the environment — it bootstraps the DB connection itself.
        db = connect_audit_db(os.environ.get("DATABASE_URL"))
        config = PipelineConfig.from_env_and_db(db)
    else:
        db = connect_audit_db(config.database_url)
    mappings = resolve_table_mappings(config)

    if config.max_table_workers <= 1 or len(mappings) <= 1:
        # Language-major two-phase flow — at most ONE spaCy model resident:
        #
        # Phase 1 (samples only, ≤500 rows per table):
        #   load EN engine → classify every table (A1 + B1 + B2-en) →
        #   release engine → FR model → score all tables → release →
        #   DE model → score all tables → release (lb shares DE's model).
        # Phase 2 (zero models resident):
        #   per table: full read → apply policies → transforms → write.
        #
        # Classification only ever needs column names + a value sample, so
        # the full tables are read exactly once, in Phase 2, with no model
        # in memory.  NOTE: if the row-by-row Presidio scan is re-enabled,
        # revisit — cell-level analysis needs an engine during Phase 2.
        samples: list = []
        for mapping in mappings:
            try:
                samples.append(_read_source_sample(config, mapping))
            except Exception as exc:
                logger.warning(
                    "Sample read failed for table %r (non-fatal — empty policies): %s",
                    mapping.table_name, exc,
                )
                samples.append(None)

        policies_by_table: list[dict] = [{} for _ in mappings]
        classified = [(i, s) for i, s in enumerate(samples) if s is not None]
        if classified:
            analyzer = build_engines(("en",))
            try:
                results = classify_pii_columns_multi_pass(
                    [s for _, s in classified], analyzer=analyzer,
                )
                for (i, _), table_policies in zip(classified, results):
                    policies_by_table[i] = table_policies
            except Exception as exc:
                logger.warning("Multi-pass classification failed (non-fatal): %s", exc)
            finally:
                release_engines()
                release_sequential_model()
        del samples

        # Phase 2: full-table processing, one table at a time.  Each table's
        # frame is dropped and the native heap trimmed before the next read,
        # so RSS does not stack table-over-table — peak ≈ one table, not the
        # sum of all tables.
        results: list[dict] = []
        for i, mapping in enumerate(mappings):
            try:
                results.append(run_table(config, mapping, db, policies=policies_by_table[i]))
            except Exception as exc:
                # Per-table isolation: run_table already recorded the failure
                # (audit status=failure + alert) before re-raising. A single
                # table's error (Key Vault, read/write, classification, …) must
                # not stop the run — log it and continue with the rest.
                logger.error(
                    "Table %r failed — recorded and skipped; continuing with remaining tables: %s",
                    mapping.table_name or mapping.source_uri, exc,
                )
            finally:
                _release_between_tables()
        return results

    # Parallel path: each worker is a separate process (ProcessPoolExecutor),
    # so objects cannot be shared across the process boundary. Each worker
    # builds its own engines. build_engines() is intentionally not called here.
    workers = min(config.max_table_workers, len(mappings))
    logger.info("Processing %d table(s) with %d process worker(s)", len(mappings), workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_table_worker, config, mapping): mapping for mapping in mappings}
        results: list[dict] = []
        for future, mapping in futures.items():
            try:
                results.append(future.result())
            except Exception as exc:
                # Per-table isolation (parallel path): one worker's failure must
                # not abort the others. The worker recorded the failure audit
                # before raising.
                logger.error(
                    "Table %r failed in worker — recorded and skipped; continuing: %s",
                    mapping.table_name or mapping.source_uri, exc,
                )
        return results


def _run_table_worker(config: PipelineConfig, mapping: TableMapping) -> dict:
    db = connect_audit_db(config.database_url)
    return run_table(config, mapping, db)
