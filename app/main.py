"""Thin entrypoint for the Fabric PII anonymization pipeline."""

from __future__ import annotations

import logging
import sys

from .anonymization import (
    EntityRegistry,
    _anonymize_json,
    _anonymize_text,
    _scan_json_for_pii,
    anonymize_dataframe,
    anonymize_gps_columns,
    bin_timestamp_columns,
    build_engines,
    enforce_k_anonymity,
    hash_identifier_columns,
    validate_residual_pii,
)
from .classification import (
    classify_columns,
    detect_gps_columns,
    detect_identifier_columns,
    detect_quasi_identifiers,
    detect_timestamp_columns,
    flag_free_text_columns,
    sanitize_column_names,
)
from .repository import (
    DefaultAzureCredential,
    DeltaTable,
    PIPELINE_VERSION,
    AuditDB,
    PurviewClient,
    TableMapping,
    _account_name,
    _fresh_opts,
    _storage_opts,
    acquire_token,
    connect_audit_db,
    discover_table_mappings,
    read_delta,
    read_sql_table,
    write_delta,
    write_deltalake,
)
from . import repository as _repository
from . import service as _service
from .service import PipelineConfig, record_alert, resolve_table_mappings, run_pipeline, run_table

psycopg2 = _repository.psycopg2


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main() -> None:
    _repository.DefaultAzureCredential = DefaultAzureCredential
    _repository.DeltaTable = DeltaTable
    _repository.write_deltalake = write_deltalake
    _service.connect_audit_db = connect_audit_db
    _service.discover_table_mappings = discover_table_mappings
    _service.read_delta = read_delta
    _service.read_sql_table = read_sql_table
    _service.write_delta = write_delta
    _service.run_purview_check = run_purview_check
    _service.build_engines = build_engines
    _service.anonymize_dataframe = anonymize_dataframe
    _service.validate_residual_pii = validate_residual_pii
    _service.sanitize_column_names = sanitize_column_names
    _service.flag_free_text_columns = flag_free_text_columns
    _service.detect_quasi_identifiers = detect_quasi_identifiers
    _service.detect_gps_columns = detect_gps_columns
    _service.anonymize_gps_columns = anonymize_gps_columns
    _service.detect_timestamp_columns = detect_timestamp_columns
    _service.bin_timestamp_columns = bin_timestamp_columns
    _service.detect_identifier_columns = detect_identifier_columns
    _service.hash_identifier_columns = hash_identifier_columns
    _service.enforce_k_anonymity = enforce_k_anonymity
    _service.record_alert = record_alert
    run_pipeline(PipelineConfig.from_env())


def run_purview_check(source_uri: str, df_columns: list[str], purview_account: str | None) -> dict:
    """Compatibility wrapper that keeps old tests patching main.* working."""
    original_client = _repository.PurviewClient
    original_token = _repository.acquire_token
    try:
        _repository.PurviewClient = PurviewClient
        _repository.acquire_token = acquire_token
        return _repository.run_purview_check(source_uri, df_columns, purview_account)
    finally:
        _repository.PurviewClient = original_client
        _repository.acquire_token = original_token


if __name__ == "__main__":
    main()
