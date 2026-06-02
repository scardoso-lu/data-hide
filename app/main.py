"""Thin entrypoint for the Fabric PII anonymization pipeline."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from importlib import invalidate_caches
from importlib.util import find_spec

from .infrastructure import repository as _repository
from .application import pipeline as _service
from .domain.aggregation import aggregate_gps_table, detect_speed_column
from .domain.anonymization import (
    SPACY_MODELS,
    EntityRegistry,
    _anonymize_json,
    _anonymize_text,
    anonymize_dataframe,
    anonymize_gps_columns,
    bin_numeric_columns,
    bin_timestamp_columns,
    build_engines as _build_engines,
    enforce_k_anonymity,
    pseudonymize_identifier_columns,
    validate_residual_pii,
)
from .infrastructure.keyvault import KeyVaultPseudonymizer, build_pseudonymizer_from_env
from .domain.classification import (
    classify_columns,
    detect_gps_columns,
    detect_identifier_columns,
    detect_quasi_identifiers,
    detect_timestamp_columns,
    flag_free_text_columns,
)
from .infrastructure.repository import (
    PIPELINE_VERSION,
    AuditDB,
    DefaultAzureCredential,
    DeltaTable,
    PurviewClient,
    TableMapping,
    _account_name,
    _fresh_opts,
    _storage_opts,
    acquire_cached_token,
    acquire_token,
    connect_audit_db,
    discover_table_mappings,
    read_delta,
    read_sql_table,
    write_delta,
)
from .application.pipeline import PipelineConfig, record_alert, resolve_table_mappings, run_pipeline, run_table

psycopg2 = _repository.psycopg2


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Quiet third-party loggers that dump every HTTP request/response or every
# spaCy inference at INFO level.  Their WARNING+ output is still surfaced.
#
# * Azure SDK HTTP policy: would otherwise log every Delta upload's HEAD
#   probe — those routinely return 404 (BlobNotFound) on first write because
#   we replace-on-write the table directory, which is expected behaviour.
# * Presidio analyzer: noisy DEBUG/INFO during recognizer initialisation.
for noisy_logger in (
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.identity",
    "azure.storage",
):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

_MODELS_DIR = os.environ.get("SPACY_MODELS_DIR")


def _ensure_spacy_models() -> None:
    models_dir = os.path.abspath(_MODELS_DIR or "models")
    os.makedirs(models_dir, exist_ok=True)
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)

    for model in set(SPACY_MODELS.values()):
        if find_spec(model) is None:
            logger.info("Downloading spaCy model %s -> %s", model, models_dir)
            _download_spacy_model(model, models_dir)
            invalidate_caches()
        else:
            logger.info("Using cached spaCy model %s from %s", model, models_dir)

        if find_spec(model) is None:
            raise RuntimeError(f"spaCy model {model} was downloaded but is not importable from {models_dir}")


def _download_spacy_model(model: str, models_dir: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "spacy",
            "download",
            model,
            "--target",
            models_dir,
            "--no-cache-dir",
        ],
        check=True,
    )


def build_engines():
    _ensure_spacy_models()
    return _build_engines()


def main() -> None:
    _repository.DefaultAzureCredential = DefaultAzureCredential
    _repository.DeltaTable = DeltaTable
    _service.connect_audit_db = connect_audit_db
    _service.discover_table_mappings = discover_table_mappings
    _service.read_delta = read_delta
    _service.read_sql_table = read_sql_table
    _service.write_delta = write_delta
    _service.run_purview_check = run_purview_check
    _service.build_engines = build_engines
    _service.anonymize_dataframe = anonymize_dataframe
    _service.validate_residual_pii = validate_residual_pii
    _service.detect_speed_column = detect_speed_column
    _service.aggregate_gps_table = aggregate_gps_table
    _service.classify_columns = classify_columns
    _service.detect_gps_columns = detect_gps_columns
    _service.anonymize_gps_columns = anonymize_gps_columns
    _service.detect_timestamp_columns = detect_timestamp_columns
    _service.bin_numeric_columns = bin_numeric_columns
    _service.bin_timestamp_columns = bin_timestamp_columns
    _service.pseudonymize_identifier_columns = pseudonymize_identifier_columns
    _service.build_pseudonymizer_from_env = build_pseudonymizer_from_env
    _service.enforce_k_anonymity = enforce_k_anonymity
    run_pipeline()  # connects to DB first, then calls PipelineConfig.from_env_and_db()


def run_purview_check(source_uri: str, df_columns: list[str], purview_account: str | None) -> dict:
    """Wrapper that ensures the real PurviewClient (SDK-backed) is used at
    runtime while still allowing tests to patch ``main.PurviewClient``."""
    original_client = _repository.PurviewClient
    try:
        _repository.PurviewClient = PurviewClient
        return _repository.run_purview_check(source_uri, df_columns, purview_account)
    finally:
        _repository.PurviewClient = original_client



if __name__ == "__main__":
    main()
