"""External persistence and Fabric repository adapters."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
import json
import logging
import re
from datetime import datetime
from typing import Generator, Optional

import pandas as pd
import requests

try:
    import psycopg2
    import psycopg2.extras
except ModuleNotFoundError:
    class _MissingPsycopg2:
        class extras:
            @staticmethod
            def execute_values(*args, **kwargs):
                raise ModuleNotFoundError("No module named 'psycopg2'")

        @staticmethod
        def connect(*args, **kwargs):
            raise ModuleNotFoundError("No module named 'psycopg2'")

    psycopg2 = _MissingPsycopg2()

ONELAKE_TOKEN_SCOPE = "https://storage.azure.com/.default"
PURVIEW_TOKEN_SCOPE = "https://purview.azure.net/.default"
PIPELINE_VERSION = "2.2.0"

logger = logging.getLogger(__name__)
_credential: Optional[object] = None
DefaultAzureCredential = None
DeltaTable = None
write_deltalake = None
DataLakeServiceClient = None


@dataclass(frozen=True)
class TableMapping:
    source_uri: str
    target_uri: str
    table_name: str | None = None


def _credential_instance() -> object:
    global DefaultAzureCredential
    if DefaultAzureCredential is None:
        from azure.identity import DefaultAzureCredential as _DefaultAzureCredential
        DefaultAzureCredential = _DefaultAzureCredential

    global _credential
    if _credential is None:
        _credential = DefaultAzureCredential()
    return _credential


def acquire_token(scope: str) -> str:
    token = _credential_instance().get_token(scope)
    return token.token


def _account_name(abfss_uri: str) -> str:
    m = re.search(r"@([^.@/]+)\.", abfss_uri)
    if not m:
        raise ValueError(
            f"Cannot parse account name from URI: '{abfss_uri}'. "
            "Expected: abfss://container@account.dfs.fabric.microsoft.com/..."
        )
    return m.group(1)


def _parse_abfss_uri(abfss_uri: str) -> tuple[str, str, str]:
    m = re.match(r"^abfss://([^@/]+)@([^/]+)/(.+)$", abfss_uri)
    if not m:
        raise ValueError(
            f"Cannot parse ABFSS URI: '{abfss_uri}'. "
            "Expected: abfss://filesystem@host/path"
        )
    return m.group(1), m.group(2), m.group(3)


def _parquet_file_path(uri_path: str) -> str:
    path = uri_path.strip("/")
    if path.lower().endswith(".parquet"):
        return path
    return f"{path}/part-00000.parquet"


def _storage_opts(uri: str, token: str) -> dict:
    return {"account_name": _account_name(uri), "bearer_token": token}


def _fresh_opts(uri: str) -> dict:
    return _storage_opts(uri, acquire_token(ONELAKE_TOKEN_SCOPE))


def read_delta(uri: str, storage_options: dict) -> pd.DataFrame:
    global DeltaTable
    if DeltaTable is None:
        from deltalake import DeltaTable as _DeltaTable
        DeltaTable = _DeltaTable

    logger.info("Reading Delta table uri='%s'", uri)
    return DeltaTable(uri, storage_options=storage_options).to_pandas()


def write_delta(df: pd.DataFrame, uri: str, storage_options: dict) -> None:
    """Write a single Parquet file to OneLake/ADLS.

    The function name is kept for compatibility with existing tests and call
    sites; the output is intentionally Parquet, not Delta.
    """
    global DataLakeServiceClient
    if DataLakeServiceClient is None:
        from azure.storage.filedatalake import DataLakeServiceClient as _DataLakeServiceClient
        DataLakeServiceClient = _DataLakeServiceClient

    filesystem, host, uri_path = _parse_abfss_uri(uri)
    file_path = _parquet_file_path(uri_path)
    account_url = f"https://{host}"

    logger.info("Writing Parquet file uri='abfss://%s@%s/%s'", filesystem, host, file_path)
    buffer = BytesIO()
    df.to_parquet(buffer, engine="pyarrow", index=False)
    buffer.seek(0)

    service = DataLakeServiceClient(account_url=account_url, credential=_credential_instance())
    file_client = service.get_file_client(file_system=filesystem, file_path=file_path)
    file_client.upload_data(buffer.getvalue(), overwrite=True)


def discover_table_mappings(source_base_uri: str, target_base_uri: str) -> list[TableMapping]:
    """Return one TableMapping per Delta table directory found directly under source_base_uri.

    Each target URI is built as target_base_uri/<table_name>, giving a strict
    1-to-1 source→target pairing without any manual configuration.
    """
    global DataLakeServiceClient
    if DataLakeServiceClient is None:
        from azure.storage.filedatalake import DataLakeServiceClient as _DataLakeServiceClient
        DataLakeServiceClient = _DataLakeServiceClient

    filesystem, host, base_path = _parse_abfss_uri(source_base_uri)
    base_path = base_path.rstrip("/")
    target_base = target_base_uri.rstrip("/")

    service_client = DataLakeServiceClient(
        account_url=f"https://{host}",
        credential=_credential_instance(),
    )
    fs_client = service_client.get_file_system_client(file_system=filesystem)

    mappings: list[TableMapping] = []
    for item in fs_client.get_paths(path=base_path, recursive=False):
        if not item.is_directory:
            continue
        delta_log_path = f"{item.name.rstrip('/')}/_delta_log"
        if not fs_client.get_directory_client(delta_log_path).exists():
            logger.debug("Skipping %s — no _delta_log found", item.name)
            continue
        table_name = item.name.rstrip("/").rsplit("/", 1)[-1]
        source_uri = f"abfss://{filesystem}@{host}/{item.name.rstrip('/')}"
        target_uri = f"{target_base}/{table_name}"
        mappings.append(TableMapping(source_uri=source_uri, target_uri=target_uri, table_name=table_name))

    logger.info("Discovered %d table(s) under %s", len(mappings), source_base_uri)
    return mappings


class PurviewClient:
    def __init__(self, account_name: str, token: str) -> None:
        self._base = f"https://{account_name}.purview.azure.com"
        self._headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = requests.get(self._base + path, headers=self._headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def column_classifications(self, qualified_name: str) -> dict[str, list[str]]:
        try:
            data = self._get(
                "/catalog/api/atlas/v2/entity/uniqueAttribute/type/azure_datalake_gen2_path",
                params={"attr:qualifiedName": qualified_name},
            )
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            logger.warning("Purview HTTP %s: %s", code, exc)
            return {}
        except Exception as exc:
            logger.warning("Purview request failed: %s", exc)
            return {}

        result: dict[str, list[str]] = {}
        for entity in data.get("referredEntities", {}).values():
            if entity.get("typeName") != "azure_datalake_gen2_column":
                continue
            col = entity.get("attributes", {}).get("name", "")
            labels = [c["typeName"] for c in entity.get("classifications", [])]
            if col and labels:
                result[col] = labels
        return result

    @staticmethod
    def qualified_name(abfss_uri: str) -> str:
        without_scheme = abfss_uri.replace("abfss://", "")
        container, rest = without_scheme.split("@", 1)
        host, path = rest.split("/", 1)
        return f"https://{host}/{container}/{path}"


def run_purview_check(source_uri: str, df_columns: list[str], purview_account: str | None) -> dict:
    empty = {"available": False, "flagged_columns": [], "column_labels": {}, "discrepancies": []}
    if not purview_account:
        return empty
    try:
        client = PurviewClient(purview_account, acquire_token(PURVIEW_TOKEN_SCOPE))
        col_labels = client.column_classifications(PurviewClient.qualified_name(source_uri))
        flagged = list(col_labels.keys())
        return {
            "available": True,
            "flagged_columns": flagged,
            "column_labels": col_labels,
            "discrepancies": [c for c in flagged if c not in df_columns],
        }
    except Exception as exc:
        logger.warning("Purview check failed (non-fatal): %s", exc)
        return empty


_DDL_RUNS = """
CREATE TABLE IF NOT EXISTS pii_pipeline_runs (
    run_id UUID PRIMARY KEY,
    pipeline_version TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    table_name TEXT,
    source_uri TEXT NOT NULL,
    target_uri TEXT NOT NULL,
    total_rows INTEGER,
    total_columns INTEGER,
    columns_scanned INTEGER,
    columns_hit JSONB,
    entities_total INTEGER,
    entity_counts JSONB,
    unique_entities JSONB,
    free_text_cols JSONB,
    k_anonymity_k INTEGER,
    quasi_columns JSONB,
    suppressed_rows INTEGER NOT NULL DEFAULT 0,
    residual_pii INTEGER NOT NULL DEFAULT 0,
    column_renames JSONB,
    hashed_columns JSONB,
    purview_ok BOOLEAN NOT NULL DEFAULT FALSE,
    purview_flagged JSONB,
    purview_diffs JSONB,
    status TEXT NOT NULL DEFAULT 'running',
    error_msg TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_DDL_COLUMN_EVENTS = """
CREATE TABLE IF NOT EXISTS pii_pipeline_column_events (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES pii_pipeline_runs(run_id),
    column_name TEXT NOT NULL,
    detections INTEGER NOT NULL DEFAULT 0,
    entity_counts JSONB,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_DDL_ALERTS = """
CREATE TABLE IF NOT EXISTS pii_pipeline_alerts (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID,
    table_name TEXT,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'error',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


class AuditDB:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._init_schema()

    @contextmanager
    def _cursor(self) -> Generator:
        conn = psycopg2.connect(self._dsn)
        try:
            with conn.cursor() as cur:
                yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._cursor() as cur:
            cur.execute(_DDL_RUNS)
            cur.execute(_DDL_COLUMN_EVENTS)
            cur.execute(_DDL_ALERTS)

    def open_run(self, run_id: str, started_at: datetime, mapping: TableMapping) -> None:
        sql = """
            INSERT INTO pii_pipeline_runs
                (run_id, pipeline_version, started_at, table_name, source_uri, target_uri, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        with self._cursor() as cur:
            cur.execute(sql, (run_id, PIPELINE_VERSION, started_at, mapping.table_name, mapping.source_uri, mapping.target_uri, "running"))

    def record_columns(self, run_id: str, column_stats: list) -> None:
        rows = [(run_id, s["column"], s["detections"], json.dumps(s["entity_counts"])) for s in column_stats]
        sql = """
            INSERT INTO pii_pipeline_column_events
                (run_id, column_name, detections, entity_counts)
            VALUES %s
        """
        with self._cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, rows)

    def record_alert(self, run_id: str | None, table_name: str | None, subject: str, body: str, severity: str = "error") -> None:
        sql = """
            INSERT INTO pii_pipeline_alerts (run_id, table_name, subject, body, severity)
            VALUES (%s, %s, %s, %s, %s)
        """
        with self._cursor() as cur:
            cur.execute(sql, (run_id, table_name, subject, body, severity))

    def close_run(self, run_id: str, audit: dict) -> None:
        sql = """
            UPDATE pii_pipeline_runs SET
                finished_at = %s,
                total_rows = %s,
                total_columns = %s,
                columns_scanned = %s,
                columns_hit = %s,
                entities_total = %s,
                entity_counts = %s,
                unique_entities = %s,
                free_text_cols = %s,
                k_anonymity_k = %s,
                quasi_columns = %s,
                suppressed_rows = %s,
                residual_pii = %s,
                column_renames = %s,
                hashed_columns = %s,
                purview_ok = %s,
                purview_flagged = %s,
                purview_diffs = %s,
                status = %s,
                error_msg = %s
            WHERE run_id = %s
        """
        with self._cursor() as cur:
            cur.execute(sql, (
                audit.get("pipeline_end_ts"),
                audit.get("total_rows_processed"),
                audit.get("total_columns_in_table"),
                audit.get("total_columns_scanned"),
                json.dumps(audit.get("columns_anonymized", [])),
                audit.get("total_entities_detected"),
                json.dumps(audit.get("entity_counts", {})),
                json.dumps(audit.get("unique_entities", {})),
                json.dumps(audit.get("free_text_columns", [])),
                audit.get("k_anonymity_k"),
                json.dumps(audit.get("quasi_columns", [])),
                audit.get("suppressed_rows", 0),
                audit.get("residual_pii_count", 0),
                json.dumps(audit.get("column_renames", {})),
                json.dumps(audit.get("hashed_columns", [])),
                audit.get("purview_available", False),
                json.dumps(audit.get("purview_flagged_columns", [])),
                json.dumps(audit.get("purview_discrepancies", [])),
                audit.get("status"),
                audit.get("error_message"),
                run_id,
            ))


def connect_audit_db(database_url: str | None) -> Optional[AuditDB]:
    if not database_url:
        logger.info("DATABASE_URL not set; audit DB disabled.")
        return None
    try:
        return AuditDB(database_url)
    except Exception as exc:
        logger.warning("Audit DB connection failed (non-fatal): %s", exc)
        return None
