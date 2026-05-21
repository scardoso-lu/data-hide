"""External persistence and Fabric repository adapters."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
import json
import logging
import re
import struct
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
SQL_TOKEN_SCOPE = "https://database.windows.net/.default"
FABRIC_TOKEN_SCOPE = "https://api.fabric.microsoft.com/.default"
PIPELINE_VERSION = "2.2.0"

logger = logging.getLogger(__name__)
_credential: Optional[object] = None
DefaultAzureCredential = None
DeltaTable = None
write_deltalake = None
DataLakeServiceClient = None
_pyodbc = None


@dataclass(frozen=True)
class TableMapping:
    source_uri: str
    target_uri: str
    table_name: str | None = None
    read_mode: str = "delta"  # "delta" | "sql"


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


def _account_name(uri: str) -> str:
    try:
        _, host, _ = _parse_abfss_uri(uri)
    except ValueError:
        raise ValueError(
            f"Cannot parse account name from URI: '{uri}'. "
            "Expected: abfss://container@account.dfs.fabric.microsoft.com/... "
            "or https://account.dfs.fabric.microsoft.com/container/..."
        ) from None
    return host.split(".", 1)[0]


def _parse_abfss_uri(uri: str) -> tuple[str, str, str]:
    abfss_match = re.match(r"^abfss://([^@/]+)@([^/]+)(?:/(.*))?$", uri)
    if abfss_match:
        return abfss_match.group(1), abfss_match.group(2), abfss_match.group(3) or ""

    https_match = re.match(r"^https://([^/]+)/([^/]+)/(.+)$", uri)
    if https_match:
        return https_match.group(2), https_match.group(1), https_match.group(3)

    if not abfss_match:
        raise ValueError(
            f"Cannot parse storage URI: '{uri}'. "
            "Expected: abfss://filesystem@host/path or https://host/filesystem/path"
        )


def _format_abfss_uri(filesystem: str, host: str, path: str) -> str:
    return f"abfss://{filesystem}@{host}/{path.strip('/')}"


def _looks_like_uuid(value: str) -> bool:
    return bool(re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        value,
    ))


def _fabric_item_display_name(workspace_id: str, item_id: str) -> str | None:
    token = acquire_token(FABRIC_TOKEN_SCOPE)
    response = requests.get(
        f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}/items/{item_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    response.raise_for_status()
    display_name = response.json().get("displayName")
    return display_name if isinstance(display_name, str) and display_name else None


def _resolve_onelake_item_id_path(filesystem: str, host: str, path: str) -> str:
    if host.lower() != "onelake.dfs.fabric.microsoft.com":
        return path

    parts = path.strip("/").split("/", 1)
    if not parts or not _looks_like_uuid(parts[0]):
        return path

    # OneLake requires workspace and artifact identifiers to use the same mode:
    # GUID+GUID or friendly-name+friendly-name.  Do not rewrite a GUID workspace
    # path to a friendly lakehouse name, because OneLake rejects that mix with
    # FriendlyNameSupportDisabled.
    if _looks_like_uuid(filesystem):
        return path

    item_id = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    try:
        display_name = _fabric_item_display_name(filesystem, item_id)
    except Exception as exc:
        logger.warning("Could not resolve Fabric item id '%s' in workspace '%s': %s", item_id, filesystem, exc)
        return path

    if not display_name:
        return path

    item_path = display_name if display_name.endswith(".Lakehouse") else f"{display_name}.Lakehouse"
    return f"{item_path}/{rest}" if rest else item_path


def _mapping_for_delta_path(filesystem: str, host: str, table_path: str, target_base: str) -> TableMapping:
    table_path = table_path.rstrip("/")
    table_name = table_path.rsplit("/", 1)[-1]
    return TableMapping(
        source_uri=_format_abfss_uri(filesystem, host, table_path),
        target_uri=f"{target_base}/{table_name}",
        table_name=table_name,
    )


def _discover_delta_mappings(fs_client, filesystem: str, host: str, base_path: str, target_base: str) -> list[TableMapping]:
    mappings: list[TableMapping] = []
    seen_paths: set[str] = set()

    immediate_items = list(fs_client.get_paths(path=base_path, recursive=False))
    logger.info("Listed %d immediate path(s) under %s", len(immediate_items), base_path)
    for item in immediate_items:
        if not item.is_directory:
            continue
        item_path = item.name.rstrip("/")
        delta_log_path = f"{item_path}/_delta_log"
        if not fs_client.get_directory_client(delta_log_path).exists():
            logger.debug("Skipping %s - no _delta_log found", item.name)
            continue
        seen_paths.add(item_path)
        mappings.append(_mapping_for_delta_path(filesystem, host, item_path, target_base))

    if mappings:
        return mappings

    logger.info("No immediate Delta tables found under %s; scanning recursively for _delta_log", base_path)
    for item in fs_client.get_paths(path=base_path, recursive=True):
        if not item.is_directory:
            continue
        item_path = item.name.rstrip("/")
        if not item_path.endswith("/_delta_log"):
            continue
        table_path = item_path[: -len("/_delta_log")]
        if table_path in seen_paths:
            continue
        seen_paths.add(table_path)
        mappings.append(_mapping_for_delta_path(filesystem, host, table_path, target_base))

    return mappings


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


def _sql_connection(sql_endpoint: str, database: str):
    global _pyodbc
    if _pyodbc is None:
        import pyodbc as _mod
        _pyodbc = _mod

    token = acquire_token(SQL_TOKEN_SCOPE)
    token_bytes = token.encode("utf-8")
    exptoken = b"".join(bytes([b, 0]) for b in token_bytes)
    token_struct = struct.pack("=i", len(exptoken)) + exptoken

    conn_str = (
        f"Driver={{ODBC Driver 18 for SQL Server}};"
        f"Server={sql_endpoint},1433;"
        f"Database={database};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
    )
    return _pyodbc.connect(conn_str, attrs_before={1256: token_struct})


def _discover_sql_table_names(sql_endpoint: str, database: str) -> list[str]:
    logger.info("Querying INFORMATION_SCHEMA for shortcut tables at '%s'", sql_endpoint)
    conn = _sql_connection(sql_endpoint, database)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_SCHEMA = 'dbo' AND TABLE_TYPE = 'BASE TABLE'"
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def read_sql_table(table_name: str, sql_endpoint: str, database: str) -> pd.DataFrame:
    logger.info("Reading SQL table '%s' from endpoint '%s'", table_name, sql_endpoint)
    conn = _sql_connection(sql_endpoint, database)
    try:
        return pd.read_sql_query(f"SELECT * FROM [dbo].[{table_name}]", conn)
    finally:
        conn.close()


def discover_table_mappings(source_base_uri: str, target_base_uri: str, *, sql_endpoint: str | None = None, sql_database: str | None = None) -> list[TableMapping]:
    """Return TableMappings for every Delta table under source_base_uri plus any
    SQL-only shortcuts discovered via the Fabric SQL Analytics Endpoint.

    Delta tables discovered via ADLS always take precedence; a table that appears
    in both ADLS and SQL is included exactly once as read_mode="delta".
    """
    global DataLakeServiceClient
    if DataLakeServiceClient is None:
        from azure.storage.filedatalake import DataLakeServiceClient as _DataLakeServiceClient
        DataLakeServiceClient = _DataLakeServiceClient

    filesystem, host, base_path = _parse_abfss_uri(source_base_uri)
    base_path = _resolve_onelake_item_id_path(filesystem, host, base_path)
    base_path = base_path.rstrip("/")
    target_filesystem, target_host, target_path = _parse_abfss_uri(target_base_uri)
    target_path = _resolve_onelake_item_id_path(target_filesystem, target_host, target_path)
    target_base = _format_abfss_uri(target_filesystem, target_host, target_path).rstrip("/")

    service_client = DataLakeServiceClient(
        account_url=f"https://{host}",
        credential=_credential_instance(),
    )
    fs_client = service_client.get_file_system_client(file_system=filesystem)

    mappings = _discover_delta_mappings(fs_client, filesystem, host, base_path, target_base)
    delta_table_names = {mapping.table_name for mapping in mappings}
    logger.info("Discovered %d Delta table(s) under %s", len(mappings), source_base_uri)

    if sql_endpoint and sql_database:
        try:
            sql_names = _discover_sql_table_names(sql_endpoint, sql_database)
            shortcuts = [n for n in sql_names if n not in delta_table_names]
            for name in shortcuts:
                source_uri = f"sql://{sql_endpoint}/{sql_database}/{name}"
                target_uri = f"{target_base}/{name}"
                mappings.append(TableMapping(
                    source_uri=source_uri,
                    target_uri=target_uri,
                    table_name=name,
                    read_mode="sql",
                ))
            logger.info(
                "Discovered %d SQL shortcut(s) via '%s' (%d already covered by Delta)",
                len(shortcuts), sql_endpoint, len(sql_names) - len(shortcuts),
            )
        except Exception as exc:
            logger.warning("SQL shortcut discovery failed (non-fatal): %s", exc)

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
    gps_columns JSONB,
    timestamp_cols_binned JSONB,
    numeric_cols_binned JSONB,
    hashed_columns JSONB,
    purview_ok BOOLEAN NOT NULL DEFAULT FALSE,
    purview_flagged JSONB,
    purview_diffs JSONB,
    output_type TEXT NOT NULL DEFAULT 'anonymized_rows',
    aggregate_cells INTEGER,
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
                gps_columns = %s,
                timestamp_cols_binned = %s,
                numeric_cols_binned = %s,
                hashed_columns = %s,
                purview_ok = %s,
                purview_flagged = %s,
                purview_diffs = %s,
                output_type = %s,
                aggregate_cells = %s,
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
                json.dumps(audit.get("gps_columns_anonymized", [])),
                json.dumps(audit.get("timestamp_columns_binned", [])),
                json.dumps(audit.get("numeric_columns_binned", [])),
                json.dumps(audit.get("hashed_columns", [])),
                audit.get("purview_available", False),
                json.dumps(audit.get("purview_flagged_columns", [])),
                json.dumps(audit.get("purview_discrepancies", [])),
                audit.get("output_type", "anonymized_rows"),
                audit.get("aggregate_cells"),
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
