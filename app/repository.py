"""External persistence and Fabric repository adapters."""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import logging
import os
import pathlib
import re
import struct
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Generator

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
PIPELINE_VERSION = "2.3.0"

logger = logging.getLogger(__name__)
_credential: object | None = None
_token_cache: dict[str, tuple[str, float]] = {}
_service_client_cache: dict[tuple[str, int], object] = {}
_fabric_item_name_cache: dict[tuple[str, str], str | None] = {}
DefaultAzureCredential = None
DataLakeServiceClient = None
DeltaTable = None
_pyodbc = None
_duckdb = None

TEMPORAL_NAME_TOKENS = {
    "date",
    "time",
    "timestamp",
    "datetime",
    "created",
    "updated",
    "recorded",
    "captured",
    "occurred",
    "ts",
    "dt",
}
DEFAULT_READ_LOOKBACK_DAYS = 365
DELTA_DISCOVERY_RECURSIVE_THRESHOLD = 20


@dataclass(frozen=True)
class TableMapping:
    source_uri: str
    target_uri: str
    table_name: str | None = None
    read_mode: str = "delta"  # "delta" | "sql"
    # SQL schema of the source for `read_mode="sql"` mappings; preserved
    # in the destination path so the closest source-to-target layout is
    # maintained across environments.  Ignored for `read_mode="delta"`.
    schema: str | None = None


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


def acquire_cached_token(scope: str) -> str:
    now = time.time()
    cached = _token_cache.get(scope)
    if cached and cached[1] - now > 300:
        return cached[0]

    token = _credential_instance().get_token(scope)
    expires_on = float(getattr(token, "expires_on", now + 3600))
    _token_cache[scope] = (token.token, expires_on)
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
    cache_key = (workspace_id, item_id)
    if cache_key in _fabric_item_name_cache:
        return _fabric_item_name_cache[cache_key]

    token = acquire_cached_token(FABRIC_TOKEN_SCOPE)
    response = requests.get(
        f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}/items/{item_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    response.raise_for_status()
    display_name = response.json().get("displayName")
    result = display_name if isinstance(display_name, str) and display_name else None
    _fabric_item_name_cache[cache_key] = result
    return result


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
    """Build the source/target mapping for a Delta table discovered under a source lakehouse.

    The schema-enabled Fabric lakehouse layout is
    ``…/<lakehouse>.Lakehouse/Tables/<schema>/<table>/_delta_log/`` — Fabric's
    UI only registers folders that appear under a schema directory.  We
    therefore preserve everything after ``/Tables/`` in the source path
    (including any schema folder) and append it to the target base.  For
    schema-less lakehouses the source has ``Tables/<table>/`` and the target
    becomes ``<target_base>/<table>/`` — same as before.
    """
    table_path = table_path.rstrip("/")
    table_name = table_path.rsplit("/", 1)[-1]
    # Everything after the LAST `/Tables/` is the part the destination must
    # mirror, so a `dbo/foo` source lands at `<target_base>/dbo/foo`.
    after_tables = table_path.rsplit("/Tables/", 1)
    relative = after_tables[1] if len(after_tables) == 2 else table_name
    return TableMapping(
        source_uri=_format_abfss_uri(filesystem, host, table_path),
        target_uri=f"{target_base}/{relative}",
        table_name=table_name,
    )


def _ensure_lakehouse_tables_target_base(path: str) -> None:
    normalized = path.strip("/")
    if "/Files/" in f"/{normalized}/":
        raise ValueError(
            "TARGET_BASE_ABFSS_URI points to Lakehouse Files. Delta table output must target "
            "the Lakehouse Tables root, for example "
            "abfss://<workspace>@onelake.dfs.fabric.microsoft.com/<lakehouse>.Lakehouse/Tables."
        )
    if not normalized.endswith("/Tables"):
        raise ValueError(
            "TARGET_BASE_ABFSS_URI must end with '<lakehouse>.Lakehouse/Tables' so Fabric "
            "registers each output as a Lakehouse table."
        )


def _ensure_lakehouse_delta_table_uri(path: str) -> None:
    normalized = path.strip("/")
    if "/Files/" in f"/{normalized}/":
        raise ValueError(
            "Delta table output cannot be written under Lakehouse Files. Set "
            "TARGET_BASE_ABFSS_URI to '<lakehouse>.Lakehouse/Tables'."
        )
    if "/Tables/" not in f"/{normalized}/":
        raise ValueError("Delta table output URI must point under '<lakehouse>.Lakehouse/Tables/<table>'.")


def _discover_delta_mappings(fs_client, filesystem: str, host: str, base_path: str, target_base: str) -> list[TableMapping]:
    mappings: list[TableMapping] = []
    seen_paths: set[str] = set()

    immediate_items = list(fs_client.get_paths(path=base_path, recursive=False))
    logger.info("Listed %d immediate path(s) under %s", len(immediate_items), base_path)
    immediate_dirs = [item for item in immediate_items if item.is_directory]
    if len(immediate_dirs) > DELTA_DISCOVERY_RECURSIVE_THRESHOLD:
        logger.info(
            "Immediate listing has %d directories; scanning recursively for _delta_log",
            len(immediate_dirs),
        )
        return _discover_delta_mappings_recursive(fs_client, filesystem, host, base_path, target_base, seen_paths)

    for item in immediate_dirs:
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
    return _discover_delta_mappings_recursive(fs_client, filesystem, host, base_path, target_base, seen_paths)


def _discover_delta_mappings_recursive(
    fs_client,
    filesystem: str,
    host: str,
    base_path: str,
    target_base: str,
    seen_paths: set[str],
) -> list[TableMapping]:
    mappings: list[TableMapping] = []
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


def _storage_opts(uri: str, token: str) -> dict:
    return {"account_name": _account_name(uri), "bearer_token": token}


def _fresh_opts(uri: str) -> dict:
    return _storage_opts(uri, acquire_cached_token(ONELAKE_TOKEN_SCOPE))


def _env_int_at_least(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be {minimum} or greater")
    return value


def _max_upload_workers() -> int:
    return _env_int_at_least("MAX_UPLOAD_WORKERS", 4, 1)


def _is_not_found_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 404:
        return True
    error_code = getattr(exc, "error_code", None)
    return str(error_code).lower() in {"pathnotfound", "resourcenotfound"}


def _delete_remote_directory_if_exists(fs_client, path: str) -> None:
    directory_client = fs_client.get_directory_client(path)
    try:
        if directory_client.exists():
            directory_client.delete_directory()
    except Exception as exc:
        if _is_not_found_error(exc):
            return
        raise


def _partition_delta_files(local_files: list[pathlib.Path]) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    data_files = [path for path in local_files if "_delta_log" not in path.parts]
    log_files = [path for path in local_files if "_delta_log" in path.parts]
    return data_files, log_files


def _remote_delta_file_path(base_remote_path: str, local_table_path: pathlib.Path, local_file: pathlib.Path) -> str:
    return f"{base_remote_path}/{local_file.relative_to(local_table_path).as_posix()}"


def _upload_delta_file_group(
    fs_client,
    local_table_path: pathlib.Path,
    base_remote_path: str,
    files: list[pathlib.Path],
    workers: int,
) -> None:
    if not files:
        return

    def upload_file(local_file: pathlib.Path) -> None:
        remote_path = _remote_delta_file_path(base_remote_path, local_table_path, local_file)
        file_client = fs_client.get_file_client(file_path=remote_path)
        with open(local_file, "rb") as fh:
            file_client.upload_data(fh, overwrite=True)

    with ThreadPoolExecutor(max_workers=min(workers, len(files))) as executor:
        list(executor.map(upload_file, files))


def _read_lookback_days() -> int:
    return _env_int_at_least("READ_LOOKBACK_DAYS", DEFAULT_READ_LOOKBACK_DAYS, 0)


def read_cutoff_ts(now: datetime | None = None) -> datetime:
    """Return the UTC lower bound for source reads."""
    base = now or datetime.now(timezone.utc)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    return base.astimezone(timezone.utc) - timedelta(days=_read_lookback_days())


def _identifier_tokens(name: str) -> set[str]:
    return {part.lower() for part in re.split(r"[^A-Za-z0-9]+|(?<=[a-z])(?=[A-Z])", name) if part}


def _looks_temporal_by_name(name: str) -> bool:
    return bool(_identifier_tokens(name) & TEMPORAL_NAME_TOKENS)


def _delta_temporal_columns(schema) -> list[str]:
    import pyarrow.types as pa_types

    columns: list[str] = []
    for field in schema:
        if pa_types.is_date(field.type) or pa_types.is_timestamp(field.type):
            columns.append(field.name)
        elif _looks_temporal_by_name(field.name) and (
            pa_types.is_string(field.type) or pa_types.is_large_string(field.type)
        ):
            columns.append(field.name)
    return columns


def _quote_duckdb_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _duckdb_temporal_filter_sql(columns: list[str]) -> str:
    predicates = [
        f"TRY_CAST({_quote_duckdb_ident(col)} AS TIMESTAMP) >= ?"
        for col in columns
    ]
    return 'SELECT * FROM "_source_rows" WHERE ' + " OR ".join(predicates)


def read_delta(uri: str, storage_options: dict) -> pd.DataFrame:
    global DeltaTable
    if DeltaTable is None:
        from deltalake import DeltaTable as _DeltaTable
        DeltaTable = _DeltaTable

    cutoff = read_cutoff_ts()
    logger.info("Reading Delta table uri='%s' with cutoff >= %s", uri, cutoff.isoformat())
    table = DeltaTable(uri, storage_options=storage_options)
    dataset = table.to_pyarrow_dataset()
    temporal_columns = _delta_temporal_columns(dataset.schema)
    if not temporal_columns:
        logger.warning("No temporal columns found in Delta table '%s'; reading all rows", uri)

    global _duckdb
    if _duckdb is None:
        import duckdb as _ddb
        _duckdb = _ddb

    conn = _duckdb.connect()
    try:
        conn.register("_source_rows", dataset)
        if not temporal_columns:
            return conn.execute('SELECT * FROM "_source_rows"').df()

        return conn.execute(_duckdb_temporal_filter_sql(temporal_columns), [cutoff] * len(temporal_columns)).df()
    finally:
        conn.close()


def _quote_tsql_ident(name: str) -> str:
    return "[" + name.replace("]", "]]") + "]"


def _tsql_temporal_filter_sql(table_ref: str, columns: list[str]) -> str:
    predicates = [
        f"TRY_CONVERT(datetime2, {_quote_tsql_ident(col)}) >= ?"
        for col in columns
    ]
    return f"SELECT * FROM {table_ref} WHERE " + " OR ".join(predicates)


def _sql_temporal_columns(cursor, table_name: str, schema: str = "dbo") -> list[str]:
    cursor.execute(
        "SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? "
        "ORDER BY ORDINAL_POSITION",
        schema,
        table_name,
    )
    temporal_types = {"date", "datetime", "datetime2", "datetimeoffset", "smalldatetime"}
    string_types = {"char", "nchar", "varchar", "nvarchar", "text", "ntext"}
    columns: list[str] = []
    for name, data_type in cursor.fetchall():
        data_type = str(data_type).lower()
        if data_type in temporal_types:
            columns.append(name)
        elif data_type in string_types and _looks_temporal_by_name(name):
            columns.append(name)
    return columns


def write_delta(df: pd.DataFrame, uri: str, storage_options: dict) -> None:
    """Write df as a Delta Lake table to OneLake/ADLS.

    Builds the table locally with delta-rs then uploads the
    full directory tree — data files and _delta_log — so Fabric recognises
    the output as a proper Delta table without additional conversion steps.
    """
    if len(df.columns) == 0:
        raise ValueError("Delta output requires at least one column")

    global DataLakeServiceClient
    if DataLakeServiceClient is None:
        from azure.storage.filedatalake import DataLakeServiceClient as _DataLakeServiceClient
        DataLakeServiceClient = _DataLakeServiceClient
    from deltalake import write_deltalake

    filesystem, host, uri_path = _parse_abfss_uri(uri)
    _ensure_lakehouse_delta_table_uri(uri_path)
    table_name = uri_path.strip("/").rsplit("/", 1)[-1]
    account_url = f"https://{host}"
    base_remote_path = uri_path.strip("/")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = pathlib.Path(tmpdir) / table_name

        write_deltalake(str(local_path), df, mode="overwrite")

        local_files = sorted(f for f in local_path.rglob("*") if f.is_file())
        logger.info(
            "Uploading Delta table '%s' → abfss://%s@%s/%s (%d file(s))",
            table_name, filesystem, host, base_remote_path, len(local_files),
        )
        service = _data_lake_service_client(account_url)
        fs_client = service.get_file_system_client(file_system=filesystem)

        data_files, log_files = _partition_delta_files(local_files)

        logger.info("Replacing existing Delta table directory at %s", base_remote_path)
        _delete_remote_directory_if_exists(fs_client, base_remote_path)

        workers = min(_max_upload_workers(), max(1, len(local_files)))
        _upload_delta_file_group(fs_client, local_path, base_remote_path, data_files, workers)
        _upload_delta_file_group(fs_client, local_path, base_remote_path, log_files, workers)


def _data_lake_service_client(account_url: str):
    cache_key = (account_url, id(DataLakeServiceClient))
    cached = _service_client_cache.get(cache_key)
    if cached is not None:
        return cached
    service = DataLakeServiceClient(account_url=account_url, credential=_credential_instance())
    _service_client_cache[cache_key] = service
    return service


def _sql_connection(sql_endpoint: str, database: str):
    global _pyodbc
    if _pyodbc is None:
        import pyodbc as _mod
        _pyodbc = _mod

    token = acquire_cached_token(SQL_TOKEN_SCOPE)
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


def _discover_sql_table_names(sql_endpoint: str, database: str) -> list[tuple[str, str]]:
    """Return ``(schema, table_name)`` pairs for every base table the SQL
    endpoint exposes, across all schemas.

    Returning the full ``(schema, table)`` pair (rather than just the name)
    lets the caller mirror the source schema layout into the destination
    lakehouse — required for schema-enabled Fabric lakehouses, which only
    register Delta directories under ``Tables/<schema>/<table>/``.
    """
    logger.info("Querying INFORMATION_SCHEMA for shortcut tables at '%s'", sql_endpoint)
    conn = _sql_connection(sql_endpoint, database)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_TYPE = 'BASE TABLE'"
        )
        return [(row[0], row[1]) for row in cursor.fetchall()]
    finally:
        conn.close()


def read_sql_table(table_name: str, sql_endpoint: str, database: str, schema: str = "dbo") -> pd.DataFrame:
    cutoff = read_cutoff_ts()
    sql_cutoff = cutoff.replace(tzinfo=None)
    logger.info(
        "Reading SQL table '%s.%s' from endpoint '%s' with cutoff >= %s",
        schema, table_name, sql_endpoint, cutoff.isoformat(),
    )
    conn = _sql_connection(sql_endpoint, database)
    try:
        cursor = conn.cursor()
        temporal_columns = _sql_temporal_columns(cursor, table_name, schema=schema)
        table_ref = f"{_quote_tsql_ident(schema)}.{_quote_tsql_ident(table_name)}"
        if temporal_columns:
            cursor.execute(_tsql_temporal_filter_sql(table_ref, temporal_columns), *([sql_cutoff] * len(temporal_columns)))
        else:
            logger.warning("No temporal columns found in SQL table '%s.%s'; reading all rows", schema, table_name)
            cursor.execute(f"SELECT * FROM {table_ref}")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame.from_records(rows, columns=columns)
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
    _ensure_lakehouse_tables_target_base(target_path)
    target_base = _format_abfss_uri(target_filesystem, target_host, target_path).rstrip("/")

    service_client = _data_lake_service_client(f"https://{host}")
    fs_client = service_client.get_file_system_client(file_system=filesystem)

    mappings = _discover_delta_mappings(fs_client, filesystem, host, base_path, target_base)
    delta_table_names = {mapping.table_name for mapping in mappings}
    logger.info("Discovered %d Delta table(s) under %s", len(mappings), source_base_uri)

    if sql_endpoint and sql_database:
        try:
            sql_specs = _discover_sql_table_names(sql_endpoint, sql_database)
            # Compute the SQL-endpoint identity each Delta source already
            # covers.  Fabric's SQL endpoint exposes a Delta table at
            # `Tables/<schema>/<name>` as `<schema>.<name>`, and a
            # schema-less Delta at `Tables/<name>` as `dbo.<name>` (the
            # default schema).  Dedup against both forms so a SQL spec
            # pointing at the same underlying Delta directory is skipped.
            delta_sql_keys: set[str] = set()
            for mapping in mappings:
                relative = mapping.target_uri[len(target_base):].lstrip("/")
                if "/" in relative:
                    delta_sql_keys.add(relative)
                else:
                    delta_sql_keys.add(f"dbo/{relative}")
            shortcuts = [
                (schema, name) for schema, name in sql_specs
                if f"{schema}/{name}" not in delta_sql_keys
            ]
            for schema, name in shortcuts:
                # Source URI carries the schema so `read_sql_table` can
                # later resolve the fully-qualified ``[schema].[table]``.
                source_uri = f"sql://{sql_endpoint}/{sql_database}/{schema}/{name}"
                target_uri = f"{target_base}/{schema}/{name}"
                mappings.append(TableMapping(
                    source_uri=source_uri,
                    target_uri=target_uri,
                    table_name=name,
                    read_mode="sql",
                    schema=schema,
                ))
            logger.info(
                "Discovered %d SQL shortcut(s) via '%s' (%d already covered by Delta)",
                len(shortcuts), sql_endpoint, len(sql_specs) - len(shortcuts),
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
        client = PurviewClient(purview_account, acquire_cached_token(PURVIEW_TOKEN_SCOPE))
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
    key_vault_key_version TEXT,
    stage_seconds JSONB,
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

_DDL_RUNS_MIGRATIONS = [
    "ALTER TABLE pii_pipeline_runs ADD COLUMN IF NOT EXISTS key_vault_key_version TEXT",
    "ALTER TABLE pii_pipeline_runs ADD COLUMN IF NOT EXISTS stage_seconds JSONB",
]

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
            for migration in _DDL_RUNS_MIGRATIONS:
                cur.execute(migration)

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
                key_vault_key_version = %s,
                stage_seconds = %s,
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
                audit.get("key_vault_key_version"),
                json.dumps(audit.get("stage_seconds", {})),
                audit.get("purview_available", False),
                json.dumps(audit.get("purview_flagged_columns", [])),
                json.dumps(audit.get("purview_discrepancies", [])),
                audit.get("output_type", "anonymized_rows"),
                audit.get("aggregate_cells"),
                audit.get("status"),
                audit.get("error_message"),
                run_id,
            ))


def connect_audit_db(database_url: str | None) -> AuditDB | None:
    if not database_url:
        logger.info("DATABASE_URL not set; audit DB disabled.")
        return None
    try:
        return AuditDB(database_url)
    except Exception as exc:
        logger.warning("Audit DB connection failed (non-fatal): %s", exc)
        return None
