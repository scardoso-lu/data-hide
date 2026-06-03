"""Delta Lake read/write and ADLS upload helpers.

Mutable singletons (``DeltaTable``, ``DataLakeServiceClient``, ``_duckdb``,
``_service_client_cache``) live in the *package* namespace so tests can
patch them via ``app.repository.<name>``.  Every function that needs them
fetches them at call time through ``import app.infrastructure.repository as _r``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import pathlib
import tempfile

import polars as pl

from ._types import TableMapping
from ._utils import (
    _parse_abfss_uri,
    _format_abfss_uri,
    _ensure_lakehouse_delta_table_uri,
    _looks_temporal_by_name,
    _max_upload_workers,
    _read_lookback_days,
    read_cutoff_ts,
    DELTA_DISCOVERY_RECURSIVE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Memory / size diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def _process_rss_mb() -> float | None:
    """Return *current* process RSS in MB, or None if unavailable.

    Reads ``/proc/self/statm`` (field 2 = resident pages) on Linux — this is
    live RSS that rises AND falls as memory is freed.  ``resource.getrusage``
    is deliberately NOT used: its ``ru_maxrss`` is the peak high-water mark,
    which is monotonic and would make every post-peak stage look identical
    (and "never released") even when current usage has dropped.
    """
    # Linux: live RSS from /proc.
    try:
        with open("/proc/self/statm", "r") as fh:
            resident_pages = int(fh.readline().split()[1])
        import os
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1_048_576
    except Exception:
        pass
    # Non-Linux fallback (peak only — best effort, e.g. local dev on Windows).
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return None


def _log_pre_read_diagnostics(table, uri: str) -> None:
    """Log Delta table metadata (no row data loaded) and current process RSS.

    Called immediately after ``DeltaTable(uri)`` — before ``to_pyarrow_dataset()``
    — so the numbers appear in logs even when the subsequent data load causes an
    OOM kill.  All three pieces of information (file count, compressed size, RSS)
    are available from Delta transaction-log metadata alone; no Parquet files are
    opened.
    """
    table_name = uri.rsplit("/", 1)[-1]

    n_files: int | str = "?"
    try:
        n_files = len(table.files())
    except Exception:
        pass

    size_str = "?"
    try:
        import pyarrow.compute as _pc
        actions = table.get_add_actions(flatten=True)
        total_bytes = _pc.sum(actions.column("size_bytes")).as_py()
        if total_bytes and total_bytes >= 1_073_741_824:
            size_str = f"{total_bytes / 1_073_741_824:.2f} GB"
        elif total_bytes:
            size_str = f"{total_bytes / 1_048_576:.1f} MB"
    except Exception:
        pass

    n_cols: int | str = "?"
    try:
        schema = table.schema()
        n_cols = len(schema.fields)
    except Exception:
        pass

    rss = _process_rss_mb()
    rss_str = f"{rss:.0f} MB" if rss is not None else "?"

    logger.info(
        "Delta pre-read: table='%s' parquet_files=%s columns=%s "
        "compressed_size=%s process_rss_before=%s",
        table_name, n_files, n_cols, size_str, rss_str,
    )


def _log_post_read_diagnostics(df: pl.DataFrame, uri: str) -> None:
    """Log DataFrame in-memory size and process RSS after the load completes."""
    table_name = uri.rsplit("/", 1)[-1]

    mem_str = "?"
    try:
        mem_str = f"{df.estimated_size('mb'):.0f} MB"
    except Exception:
        pass

    rss = _process_rss_mb()
    rss_str = f"{rss:.0f} MB" if rss is not None else "?"

    logger.info(
        "Delta post-read: table='%s' rows=%d cols=%d "
        "dataframe_ram=%s process_rss_after=%s",
        table_name, len(df), len(df.columns), mem_str, rss_str,
    )


def _storage_opts(uri: str, token: str) -> dict:
    from ._utils import _account_name
    return {"account_name": _account_name(uri), "bearer_token": token}


def _fresh_opts(uri: str) -> dict:
    import app.infrastructure.repository as _r
    return _storage_opts(uri, _r.acquire_cached_token(_r.ONELAKE_TOKEN_SCOPE))


def _delta_temporal_columns(schema) -> list[str]:
    typed, string = _split_temporal_columns(schema)
    return [*typed, *string]


def _split_temporal_columns(schema) -> tuple[list[str], list[str]]:
    """Split temporal columns into ``(typed, string)``.

    ``typed`` = real date/timestamp columns. These can be compared directly
    (``col >= cutoff``) so DuckDB pushes the predicate into the Parquet scan and
    prunes whole row groups by min/max statistics — the data older than the
    cutoff is never read into memory.

    ``string`` = columns matched only by name whose values are strings. They need
    ``TRY_CAST(col AS TIMESTAMP)``, which wraps the column in a function and so
    cannot be pruned; they are read in full.
    """
    import pyarrow.types as pa_types

    typed: list[str] = []
    string: list[str] = []
    for field in schema:
        if pa_types.is_date(field.type) or pa_types.is_timestamp(field.type):
            typed.append(field.name)
        elif _looks_temporal_by_name(field.name) and (
            pa_types.is_string(field.type) or pa_types.is_large_string(field.type)
        ):
            string.append(field.name)
    return typed, string


def _quote_duckdb_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _duckdb_temporal_filter_sql(typed_cols: list[str], string_cols: list[str]) -> str:
    # Typed date/timestamp columns are compared directly so DuckDB can prune
    # Parquet row groups via statistics (no TRY_CAST wrapper). String columns
    # still need the cast. Param order matches: typed first, then string.
    predicates = [f"{_quote_duckdb_ident(col)} >= ?" for col in typed_cols]
    predicates += [
        f"TRY_CAST({_quote_duckdb_ident(col)} AS TIMESTAMP) >= ?" for col in string_cols
    ]
    return 'SELECT * FROM "_source_rows" WHERE ' + " OR ".join(predicates)


def read_delta(uri: str, storage_options: dict) -> pl.DataFrame:
    import app.infrastructure.repository as _r
    if _r.DeltaTable is None:
        from deltalake import DeltaTable as _DT
        _r.DeltaTable = _DT

    cutoff = read_cutoff_ts()
    lookback_days = _read_lookback_days()
    logger.info(
        "Reading Delta table uri='%s' with cutoff >= %s (lookback_days=%d)",
        uri, cutoff.isoformat(), lookback_days,
    )
    table = _r.DeltaTable(uri, storage_options=storage_options)
    _log_pre_read_diagnostics(table, uri)

    dataset = table.to_pyarrow_dataset()
    typed_cols, string_cols = _split_temporal_columns(dataset.schema)
    temporal_columns = [*typed_cols, *string_cols]
    if not temporal_columns:
        logger.warning("No temporal columns found in Delta table '%s'; reading all rows", uri)

    if _r._duckdb is None:
        import duckdb as _ddb
        _r._duckdb = _ddb

    conn = _r._duckdb.connect()
    try:
        conn.register("_source_rows", dataset)
        # DuckDB → Polars: Arrow-backed strings use 3-5× less RAM than
        # pandas object dtype for string-heavy PII tables.
        if not temporal_columns:
            df = conn.execute('SELECT * FROM "_source_rows"').pl()
        else:
            # Cutoff filter. Predicates on typed date/timestamp columns push
            # into the Parquet scan, so DuckDB skips row groups whose max is
            # below the cutoff instead of reading the whole table into memory.
            df = conn.execute(
                _duckdb_temporal_filter_sql(typed_cols, string_cols),
                [cutoff] * len(temporal_columns),
            ).pl()
            if len(df) == 0:
                logger.warning(
                    "%d-day filter returned 0 rows for '%s'; "
                    "table is too old or READ_LOOKBACK_DAYS is too short — reading all rows",
                    lookback_days, uri,
                )
                df = conn.execute('SELECT * FROM "_source_rows"').pl()
        _log_post_read_diagnostics(df, uri)
        return df
    finally:
        conn.close()


def read_delta_sample(uri: str, storage_options: dict, n: int = 500) -> pl.DataFrame:
    """Read at most ``n`` rows of a Delta table — for column classification.

    The language-major classification passes only need column names plus a
    small value sample, so this avoids materialising the full table during
    Phase 1.  DuckDB's LIMIT is pushed into the Parquet scan; only the row
    groups needed for ``n`` rows are read.
    """
    import app.infrastructure.repository as _r
    if _r.DeltaTable is None:
        from deltalake import DeltaTable as _DT
        _r.DeltaTable = _DT
    if _r._duckdb is None:
        import duckdb as _ddb
        _r._duckdb = _ddb

    table = _r.DeltaTable(uri, storage_options=storage_options)
    dataset = table.to_pyarrow_dataset()
    conn = _r._duckdb.connect()
    try:
        conn.register("_source_rows", dataset)
        return conn.execute(f'SELECT * FROM "_source_rows" LIMIT {int(n)}').pl()
    finally:
        conn.close()


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


def _partition_delta_files(
    local_files: list[pathlib.Path],
) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    data_files = [p for p in local_files if "_delta_log" not in p.parts]
    log_files = [p for p in local_files if "_delta_log" in p.parts]
    return data_files, log_files


def _remote_delta_file_path(
    base_remote_path: str,
    local_table_path: pathlib.Path,
    local_file: pathlib.Path,
) -> str:
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


def _coerce_null_columns_arrow(table, null_col_names: list[str]):
    """Cast null-typed columns in an existing PyArrow Table to ``pa.string()``.

    When every value in a column is null, the Arrow schema carries a
    ``pa.null()`` type which delta-rs rejects with ``SchemaMismatchError:
    Invalid data type for Delta Lake: Null``.  Casting to ``pa.string()``
    preserves the nulls while giving delta-rs a concrete, writable type.
    """
    import pyarrow as pa
    logger.debug(
        "Coercing %d all-null column(s) to pa.string() for Delta write: %s",
        len(null_col_names), null_col_names,
    )
    column_order = [table.schema.field(i).name for i in range(table.num_columns)]
    columns: dict = {}
    for name in column_order:
        if name in null_col_names:
            columns[name] = pa.array([None] * len(table), type=pa.string())
        else:
            columns[name] = table.column(name)
    return pa.table(columns)


def write_delta(df: pl.DataFrame, uri: str, storage_options: dict) -> None:
    """Write a Polars DataFrame as a Delta Lake table to OneLake/ADLS.

    Polars → Arrow conversion is zero-copy for most types.  Builds the table
    locally with delta-rs then uploads the full directory tree - data files
    and _delta_log - so Fabric recognises the output as a proper Delta table
    without additional conversion steps.
    """
    if len(df.columns) == 0:
        raise ValueError("Delta output requires at least one column")

    import app.infrastructure.repository as _r
    if _r.DataLakeServiceClient is None:
        from azure.storage.filedatalake import DataLakeServiceClient as _DLSC
        _r.DataLakeServiceClient = _DLSC
    from deltalake import write_deltalake

    filesystem, host, uri_path = _parse_abfss_uri(uri)
    _ensure_lakehouse_delta_table_uri(uri_path)
    table_name = uri_path.strip("/").rsplit("/", 1)[-1]
    account_url = f"https://{host}"
    base_remote_path = uri_path.strip("/")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = pathlib.Path(tmpdir) / table_name

        import pyarrow as pa
        # Object columns (heterogeneous Python values) cannot convert to
        # Arrow directly — re-infer a concrete dtype from the values.  A
        # truly mixed column raises here, matching the historical
        # pa.Table.from_pandas(ArrowInvalid) failure mode.
        for _col, _dtype in df.schema.items():
            if _dtype == pl.Object:
                df = df.with_columns(pl.Series(_col, df[_col].to_list()))
        arrow_table = df.to_arrow()
        null_cols = [f.name for f in arrow_table.schema if pa.types.is_null(f.type)]
        if null_cols:
            arrow_table = _coerce_null_columns_arrow(arrow_table, null_cols)
        write_deltalake(str(local_path), arrow_table, mode="overwrite")

        local_files = sorted(f for f in local_path.rglob("*") if f.is_file())
        logger.info(
            "Uploading Delta table '%s' â†' abfss://%s@%s/%s (%d file(s))",
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
    import app.infrastructure.repository as _r
    cache_key = (account_url, id(_r.DataLakeServiceClient))
    cached = _r._service_client_cache.get(cache_key)
    if cached is not None:
        return cached
    service = _r.DataLakeServiceClient(
        account_url=account_url,
        credential=_r._credential_instance(),
    )
    _r._service_client_cache[cache_key] = service
    return service


def _mapping_for_delta_path(
    filesystem: str,
    host: str,
    table_path: str,
    target_base: str,
) -> TableMapping:
    """Build the source/target mapping for a Delta table discovered under a source lakehouse.

    The schema-enabled Fabric lakehouse layout is
    ``â€¦/<lakehouse>.Lakehouse/Tables/<schema>/<table>/_delta_log/`` â€" Fabric's
    UI only registers folders that appear under a schema directory.  We
    therefore preserve everything after ``/Tables/`` in the source path
    (including any schema folder) and append it to the target base.
    """
    table_path = table_path.rstrip("/")
    table_name = table_path.rsplit("/", 1)[-1]
    after_tables = table_path.rsplit("/Tables/", 1)
    relative = after_tables[1] if len(after_tables) == 2 else table_name
    return TableMapping(
        source_uri=_format_abfss_uri(filesystem, host, table_path),
        target_uri=f"{target_base}/{relative}",
        table_name=table_name,
    )


def _discover_delta_mappings(
    fs_client,
    filesystem: str,
    host: str,
    base_path: str,
    target_base: str,
) -> list[TableMapping]:
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
        return _discover_delta_mappings_recursive(
            fs_client, filesystem, host, base_path, target_base, seen_paths,
        )

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

    logger.info(
        "No immediate Delta tables found under %s; scanning recursively for _delta_log",
        base_path,
    )
    return _discover_delta_mappings_recursive(
        fs_client, filesystem, host, base_path, target_base, seen_paths,
    )


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
