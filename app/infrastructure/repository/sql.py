"""SQL Analytics Endpoint read adapter (Fabric T-SQL shortcuts via pyodbc).

``_pyodbc`` lives in the *package* namespace so tests can patch it.
Token acquisition also goes through the package so ``app.repository.acquire_cached_token``
patches are effective here.
"""

from __future__ import annotations

import logging
import struct

import pandas as pd

from ._utils import _looks_temporal_by_name, read_cutoff_ts

logger = logging.getLogger(__name__)


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


def _sql_connection(sql_endpoint: str, database: str):
    import app.infrastructure.repository as _r
    if _r._pyodbc is None:
        import pyodbc as _mod
        _r._pyodbc = _mod

    token = _r.acquire_cached_token(_r.SQL_TOKEN_SCOPE)
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
    return _r._pyodbc.connect(conn_str, attrs_before={1256: token_struct})


def _discover_sql_table_names(sql_endpoint: str, database: str) -> list[tuple[str, str]]:
    """Return ``(schema, table_name)`` pairs for every base table the SQL
    endpoint exposes, across all schemas."""
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


def read_sql_table(
    table_name: str,
    sql_endpoint: str,
    database: str,
    schema: str = "dbo",
) -> pd.DataFrame:
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
            cursor.execute(
                _tsql_temporal_filter_sql(table_ref, temporal_columns),
                *([sql_cutoff] * len(temporal_columns)),
            )
            rows = cursor.fetchall()
            if not rows:
                logger.warning(
                    "365-day filter returned 0 rows for '%s.%s'; table is too small or old â€” reading all rows",
                    schema, table_name,
                )
                cursor.execute(f"SELECT * FROM {table_ref}")
                rows = cursor.fetchall()
        else:
            logger.warning(
                "No temporal columns found in SQL table '%s.%s'; reading all rows",
                schema, table_name,
            )
            cursor.execute(f"SELECT * FROM {table_ref}")
            rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame.from_records(rows, columns=columns)
    finally:
        conn.close()
