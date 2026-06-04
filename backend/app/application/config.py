"""Pipeline configuration — application layer.

Holds ``PipelineConfig`` and the helper functions that parse environment
variables and database overrides into a typed, frozen configuration object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os

from ..infrastructure.repository import AuditDB, TableMapping

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
    # Tables on which k-anonymity is enabled.
    # Lowercase names parsed from K_ANONYMITY_TABLES=table1,table2.
    # K-anonymity is skipped for every table NOT in this set.
    k_anonymity_tables: frozenset[str] = field(default_factory=frozenset)
    # Per-table quasi-identifier columns.
    # Keys are lowercase table names; values are the column tuples.
    # Configured via QUASI_IDENTIFIER_COLS__<table_name>=col1,col2.
    quasi_identifier_cols_by_table: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Per-table columns excluded from anonymization entirely.
    # Loaded from pii_column_exclusions in PostgreSQL.
    excluded_columns_by_table: dict[str, frozenset[str]] = field(default_factory=dict)
    identifier_cols: tuple[str, ...] = ()
    source_base_uri: str | None = None
    target_base_uri: str | None = None
    sql_endpoint: str | None = None
    sql_database: str | None = None
    gps_precision: int = 1
    max_table_workers: int = 1
    # Explicit per-run table pairs loaded from pii_table_targets in PostgreSQL.
    # When non-empty, auto-discovery under source_base_uri is skipped entirely.
    table_targets: tuple[TableMapping, ...] = ()
    # Per-table columns that receive the targeted row-by-row Presidio scan.
    # Loaded from the apply_row_scan table. Keys are lowercase table names;
    # values are the set of column names to scan cell-by-cell.
    row_scan_columns_by_table: dict[str, frozenset[str]] = field(default_factory=dict)
    # Custom Purview classification type that means "redact this column".
    # Required whenever PURVIEW_ACCOUNT_NAME is set.  DB-overridable.
    purview_must_anonymize_type: str | None = None
    # Optional custom Purview classification type that means "verified non-PII,
    # skip all analysis".  Columns bearing this type bypass both column-policy
    # masking and row-level Presidio, exactly like pii_column_exclusions rows.
    purview_not_pii_type: str | None = None

    @classmethod
    def from_env(
        cls,
        config_overrides: "dict[str, str] | None" = None,
        excluded_columns: "dict[str, frozenset[str]] | None" = None,
        table_targets: "tuple[TableMapping, ...] | None" = None,
        row_scan_columns: "dict[str, frozenset[str]] | None" = None,
    ) -> "PipelineConfig":
        """Build from environment variables, optionally overlaid with DB overrides.

        ``config_overrides`` is a ``{key: value}`` dict loaded from
        ``pii_pipeline_config`` — values here win over the corresponding env
        vars for every runtime-tunable parameter.

        Secrets and connectivity settings are **always** read from the
        environment and cannot be overridden via the database:
        ``DATABASE_URL`` (or ``DB_HOST`` / ``DB_PORT`` / ``DB_USER`` /
        ``DB_PASSWORD`` / ``DB_NAME``), ``AZURE_*``, ``KEY_VAULT_URL``,
        ``KEY_VAULT_RSA_KEY_NAME``, ``HASH_SALT``,
        ``SOURCE_BASE_ABFSS_URI``, ``TARGET_BASE_ABFSS_URI``.
        """
        overrides = config_overrides or {}

        def _get(key: str, default: str = "") -> str:
            # Precedence: DB override wins over the environment variable, which
            # in turn wins over the hard-coded default. Reading os.environ here
            # is load-bearing — runtime-tunable config is supplied via env vars
            # (.env / docker-compose) and must take effect when no DB row exists.
            return overrides.get(key, os.environ.get(key, default))

        def _int_at_least(key: str, default: int, minimum: int) -> int:
            raw = _get(key, str(default))
            try:
                value = int(raw)
            except ValueError:
                raise ValueError(f"{key} must be an integer, got {raw!r}") from None
            if value < minimum:
                raise ValueError(f"{key} must be {minimum} or greater")
            return value

        purview_account_name = _get("PURVIEW_ACCOUNT_NAME") or None
        purview_must_anonymize_type = _get("PURVIEW_MUST_ANONYMIZE_TYPE") or None
        purview_not_pii_type = _get("PURVIEW_NOT_PII_TYPE") or None
        if purview_account_name:
            # Credential vars are secrets — read directly from os.environ,
            # not _get(), so they can never be stored in the DB config table.
            missing_creds = [
                v for v in ("PURVIEW_CLIENT_ID", "PURVIEW_CLIENT_SECRET")
                if not os.environ.get(v)
            ]
            if missing_creds:
                raise ValueError(
                    f"Purview credentials not set: {', '.join(missing_creds)} "
                    "must be configured when PURVIEW_ACCOUNT_NAME is set"
                )
            if not purview_must_anonymize_type:
                raise ValueError(
                    "PURVIEW_MUST_ANONYMIZE_TYPE must be set when PURVIEW_ACCOUNT_NAME is configured"
                )

        return cls(
            # ── secrets / connectivity: always from env, never from DB ────────
            database_url=os.environ.get("DATABASE_URL"),
            key_vault_url=os.environ.get("KEY_VAULT_URL"),
            key_vault_rsa_key_name=os.environ.get("KEY_VAULT_RSA_KEY_NAME"),
            hash_salt=os.environ.get("HASH_SALT"),
            source_base_uri=os.environ.get("SOURCE_BASE_ABFSS_URI"),
            target_base_uri=os.environ.get("TARGET_BASE_ABFSS_URI"),
            # ── runtime-tunable: DB wins over env ─────────────────────────────
            purview_account_name=purview_account_name,
            k_anonymity_min=_int_at_least("K_ANONYMITY_MIN", 5, 1),
            key_vault_enabled=_get("ENABLE_KEY_VAULT", "1").strip().lower() in {
                "1", "true", "yes", "on",
            },
            k_anonymity_tables=_parse_k_anonymity_tables(_get("K_ANONYMITY_TABLES", "")),
            quasi_identifier_cols_by_table=_parse_table_qi_cols(overrides),
            identifier_cols=_csv(_get("IDENTIFIER_COLS", "")),
            sql_endpoint=_get("SQL_ENDPOINT_URL") or None,
            sql_database=_get("SQL_DATABASE") or None,
            gps_precision=_int_at_least("GPS_PRECISION", 1, 0),
            max_table_workers=_int_at_least("MAX_TABLE_WORKERS", 1, 1),
            # ── loaded from pii_column_exclusions table ────────────────────────
            excluded_columns_by_table=excluded_columns or {},
            # ── loaded from pii_table_targets table ───────────────────────────
            table_targets=table_targets or (),
            # ── loaded from the apply_row_scan table ──────────────────────────
            row_scan_columns_by_table=row_scan_columns or {},
            purview_must_anonymize_type=purview_must_anonymize_type,
            purview_not_pii_type=purview_not_pii_type,
        )

    @classmethod
    def from_env_and_db(cls, db: "AuditDB | None") -> "PipelineConfig":
        """Build config from environment variables, overlaid with runtime values
        from ``pii_pipeline_config`` and column exclusions from
        ``pii_column_exclusions`` in PostgreSQL.

        Falls back gracefully to env-only when the DB is unavailable or
        when either table query fails.
        """
        if db is None:
            return cls.from_env()
        overrides: dict[str, str] = {}
        exclusions: dict[str, frozenset[str]] = {}
        targets: tuple[TableMapping, ...] = ()
        try:
            overrides = db.load_runtime_config()
        except Exception as exc:
            logger.warning("Could not load runtime config from DB (using env only): %s", exc)
        try:
            exclusions = db.load_column_exclusions()
        except Exception as exc:
            logger.warning("Could not load column exclusions from DB (none applied): %s", exc)
        try:
            targets = tuple(db.load_table_targets())
        except Exception as exc:
            logger.warning("Could not load table targets from DB (using auto-discovery): %s", exc)
        row_scan: dict[str, frozenset[str]] = {}
        try:
            loaded = db.load_row_scan_columns()
            if isinstance(loaded, dict):
                row_scan = loaded
        except Exception as exc:
            logger.warning("Could not load apply_row_scan config from DB (targeted row scan disabled): %s", exc)
        return cls.from_env(
            config_overrides=overrides,
            excluded_columns=exclusions,
            table_targets=targets,
            row_scan_columns=row_scan,
        )


def _csv(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def _parse_k_anonymity_tables(raw: str = "") -> frozenset[str]:
    """Parse a comma-separated list of table names into a frozenset of lowercase names.

    K-anonymity runs only for tables whose name appears in this set.
    An empty string means k-anonymity is disabled for every table.
    Called by ``PipelineConfig.from_env()`` with the resolved value of
    ``K_ANONYMITY_TABLES`` (DB override wins over env var).
    """
    raw = raw.strip()
    if not raw:
        return frozenset()
    return frozenset(t.strip().lower() for t in raw.split(",") if t.strip())


_QI_PREFIX = "QUASI_IDENTIFIER_COLS__"


def _parse_table_qi_cols(overrides: "dict[str, str] | None" = None) -> dict[str, tuple[str, ...]]:
    """Parse QUASI_IDENTIFIER_COLS__<table_name>=col1,col2,… from env and DB overrides.

    Env vars are loaded first; DB override rows with the same key win per-table.
    Keys are lowercased table names so lookups are case-insensitive.

    Configure in ``pii_pipeline_config``::

        key = 'QUASI_IDENTIFIER_COLS__gps_trips',  value = 'lat,lon,recorded_at'
        key = 'QUASI_IDENTIFIER_COLS__network_logs', value = 'source_ip,dest_ip,event_time'
    """
    result: dict[str, tuple[str, ...]] = {}
    prefix_upper = _QI_PREFIX.upper()
    # Env first
    for key, value in os.environ.items():
        if key.upper().startswith(prefix_upper) and value.strip():
            table = key[len(_QI_PREFIX):].lower()
            if table:
                result[table] = _csv(value)
    # DB overrides win per-table
    for key, value in (overrides or {}).items():
        if key.upper().startswith(prefix_upper) and value.strip():
            table = key[len(_QI_PREFIX):].lower()
            if table:
                result[table] = _csv(value)
    return result


def _env_int_at_least(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be {minimum} or greater")
    return value
