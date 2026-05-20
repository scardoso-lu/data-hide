"""
Fabric PII Anonymization Pipeline
----------------------------------
Reads a Delta table from Microsoft Fabric OneLake, anonymizes PII / GDPR /
financial data with Microsoft Presidio, optionally cross-checks sensitivity
labels from Microsoft Purview, applies k-anonymity on quasi-identifier
columns, validates no residual PII remains, writes the result to a target
Lakehouse, and records structured audit data to a PostgreSQL database.

The container is fully stateless — no files are written at runtime.
"""

import json
import logging
import os
import re
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, Optional

import pandas as pd
import psycopg2
import psycopg2.extras
import requests
from azure.identity import DefaultAzureCredential
from deltalake import DeltaTable, write_deltalake
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ONELAKE_TOKEN_SCOPE = "https://storage.azure.com/.default"
PURVIEW_TOKEN_SCOPE = "https://purview.azure.net/.default"
SPACY_MODEL         = "en_core_web_lg"
PIPELINE_VERSION    = "2.0.0"

ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "MEDICAL_LICENSE",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_BANK_NUMBER",
    "LOCATION",
    "IP_ADDRESS",
    "URL",
    "DATE_TIME",
    "NRP",
]

def _is_text_column(dtype) -> bool:
    """True for object (mixed) and pandas 3 StringDtype columns."""
    return pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)


FREE_TEXT_KEYWORDS = frozenset({
    "note", "notes", "description", "feedback", "comment", "comments",
    "text", "narrative", "summary", "transcript", "message", "messages",
    "remark", "remarks", "detail", "details", "memo", "body",
})

QI_KEYWORDS = frozenset({
    "age", "gender", "sex", "zip", "zipcode", "zip_code", "postal",
    "postalcode", "postal_code", "city", "country", "region", "state",
    "race", "ethnicity", "nationality", "dob", "birth", "birthday",
    "marital", "occupation",
})

SENSITIVE_COL_PATTERNS = {
    "ssn":        "IDENTIFIER",
    "passport":   "IDENTIFIER",
    "license":    "IDENTIFIER",
    "health":     "SENSITIVE",
    "medical":    "SENSITIVE",
    "diagnosis":  "SENSITIVE",
    "condition":  "SENSITIVE",
    "disease":    "SENSITIVE",
    "salary":     "FINANCIAL",
    "wage":       "FINANCIAL",
    "income":     "FINANCIAL",
    "race":       "SENSITIVE",
    "ethnicity":  "SENSITIVE",
    "religion":   "SENSITIVE",
    "political":  "SENSITIVE",
    "biometric":  "SENSITIVE",
    "genetic":    "SENSITIVE",
    "sexual":     "SENSITIVE",
}

# ─────────────────────────────────────────────────────────────────────────────
# Logging — stdout only; the container runtime captures and forwards these
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

run_id = str(uuid.uuid4())

# ─────────────────────────────────────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────────────────────────────────────
_credential: Optional[DefaultAzureCredential] = None


def _credential_instance() -> DefaultAzureCredential:
    """Singleton DefaultAzureCredential.

    When AZURE_TENANT_ID + AZURE_CLIENT_ID + AZURE_CLIENT_SECRET are present
    DefaultAzureCredential automatically uses the ClientSecretCredential flow.
    On Azure-managed compute it falls back to Managed Identity.
    """
    global _credential
    if _credential is None:
        _credential = DefaultAzureCredential()
    return _credential


def acquire_token(scope: str) -> str:
    logger.info("Acquiring token  scope='%s'", scope)
    token = _credential_instance().get_token(scope)
    logger.info("Token acquired.")
    return token.token


# ─────────────────────────────────────────────────────────────────────────────
# OneLake / storage helpers
# ─────────────────────────────────────────────────────────────────────────────
def _account_name(abfss_uri: str) -> str:
    """Extract account name from abfss://container@account.dfs…/path."""
    m = re.search(r"@([^.@/]+)\.", abfss_uri)
    if not m:
        raise ValueError(
            f"Cannot parse account name from URI: '{abfss_uri}'.  "
            "Expected: abfss://container@account.dfs.fabric.microsoft.com/..."
        )
    return m.group(1)


def _storage_opts(uri: str, token: str) -> dict:
    return {"account_name": _account_name(uri), "bearer_token": token}


def _fresh_opts(uri: str) -> dict:
    """Acquire a fresh token immediately before each storage call.

    DefaultAzureCredential caches tokens and only hits the auth endpoint when
    the cached token is within ~5 minutes of expiry, so this is cheap and
    prevents auth failures when anonymization outlasts the original token TTL.
    """
    return _storage_opts(uri, acquire_token(ONELAKE_TOKEN_SCOPE))


# ─────────────────────────────────────────────────────────────────────────────
# Microsoft Purview — optional sensitivity-label double-check
# ─────────────────────────────────────────────────────────────────────────────
class PurviewClient:
    """Thin wrapper around the Purview Atlas REST Catalog API."""

    def __init__(self, account_name: str, token: str) -> None:
        self._base    = f"https://{account_name}.purview.azure.com"
        self._headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = requests.get(self._base + path, headers=self._headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def column_classifications(self, qualified_name: str) -> dict[str, list[str]]:
        """Return {column_name: [label, ...]} for an ADLS Gen2 path entity.

        Returns {} when the asset is not yet catalogued or on any error.
        """
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
            col    = entity.get("attributes", {}).get("name", "")
            labels = [c["typeName"] for c in entity.get("classifications", [])]
            if col and labels:
                result[col] = labels
        return result

    @staticmethod
    def qualified_name(abfss_uri: str) -> str:
        """abfss://workspace@onelake.dfs.…/lh.Lakehouse/Tables/t
        →  https://onelake.dfs.…/workspace/lh.Lakehouse/Tables/t
        """
        without_scheme = abfss_uri.replace("abfss://", "")
        container, rest = without_scheme.split("@", 1)
        host, path      = rest.split("/", 1)
        return f"https://{host}/{container}/{path}"


def run_purview_check(
    source_uri: str,
    df_columns: list[str],
    purview_account: str | None,
) -> dict:
    """Optional Purview sensitivity-label cross-check. Never raises."""
    empty = {"available": False, "flagged_columns": [], "column_labels": {}, "discrepancies": []}

    if not purview_account:
        logger.info("PURVIEW_ACCOUNT_NAME not set — Purview check skipped.")
        return empty

    try:
        client = PurviewClient(purview_account, acquire_token(PURVIEW_TOKEN_SCOPE))
        qn     = PurviewClient.qualified_name(source_uri)
        logger.info("Querying Purview catalog  qn='%s'", qn)

        col_labels    = client.column_classifications(qn)
        flagged       = list(col_labels.keys())
        discrepancies = [c for c in flagged if c not in df_columns]

        if discrepancies:
            logger.warning("Purview flagged columns absent from DataFrame: %s", discrepancies)

        logger.info("Purview: %d sensitive column(s): %s", len(flagged), flagged)
        return {
            "available":       True,
            "flagged_columns": flagged,
            "column_labels":   col_labels,
            "discrepancies":   discrepancies,
        }
    except Exception as exc:
        logger.warning("Purview check failed (non-fatal): %s", exc)
        return empty


# ─────────────────────────────────────────────────────────────────────────────
# Delta table I/O
# ─────────────────────────────────────────────────────────────────────────────
def read_delta(uri: str, storage_options: dict) -> pd.DataFrame:
    logger.info("Reading Delta table  uri='%s'", uri)
    df = DeltaTable(uri, storage_options=storage_options).to_pandas()
    logger.info("Loaded %d row(s) × %d column(s).", len(df), len(df.columns))
    return df


def write_delta(df: pd.DataFrame, uri: str, storage_options: dict) -> None:
    logger.info("Writing Delta table  uri='%s'", uri)
    write_deltalake(uri, df, storage_options=storage_options, mode="overwrite", overwrite_schema=True)
    logger.info("Write complete — %d row(s).", len(df))


# ─────────────────────────────────────────────────────────────────────────────
# Presidio engine
# ─────────────────────────────────────────────────────────────────────────────
def build_engines() -> AnalyzerEngine:
    logger.info("Initialising Presidio with spaCy model '%s'.", SPACY_MODEL)
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
    })
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
    logger.info("Presidio analyzer ready.")
    return analyzer


# ─────────────────────────────────────────────────────────────────────────────
# Consistent pseudonym tokenisation
# ─────────────────────────────────────────────────────────────────────────────
class EntityRegistry:
    """Assigns consistent pseudonym tokens within a pipeline run.

    The same entity text always maps to the same ENTITY_TYPE_N token so
    relational consistency is preserved for downstream LLM agents — e.g.
    "alice@example.com" always becomes EMAIL_ADDRESS_0 across all rows.
    Matching is case-insensitive and strips surrounding whitespace.
    """

    def __init__(self) -> None:
        self._map: dict[tuple[str, str], str] = {}
        self._counters: dict[str, int] = {}

    def token_for(self, entity_type: str, original: str) -> str:
        key = (entity_type, original.strip().lower())
        if key not in self._map:
            n = self._counters.get(entity_type, 0)
            self._map[key] = f"{entity_type}_{n}"
            self._counters[entity_type] = n + 1
        return self._map[key]

    def unique_counts(self) -> dict[str, int]:
        """Number of distinct values pseudonymised per entity type."""
        return dict(self._counters)


def _anonymize_text(
    text: str,
    analyzer: AnalyzerEngine,
    registry: EntityRegistry,
) -> tuple[str, list]:
    """Return (anonymized_text, [RecognizerResult, ...]).

    Entities are replaced in reverse character-offset order so that earlier
    replacements don't shift the indices of later ones.
    """
    findings = analyzer.analyze(text=text, entities=ENTITIES, language="en")
    if not findings:
        return text, []
    result = text
    for r in sorted(findings, key=lambda x: x.start, reverse=True):
        token  = registry.token_for(r.entity_type, text[r.start:r.end])
        result = result[:r.start] + token + result[r.end:]
    return result, findings


def anonymize_dataframe(
    df: pd.DataFrame,
    analyzer: AnalyzerEngine,
    registry: Optional[EntityRegistry] = None,
) -> tuple[pd.DataFrame, dict]:
    """Anonymize every object-dtype column that contains genuine string values.

    Returns
    -------
    (anonymized_df, stats)  where stats = {
        "text_columns_scanned":    list[str],
        "columns_with_detections": list[str],
        "entity_counts":           {entity_type: int},
        "total_entities_detected": int,
        "column_stats":            [{column, detections, entity_counts}, ...],
    }
    """
    if registry is None:
        registry = EntityRegistry()

    df            = df.copy()
    text_cols     = [c for c in df.columns if _is_text_column(df[c].dtype)]
    entity_counts: dict[str, int] = {}
    cols_hit: list[str]           = []
    column_stats: list[dict]      = []

    logger.info("Scanning %d text column(s): %s", len(text_cols), text_cols)

    for col in text_cols:
        col_detections    = 0
        col_entity_counts: dict[str, int] = {}
        new_values: list  = []

        for val in df[col]:
            # Only process genuine string values.  Non-string objects in an
            # object-dtype column (dicts, lists, Decimal, mixed-type ids, …)
            # pass through unchanged to avoid silent data corruption.
            if not isinstance(val, str):
                new_values.append(val)
                continue
            cleaned, findings = _anonymize_text(val, analyzer, registry)
            new_values.append(cleaned)
            for f in findings:
                col_detections += 1
                entity_counts[f.entity_type] = entity_counts.get(f.entity_type, 0) + 1
                col_entity_counts[f.entity_type] = col_entity_counts.get(f.entity_type, 0) + 1

        df[col] = new_values
        if col_detections:
            cols_hit.append(col)
        column_stats.append({
            "column":        col,
            "detections":    col_detections,
            "entity_counts": col_entity_counts,
        })
        logger.info("  %-30s  detections=%d", f"column='{col}'", col_detections)

    stats = {
        "text_columns_scanned":    text_cols,
        "columns_with_detections": cols_hit,
        "entity_counts":           entity_counts,
        "total_entities_detected": sum(entity_counts.values()),
        "column_stats":            column_stats,
    }
    return df, stats


# ─────────────────────────────────────────────────────────────────────────────
# Free-text column detection
# ─────────────────────────────────────────────────────────────────────────────
def flag_free_text_columns(df: pd.DataFrame) -> list[str]:
    """Return column names whose names suggest unstructured free text.

    Only object-dtype columns are considered (numeric columns are always skipped
    by the anonymizer already).  Emits a warning so operators can decide whether
    additional review is needed.
    """
    flagged = [
        col for col in df.columns
        if _is_text_column(df[col].dtype)
        and any(kw in col.lower() for kw in FREE_TEXT_KEYWORDS)
    ]
    if flagged:
        logger.warning(
            "Free-text columns detected — verify anonymization coverage: %s", flagged
        )
    return flagged


# ─────────────────────────────────────────────────────────────────────────────
# Quasi-identifier detection and k-anonymity enforcement
# ─────────────────────────────────────────────────────────────────────────────
def detect_quasi_identifiers(
    df: pd.DataFrame,
    explicit_cols: list[str] | None = None,
) -> list[str]:
    """Return columns that act as quasi-identifiers.

    Uses explicit_cols when provided (from QUASI_IDENTIFIER_COLS env var).
    Falls back to keyword matching against QI_KEYWORDS for any column present
    in the DataFrame.  Only returns columns that actually exist in df.
    """
    if explicit_cols:
        return [c for c in explicit_cols if c in df.columns]
    return [
        col for col in df.columns
        if any(kw in col.lower() for kw in QI_KEYWORDS)
    ]


def enforce_k_anonymity(
    df: pd.DataFrame,
    quasi_cols: list[str],
    k: int,
) -> tuple[pd.DataFrame, dict]:
    """Drop rows that form groups smaller than k on the quasi-identifier columns.

    Returns (filtered_df, {"suppressed_rows": int, "k": int}).
    When quasi_cols is empty the DataFrame is returned unchanged.
    """
    if not quasi_cols:
        return df, {"suppressed_rows": 0, "k": k}

    present = [c for c in quasi_cols if c in df.columns]
    if not present:
        return df, {"suppressed_rows": 0, "k": k}

    group_sizes = df.groupby(present, dropna=False)[present[0]].transform("count")
    mask        = group_sizes >= k
    filtered    = df[mask].reset_index(drop=True)
    suppressed  = len(df) - len(filtered)

    if suppressed:
        logger.warning(
            "k-anonymity (k=%d): suppressed %d row(s) in groups smaller than k on %s",
            k, suppressed, present,
        )
    return filtered, {"suppressed_rows": suppressed, "k": k}


# ─────────────────────────────────────────────────────────────────────────────
# Column name sanitization
# ─────────────────────────────────────────────────────────────────────────────
def sanitize_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rename columns whose names reveal sensitive categories.

    Returns (df_with_renamed_cols, {old_name: new_name}).
    The rename uses the pattern <CATEGORY>_<index> so that LLMs cannot infer
    the original field semantics from the column header alone.
    """
    renames: dict[str, str] = {}
    counters: dict[str, int] = {}

    for col in df.columns:
        col_lower = col.lower()
        for pattern, category in SENSITIVE_COL_PATTERNS.items():
            if pattern in col_lower:
                n = counters.get(category, 0)
                new_name = f"{category}_{n}"
                counters[category] = n + 1
                renames[col] = new_name
                break

    if renames:
        df = df.rename(columns=renames)
        logger.info("Sanitized column names: %s", renames)
    return df, renames


# ─────────────────────────────────────────────────────────────────────────────
# Residual PII validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_residual_pii(df: pd.DataFrame, analyzer: AnalyzerEngine) -> int:
    """Scan the anonymized DataFrame for any residual PII.

    Raises RuntimeError if any PII is found — the pipeline must not write
    contaminated data to the target Lakehouse.
    Returns the total count of findings (always 0 on success).
    """
    total = 0
    for col in df.columns:
        if not _is_text_column(df[col].dtype):
            continue
        for val in df[col]:
            if not isinstance(val, str):
                continue
            findings = analyzer.analyze(text=val, entities=ENTITIES, language="en")
            total += len(findings)

    if total:
        raise RuntimeError(
            f"Residual PII detected after anonymization: {total} finding(s). "
            "Pipeline aborted — target table was NOT written."
        )
    logger.info("Residual PII validation passed — 0 findings.")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL audit persistence
# ─────────────────────────────────────────────────────────────────────────────
_DDL_RUNS = """
CREATE TABLE IF NOT EXISTS pii_pipeline_runs (
    run_id           UUID        PRIMARY KEY,
    pipeline_version TEXT        NOT NULL,
    started_at       TIMESTAMPTZ NOT NULL,
    finished_at      TIMESTAMPTZ,
    source_uri       TEXT        NOT NULL,
    target_uri       TEXT        NOT NULL,
    total_rows       INTEGER,
    total_columns    INTEGER,
    columns_scanned  INTEGER,
    columns_hit      JSONB,
    entities_total   INTEGER,
    entity_counts    JSONB,
    unique_entities  JSONB,
    free_text_cols   JSONB,
    k_anonymity_k    INTEGER,
    quasi_columns    JSONB,
    suppressed_rows  INTEGER     NOT NULL DEFAULT 0,
    residual_pii     INTEGER     NOT NULL DEFAULT 0,
    column_renames   JSONB,
    purview_ok       BOOLEAN     NOT NULL DEFAULT FALSE,
    purview_flagged  JSONB,
    purview_diffs    JSONB,
    status           TEXT        NOT NULL DEFAULT 'running',
    error_msg        TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_DDL_COLUMN_EVENTS = """
CREATE TABLE IF NOT EXISTS pii_pipeline_column_events (
    id            BIGSERIAL   PRIMARY KEY,
    run_id        UUID        NOT NULL REFERENCES pii_pipeline_runs(run_id),
    column_name   TEXT        NOT NULL,
    detections    INTEGER     NOT NULL DEFAULT 0,
    entity_counts JSONB,
    processed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


class AuditDB:
    """PostgreSQL-backed audit persistence.

    Two tables are managed:
      pii_pipeline_runs          — one row per pipeline execution
      pii_pipeline_column_events — one row per text column scanned per run

    All public methods are safe to call from a finally block: callers should
    wrap each call in try/except so a DB hiccup never aborts the pipeline.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._init_schema()

    @contextmanager
    def _cursor(self) -> Generator:
        """Open a short-lived connection, commit on success, rollback on error."""
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

    def open_run(self, started_at: datetime, source_uri: str, target_uri: str) -> None:
        """Insert a 'running' row so in-progress pipelines are visible in the DB."""
        sql = """
            INSERT INTO pii_pipeline_runs
                (run_id, pipeline_version, started_at, source_uri, target_uri, status)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        with self._cursor() as cur:
            cur.execute(sql, (run_id, PIPELINE_VERSION, started_at, source_uri, target_uri, "running"))

    def record_columns(self, column_stats: list[dict]) -> None:
        """Bulk-insert per-column processing events."""
        rows = [
            (run_id, s["column"], s["detections"], json.dumps(s["entity_counts"]))
            for s in column_stats
        ]
        sql = """
            INSERT INTO pii_pipeline_column_events
                (run_id, column_name, detections, entity_counts)
            VALUES %s
        """
        with self._cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, rows)

    def close_run(self, audit: dict) -> None:
        """Update the run row with final counters and terminal status."""
        sql = """
            UPDATE pii_pipeline_runs SET
                finished_at     = %s,
                total_rows      = %s,
                total_columns   = %s,
                columns_scanned = %s,
                columns_hit     = %s,
                entities_total  = %s,
                entity_counts   = %s,
                unique_entities = %s,
                free_text_cols  = %s,
                k_anonymity_k   = %s,
                quasi_columns   = %s,
                suppressed_rows = %s,
                residual_pii    = %s,
                column_renames  = %s,
                purview_ok      = %s,
                purview_flagged = %s,
                purview_diffs   = %s,
                status          = %s,
                error_msg       = %s
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
                audit.get("purview_available", False),
                json.dumps(audit.get("purview_flagged_columns", [])),
                json.dumps(audit.get("purview_discrepancies", [])),
                audit.get("status"),
                audit.get("error_message"),
                run_id,
            ))


def connect_audit_db(database_url: str | None) -> Optional[AuditDB]:
    """Attempt to connect. Returns None (non-fatal) when DATABASE_URL is unset
    or the connection fails."""
    if not database_url:
        logger.info("DATABASE_URL not set — audit DB disabled.")
        return None
    try:
        db = AuditDB(database_url)
        logger.info("Audit DB connected and schema verified.")
        return db
    except Exception as exc:
        logger.warning("Audit DB connection failed (non-fatal): %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Alerting
# ─────────────────────────────────────────────────────────────────────────────
def send_alert(subject: str, body: str, webhook_url: str | None) -> None:
    """POST a JSON payload to a Teams / Slack / generic incoming webhook."""
    if not webhook_url:
        logger.warning("ALERT_WEBHOOK_URL not configured — alert suppressed: %s", subject)
        return
    payload = {"text": f"*[Fabric PII Pipeline] {subject}*\n{body}"}
    try:
        resp = requests.post(webhook_url, json=payload, timeout=15)
        resp.raise_for_status()
        logger.info("Alert dispatched to webhook.")
    except Exception as exc:
        logger.error("Webhook alert delivery failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestration
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    source_uri   = os.environ["SOURCE_ABFSS_URI"]
    target_uri   = os.environ["TARGET_ABFSS_URI"]
    database_url = os.environ.get("DATABASE_URL")
    purview_acct = os.environ.get("PURVIEW_ACCOUNT_NAME")
    webhook_url  = os.environ.get("ALERT_WEBHOOK_URL")
    k_min        = int(os.environ.get("K_ANONYMITY_MIN", "5"))
    qi_env       = os.environ.get("QUASI_IDENTIFIER_COLS", "")
    qi_cols_cfg  = [c.strip() for c in qi_env.split(",") if c.strip()]

    pipeline_start = datetime.now(timezone.utc)
    logger.info("Pipeline started  run_id=%s  ts=%s", run_id, pipeline_start.isoformat())

    audit: dict = {
        "pipeline_end_ts":         None,
        "total_rows_processed":    0,
        "total_columns_in_table":  0,
        "total_columns_scanned":   0,
        "columns_anonymized":      [],
        "total_entities_detected": 0,
        "entity_counts":           {},
        "unique_entities":         {},
        "free_text_columns":       [],
        "k_anonymity_k":           k_min,
        "quasi_columns":           [],
        "suppressed_rows":         0,
        "residual_pii_count":      0,
        "column_renames":          {},
        "purview_available":       False,
        "purview_flagged_columns": [],
        "purview_discrepancies":   [],
        "status":                  "failure",
        "error_message":           None,
    }

    db = connect_audit_db(database_url)
    if db:
        try:
            db.open_run(pipeline_start, source_uri, target_uri)
        except Exception as exc:
            logger.warning("Audit open_run failed (non-fatal): %s", exc)

    try:
        # ── Extract ───────────────────────────────────────────────────────────
        df_raw = read_delta(source_uri, _fresh_opts(source_uri))
        audit["total_rows_processed"]   = len(df_raw)
        audit["total_columns_in_table"] = len(df_raw.columns)

        # ── Column name sanitization ──────────────────────────────────────────
        df_raw, col_renames = sanitize_column_names(df_raw)
        audit["column_renames"] = col_renames

        # ── Free-text column detection ────────────────────────────────────────
        free_text_cols = flag_free_text_columns(df_raw)
        audit["free_text_columns"] = free_text_cols

        # ── Purview double-check ──────────────────────────────────────────────
        pv = run_purview_check(source_uri, list(df_raw.columns), purview_acct)
        audit["purview_available"]       = pv["available"]
        audit["purview_flagged_columns"] = pv["flagged_columns"]
        audit["purview_discrepancies"]   = pv["discrepancies"]

        # ── Quasi-identifier / k-anonymity enforcement ────────────────────────
        qi_cols = detect_quasi_identifiers(df_raw, qi_cols_cfg)
        audit["quasi_columns"] = qi_cols
        if qi_cols:
            df_raw, k_info = enforce_k_anonymity(df_raw, qi_cols, k_min)
            audit["suppressed_rows"] = k_info["suppressed_rows"]
            logger.info(
                "k-anonymity: k=%d  quasi_cols=%s  suppressed=%d",
                k_min, qi_cols, k_info["suppressed_rows"],
            )

        # ── Anonymize ─────────────────────────────────────────────────────────
        analyzer = build_engines()
        registry = EntityRegistry()
        df_clean, stats = anonymize_dataframe(df_raw, analyzer, registry)
        audit["total_columns_scanned"]   = len(stats["text_columns_scanned"])
        audit["columns_anonymized"]      = stats["columns_with_detections"]
        audit["total_entities_detected"] = stats["total_entities_detected"]
        audit["entity_counts"]           = stats["entity_counts"]
        audit["unique_entities"]         = registry.unique_counts()

        if db and stats["column_stats"]:
            try:
                db.record_columns(stats["column_stats"])
            except Exception as exc:
                logger.warning("Audit record_columns failed (non-fatal): %s", exc)

        # ── Residual PII validation ───────────────────────────────────────────
        residual_count = validate_residual_pii(df_clean, analyzer)
        audit["residual_pii_count"] = residual_count

        # ── Load ──────────────────────────────────────────────────────────────
        write_delta(df_clean, target_uri, _fresh_opts(target_uri))

        # ── Mark success ──────────────────────────────────────────────────────
        pipeline_end             = datetime.now(timezone.utc)
        audit["pipeline_end_ts"] = pipeline_end.isoformat()
        audit["status"]          = "success"
        logger.info(
            "Pipeline SUCCESS  run_id=%s  rows=%d  entities=%d  cols_anonymized=%d  duration=%.1fs",
            run_id,
            audit["total_rows_processed"],
            audit["total_entities_detected"],
            len(audit["columns_anonymized"]),
            (pipeline_end - pipeline_start).total_seconds(),
        )

    except Exception as exc:
        pipeline_end             = datetime.now(timezone.utc)
        audit["pipeline_end_ts"] = pipeline_end.isoformat()
        audit["error_message"]   = str(exc)
        logger.exception("Pipeline FAILED  run_id=%s  error=%s", run_id, exc)
        send_alert(
            "Pipeline FAILED",
            f"run_id : {run_id}\nsource  : {source_uri}\nerror   : {exc}",
            webhook_url,
        )
        raise

    finally:
        if db:
            try:
                db.close_run(audit)
            except Exception as exc:
                logger.warning("Audit close_run failed (non-fatal): %s", exc)


if __name__ == "__main__":
    main()
