"""
Fabric PII Anonymization Pipeline
----------------------------------
Reads a Delta table from Microsoft Fabric OneLake, anonymizes PII / GDPR /
financial data with Microsoft Presidio, optionally cross-checks sensitivity
labels from Microsoft Purview, writes the result to a target Lakehouse, and
persists structured audit records both as local JSONL files and as a central
Delta audit table.
"""

import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from azure.identity import DefaultAzureCredential
from deltalake import DeltaTable, write_deltalake
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ONELAKE_TOKEN_SCOPE = "https://storage.azure.com/.default"
PURVIEW_TOKEN_SCOPE = "https://purview.azure.net/.default"
SPACY_MODEL         = "en_core_web_lg"
MASK_VALUE          = "***"
PIPELINE_VERSION    = "1.1.0"

ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_BANK_NUMBER",
]

# ─────────────────────────────────────────────────────────────────────────────
# Logging — stdout + persistent rolling file + per-run JSONL audit stream
# ─────────────────────────────────────────────────────────────────────────────
LOG_DIR = Path(os.environ.get("LOG_DIR", "/app/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

run_id     = str(uuid.uuid4())
jsonl_path = LOG_DIR / f"audit_{run_id}.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


def _emit(event: dict) -> None:
    """Append one structured event to the run-scoped JSONL audit file."""
    record = {"run_id": run_id, "ts": datetime.now(timezone.utc).isoformat(), **event}
    with open(jsonl_path, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────────────────────────────────────
_credential: Optional[DefaultAzureCredential] = None


def _credential_instance() -> DefaultAzureCredential:
    """Return a singleton DefaultAzureCredential.

    When AZURE_TENANT_ID + AZURE_CLIENT_ID + AZURE_CLIENT_SECRET are present
    DefaultAzureCredential automatically uses the ClientSecretCredential flow
    (service principal).  On Azure-managed compute it falls back to Managed
    Identity with no extra configuration.
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
    """Extract the storage-account name from an abfss:// URI.

    Format:  abfss://<container>@<account>.dfs.fabric.microsoft.com/<path>
    For OneLake the account is always 'onelake'.
    """
    m = re.search(r"@([^.@/]+)\.", abfss_uri)
    if not m:
        raise ValueError(
            f"Cannot parse account name from URI: '{abfss_uri}'.  "
            "Expected format: abfss://container@account.dfs.fabric.microsoft.com/..."
        )
    return m.group(1)


def _storage_opts(uri: str, token: str) -> dict:
    return {"account_name": _account_name(uri), "bearer_token": token}


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
        """Return {column_name: [sensitivity_label, ...]} for an ADLS Gen2 path entity.

        Returns an empty dict when the asset is not yet catalogued or on any error.
        """
        try:
            data = self._get(
                "/catalog/api/atlas/v2/entity/uniqueAttribute/type/azure_datalake_gen2_path",
                params={"attr:qualifiedName": qualified_name},
            )
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            if code == 404:
                logger.warning("Purview: asset not catalogued  qn='%s'", qualified_name)
            else:
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
        """Convert an abfss:// OneLake URI to a Purview Atlas qualified name.

        abfss://workspace@onelake.dfs.fabric.microsoft.com/lh.Lakehouse/Tables/tbl
        →  https://onelake.dfs.fabric.microsoft.com/workspace/lh.Lakehouse/Tables/tbl
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
    """Optional Purview sensitivity-label cross-check.  Never raises.

    Returns
    -------
    dict with keys:
        available         bool
        flagged_columns   list[str]   columns Purview classified as sensitive
        column_labels     dict        {col: [label, ...]}
        discrepancies     list[str]   Purview-flagged columns absent from DataFrame
    """
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
            logger.warning(
                "Purview flagged column(s) not in DataFrame (schema drift?): %s",
                discrepancies,
            )

        logger.info(
            "Purview: %d sensitive column(s) identified: %s", len(flagged), flagged
        )
        result = {
            "available":       True,
            "flagged_columns": flagged,
            "column_labels":   col_labels,
            "discrepancies":   discrepancies,
        }
        _emit({"event": "purview_check", "qualified_name": qn, **result})
        return result

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


def write_delta(df: pd.DataFrame, uri: str, storage_options: dict, mode: str = "overwrite") -> None:
    logger.info("Writing Delta table  mode=%s  uri='%s'", mode, uri)
    write_deltalake(uri, df, storage_options=storage_options, mode=mode, overwrite_schema=True)
    logger.info("Write complete — %d row(s).", len(df))


# ─────────────────────────────────────────────────────────────────────────────
# Presidio engines
# ─────────────────────────────────────────────────────────────────────────────
def build_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    logger.info("Initialising Presidio with spaCy model '%s'.", SPACY_MODEL)
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
    })
    analyzer  = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
    anonymizer = AnonymizerEngine()
    logger.info("Presidio engines ready.")
    return analyzer, anonymizer


def _process_value(
    text: str,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    operators: dict,
) -> tuple[str, list]:
    """Return (anonymized_text, [RecognizerResult, ...])."""
    findings = analyzer.analyze(text=text, entities=ENTITIES, language="en")
    if not findings:
        return text, []
    return anonymizer.anonymize(text=text, analyzer_results=findings, operators=operators).text, findings


def anonymize_dataframe(
    df: pd.DataFrame,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
) -> tuple[pd.DataFrame, dict]:
    """Anonymize every object-dtype column.

    Returns
    -------
    (anonymized_df, stats)  where stats = {
        "text_columns_scanned":  list[str],
        "columns_with_detections": list[str],
        "entity_counts":         {entity_type: int},
        "total_entities_detected": int,
    }
    """
    operators     = {"DEFAULT": OperatorConfig("replace", {"new_value": MASK_VALUE})}
    df            = df.copy()
    text_cols     = [c for c in df.columns if df[c].dtype == object]
    entity_counts = {e: 0 for e in ENTITIES}
    cols_hit: list[str] = []

    logger.info("Scanning %d text column(s): %s", len(text_cols), text_cols)

    for col in text_cols:
        col_detections   = 0
        col_entity_counts = {e: 0 for e in ENTITIES}
        new_values: list = []

        for val in df[col]:
            if pd.isna(val):
                new_values.append(val)
                continue
            cleaned, findings = _process_value(str(val), analyzer, anonymizer, operators)
            new_values.append(cleaned)
            for f in findings:
                col_detections += 1
                if f.entity_type in entity_counts:
                    entity_counts[f.entity_type]     += 1
                    col_entity_counts[f.entity_type] += 1

        df[col] = new_values
        if col_detections:
            cols_hit.append(col)

        # Persistent per-column audit event
        _emit({
            "event":          "column_processed",
            "column":         col,
            "total_detections": col_detections,
            "entity_counts":  col_entity_counts,
        })
        logger.info("  %-30s  detections=%d", f"column='{col}'", col_detections)

    stats = {
        "text_columns_scanned":    text_cols,
        "columns_with_detections": cols_hit,
        "entity_counts":           entity_counts,
        "total_entities_detected": sum(entity_counts.values()),
    }
    return df, stats


# ─────────────────────────────────────────────────────────────────────────────
# Central audit table
# ─────────────────────────────────────────────────────────────────────────────
def write_audit_record(record: dict, audit_uri: str | None, storage_opts: dict) -> None:
    """Append one row to the central anonymization audit Delta table.

    Schema (all list/dict values are JSON-encoded strings for portability):
        run_id, pipeline_version, pipeline_start_ts, pipeline_end_ts,
        source_uri, target_uri, total_rows_processed, total_columns_in_table,
        total_columns_scanned, columns_anonymized (JSON), total_entities_detected,
        entity_counts (JSON), purview_available, purview_flagged_columns (JSON),
        purview_discrepancies (JSON), status, error_message
    """
    if not audit_uri:
        logger.info("AUDIT_ABFSS_URI not set — central audit table write skipped.")
        return
    try:
        flat = {
            k: json.dumps(v, default=str) if isinstance(v, (list, dict)) else v
            for k, v in record.items()
        }
        write_delta(pd.DataFrame([flat]), audit_uri, storage_opts, mode="append")
    except Exception as exc:
        logger.warning("Audit table write failed (non-fatal): %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Alerting
# ─────────────────────────────────────────────────────────────────────────────
def send_alert(subject: str, body: str, webhook_url: str | None) -> None:
    """POST a JSON alert to a generic / Teams / Slack incoming webhook.

    Both Microsoft Teams (Incoming Webhook connector) and Slack accept the
    simple  {"text": "..."}  payload shape used here.
    Set ALERT_WEBHOOK_URL to enable; leave unset to suppress.
    """
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
    source_uri    = os.environ["SOURCE_ABFSS_URI"]
    target_uri    = os.environ["TARGET_ABFSS_URI"]
    audit_uri     = os.environ.get("AUDIT_ABFSS_URI")
    purview_acct  = os.environ.get("PURVIEW_ACCOUNT_NAME")
    webhook_url   = os.environ.get("ALERT_WEBHOOK_URL")

    pipeline_start = datetime.now(timezone.utc)
    logger.info("Pipeline started  run_id=%s  ts=%s", run_id, pipeline_start.isoformat())
    _emit({"event": "pipeline_start", "source_uri": source_uri, "target_uri": target_uri})

    # Audit record template — filled in progressively; always written in finally
    audit: dict = {
        "run_id":                  run_id,
        "pipeline_version":        PIPELINE_VERSION,
        "pipeline_start_ts":       pipeline_start.isoformat(),
        "pipeline_end_ts":         None,
        "source_uri":              source_uri,
        "target_uri":              target_uri,
        "total_rows_processed":    0,
        "total_columns_in_table":  0,
        "total_columns_scanned":   0,
        "columns_anonymized":      [],
        "total_entities_detected": 0,
        "entity_counts":           {},
        "purview_available":       False,
        "purview_flagged_columns": [],
        "purview_discrepancies":   [],
        "status":                  "failure",
        "error_message":           None,
    }
    audit_storage_opts: dict = {}

    try:
        # ── Auth ──────────────────────────────────────────────────────────────
        onelake_token      = acquire_token(ONELAKE_TOKEN_SCOPE)
        src_opts           = _storage_opts(source_uri, onelake_token)
        tgt_opts           = _storage_opts(target_uri, onelake_token)
        if audit_uri:
            audit_storage_opts = _storage_opts(audit_uri, onelake_token)

        # ── Extract ───────────────────────────────────────────────────────────
        df_raw = read_delta(source_uri, src_opts)
        audit["total_rows_processed"]   = len(df_raw)
        audit["total_columns_in_table"] = len(df_raw.columns)

        # ── Purview double-check ──────────────────────────────────────────────
        pv = run_purview_check(source_uri, list(df_raw.columns), purview_acct)
        audit["purview_available"]       = pv["available"]
        audit["purview_flagged_columns"] = pv["flagged_columns"]
        audit["purview_discrepancies"]   = pv["discrepancies"]

        # ── Anonymize ─────────────────────────────────────────────────────────
        analyzer, anonymizer = build_engines()
        df_clean, stats      = anonymize_dataframe(df_raw, analyzer, anonymizer)
        audit["total_columns_scanned"]   = len(stats["text_columns_scanned"])
        audit["columns_anonymized"]      = stats["columns_with_detections"]
        audit["total_entities_detected"] = stats["total_entities_detected"]
        audit["entity_counts"]           = stats["entity_counts"]

        # ── Load ──────────────────────────────────────────────────────────────
        write_delta(df_clean, target_uri, tgt_opts)

        # ── Mark success ──────────────────────────────────────────────────────
        pipeline_end           = datetime.now(timezone.utc)
        audit["pipeline_end_ts"] = pipeline_end.isoformat()
        audit["status"]          = "success"

        logger.info(
            "Pipeline SUCCESS  run_id=%s  rows=%d  entities=%d  cols_anonymized=%d",
            run_id,
            audit["total_rows_processed"],
            audit["total_entities_detected"],
            len(audit["columns_anonymized"]),
        )
        _emit({
            "event":           "pipeline_success",
            "rows":            audit["total_rows_processed"],
            "entities":        audit["total_entities_detected"],
            "cols_anonymized": audit["columns_anonymized"],
            "duration_s":      (pipeline_end - pipeline_start).total_seconds(),
        })

    except Exception as exc:
        pipeline_end             = datetime.now(timezone.utc)
        audit["pipeline_end_ts"] = pipeline_end.isoformat()
        audit["error_message"]   = str(exc)

        logger.exception("Pipeline FAILED  run_id=%s  error=%s", run_id, exc)
        _emit({"event": "pipeline_failure", "error": str(exc)})
        send_alert(
            "Pipeline FAILED",
            f"run_id : {run_id}\nsource  : {source_uri}\nerror   : {exc}",
            webhook_url,
        )
        raise

    finally:
        write_audit_record(audit, audit_uri, audit_storage_opts)


if __name__ == "__main__":
    main()
