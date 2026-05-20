"""
Fabric PII Anonymization Pipeline
----------------------------------
Reads a Delta table from a Microsoft Fabric OneLake source, anonymizes PII /
GDPR / financial data with Microsoft Presidio, and writes the result to a
separate Fabric OneLake target—all without PySpark.
"""

import logging
import os
import re
import sys
from typing import Optional

import pandas as pd
from azure.identity import DefaultAzureCredential
from deltalake import DeltaTable, write_deltalake
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
ONELAKE_TOKEN_SCOPE = "https://storage.azure.com/.default"
SPACY_MODEL = "en_core_web_lg"
MASK_VALUE = "***"

# Presidio entity types to detect and anonymize
ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_BANK_NUMBER",
]


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def acquire_bearer_token(scope: str = ONELAKE_TOKEN_SCOPE) -> str:
    """
    Acquire an OAuth2 bearer token via DefaultAzureCredential.

    When AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET are set,
    DefaultAzureCredential automatically uses the ClientSecretCredential flow
    (service principal).  Inside Azure-hosted compute it falls back to Managed
    Identity with no extra configuration.
    """
    logger.info("Acquiring bearer token for scope '%s'.", scope)
    credential = DefaultAzureCredential()
    token = credential.get_token(scope)
    logger.info("Bearer token acquired successfully.")
    return token.token


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _parse_account_name(abfss_uri: str) -> str:
    """
    Extract the storage-account name from an abfss:// URI.

    Format: abfss://<container>@<account>.dfs.fabric.microsoft.com/<path>
    For OneLake the account portion is always 'onelake'.
    """
    match = re.search(r"@([^.@/]+)\.", abfss_uri)
    if not match:
        raise ValueError(
            f"Cannot parse account name from URI: '{abfss_uri}'. "
            "Expected format: abfss://container@account.dfs.fabric.microsoft.com/..."
        )
    return match.group(1)


def build_storage_options(abfss_uri: str, bearer_token: str) -> dict:
    """Return delta-rs storage_options for Azure ADLS Gen2 / OneLake."""
    account_name = _parse_account_name(abfss_uri)
    logger.debug("Using storage account: %s", account_name)
    return {
        "account_name": account_name,
        "bearer_token": bearer_token,
    }


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def read_source_table(uri: str, storage_options: dict) -> pd.DataFrame:
    """Load a Delta table from OneLake into a Pandas DataFrame via delta-rs."""
    logger.info("Reading Delta table from: %s", uri)
    dt = DeltaTable(uri, storage_options=storage_options)
    df = dt.to_pandas()
    logger.info("Loaded %d row(s) across %d column(s).", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Presidio anonymization engine
# ---------------------------------------------------------------------------

def build_presidio_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    """
    Initialise Presidio AnalyzerEngine (with en_core_web_lg) and
    AnonymizerEngine.  Building the engines once and reusing them for every
    row is significantly faster than re-initialising per call.
    """
    logger.info("Initialising Presidio NLP engine with model '%s'.", SPACY_MODEL)
    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
    }
    provider = NlpEngineProvider(nlp_configuration=nlp_config)
    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    anonymizer = AnonymizerEngine()
    logger.info("Presidio engines ready.")
    return analyzer, anonymizer


def _anonymize_text(
    text: str,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    operators: dict,
) -> str:
    """Run Presidio analysis + anonymization on a single text value."""
    results = analyzer.analyze(text=text, entities=ENTITIES, language="en")
    if not results:
        return text
    return anonymizer.anonymize(
        text=text, analyzer_results=results, operators=operators
    ).text


def anonymize_dataframe(
    df: pd.DataFrame,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
) -> pd.DataFrame:
    """
    Apply Presidio anonymization to every string-typed column in the DataFrame.
    Non-string columns (int, float, datetime, …) are left unchanged.
    """
    operators = {"DEFAULT": OperatorConfig("replace", {"new_value": MASK_VALUE})}
    df = df.copy()

    text_cols = [col for col in df.columns if df[col].dtype == object]
    logger.info(
        "Anonymising %d text column(s): %s", len(text_cols), text_cols
    )

    for col in text_cols:
        df[col] = df[col].apply(
            lambda val: _anonymize_text(str(val), analyzer, anonymizer, operators)
            if pd.notna(val)
            else val
        )
        logger.info("  ✓ Column '%s' anonymised.", col)

    return df


# ---------------------------------------------------------------------------
# Data load
# ---------------------------------------------------------------------------

def write_target_table(
    df: pd.DataFrame, uri: str, storage_options: dict
) -> None:
    """Write the anonymized DataFrame as a Delta table to the target OneLake path."""
    logger.info("Writing anonymised Delta table to: %s", uri)
    write_deltalake(
        uri,
        df,
        storage_options=storage_options,
        mode="overwrite",
        overwrite_schema=True,
    )
    logger.info("Write complete — %d row(s) written.", len(df))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    source_uri = os.environ["SOURCE_ABFSS_URI"]
    target_uri = os.environ["TARGET_ABFSS_URI"]

    # Single token is valid for both source and target (same OneLake scope)
    bearer_token = acquire_bearer_token()

    source_opts = build_storage_options(source_uri, bearer_token)
    target_opts = build_storage_options(target_uri, bearer_token)

    df_raw = read_source_table(source_uri, source_opts)

    analyzer, anonymizer = build_presidio_engines()
    df_clean = anonymize_dataframe(df_raw, analyzer, anonymizer)

    write_target_table(df_clean, target_uri, target_opts)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
