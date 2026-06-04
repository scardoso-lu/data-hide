# Fabric PII Anonymization Pipeline

A containerized, **stateless** Python pipeline that discovers every Delta table under a OneLake base path, anonymizes all personal data found in text columns using a layered detection stack (Purview metadata → Presidio-structured value sampling → spaCy embedding similarity on column names → row-by-row Presidio NER with a semantic Art. 9 / Art. 10 detector), and writes clean Delta tables back to OneLake — with a full PostgreSQL audit trail on every run.

The pipeline contains **no hand-curated keyword lists**. Every category of PII — including health conditions, religion, ethnicity, sexual orientation, trade-union membership, and criminal records — is detected through spaCy embeddings + rapidfuzz typo tolerance against a tiny set of concept anchors carried in code. Compliance and engineering teams never have to maintain `.txt` files of variants.

---

## How it works

```mermaid
flowchart TD
    A["OneLake — Delta tables\n(SOURCE_BASE_ABFSS_URI)"]
    A2["SQL Analytics Endpoint\n(SQL_ENDPOINT_URL · optional)"]
    B["DuckDB-filtered source read\npast READ_LOOKBACK_DAYS"]
    B2["Pandas DataFrame\nlimited rows per table"]
    C["Dynamic Column Classification\nclassify_columns"]
    D["Identifier pseudonymisation +\nGPS rounding + temporal binning"]
    CP["Column-Policy Layer\n(Purview · Presidio-structured ·\nspaCy embedding similarity)"]
    E["Presidio row-by-row scan\n+ semantic Art. 9 / Art. 10 recognizer\n(EN · FR · DE · LB)"]
    F["Purview sensitivity check\nrun_purview_check (optional)"]
    G["Residual PII Validation\nvalidate_residual_pii"]
    H["OneLake — Delta tables\n(TARGET_BASE_ABFSS_URI)"]
    I["Pipeline aborts\ntarget not written"]
    J["PostgreSQL Audit DB\nAuditDB · open_run / close_run"]

    A -->|"discover_table_mappings\nread_mode=delta"| B
    A2 -.->|"shortcuts not in ADLS\nread_mode=sql"| B
    B --> B2
    B2 --> C
    C --> D
    D --> CP
    CP --> E
    B -.->|optional| F
    F -.-> E
    E --> G
    G -->|clean| H
    G -->|PII found| I
    E --> J
```

See [`docs/learn_more.md`](docs/learn_more.md) for a breakdown of each detection layer.

---

## GDPR coverage

| GDPR Article | Obligation | Implementation |
|---|---|---|
| Art. 4(1) — Personal data | Identify and protect all natural-person identifiers | NLP entity detection across text columns |
| Art. 5(1)(b) — Purpose limitation | Write anonymized data to a separate target; never overwrite source | Source ≠ target URI guard before any write |
| Art. 5(1)(c) — Data minimisation | Limit source rows by time | Source reads default to the past `365` days via temporal columns before rows are materialized for anonymization |
| Art. 5(1)(c) — Data minimisation | Remove direct identifiers | Auto-detect identifier columns and replace with deterministic tokens derived from an RSA key in Azure Key Vault |
| Art. 5(1)(c) — Data minimisation | Reduce GPS precision | Round lat/lon to N decimal places; floor co-located timestamps to day |
| Art. 5(1)(c) — Data minimisation | Suppress rare quasi-identifier combinations | k-anonymity: groups smaller than `K_ANONYMITY_MIN` are dropped |
| Art. 5(1)(d) — Accuracy | Guarantee no residual PII in output | Scan every text cell after anonymization; abort if anything remains |
| Art. 5(1)(f) — Integrity & confidentiality | Cryptographically protect identifiers | HSM-bound key derivation: an RSA key in Azure Key Vault signs a fixed constant once per run; the resulting secret stays in process memory and drives HMAC-SHA256 pseudonymization |
| Art. 25 — Data protection by design | Pseudonymise by default | Replace PII spans with stable tokens (`PERSON_0`, `EMAIL_ADDRESS_1`, …) |
| Art. 30 — Records of processing activities | Log every processing activity | One PostgreSQL row per run and per column with entity counts |
| Art. 32 — Security of processing | Appropriate technical and organisational measures | Non-root container, no runtime file writes, ABFSS TLS, salted hashes |
| Art. 35 — Data protection impact assessment | Document processing risks | Per-run audit records include entity counts, suppressed rows, and residual PII count |

For full details on each GDPR article see [gdpr-info.eu](https://gdpr-info.eu).

---

## Installation

### Prerequisites

| Tool | Version |
|---|---|
| [uv](https://docs.astral.sh/uv/) | 0.5+ (for local development) |
| Docker + Docker Compose | 24+ |
| Azure Service Principal or Managed Identity | — |
| PostgreSQL | 14+ (provided by Compose for local runs) |

uv manages the virtualenv, the Python version (read from `.python-version`), and the dependency graph (resolved into the committed `uv.lock`). Install it via `winget install --id=astral-sh.uv`, `brew install uv`, or `curl -LsSf https://astral.sh/uv/install.sh | sh`.

The Service Principal needs:

| Resource | Required role |
|---|---|
| Source OneLake | Storage Blob Data Reader |
| Target OneLake | Storage Blob Data Contributor |

### 1 — Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your values:

| Variable | Required | Description |
|---|---|---|
| `AZURE_TENANT_ID` | Yes | Azure AD tenant ID |
| `AZURE_CLIENT_ID` | Yes | Service principal client ID |
| `AZURE_CLIENT_SECRET` | Yes | Service principal secret |
| `DATABASE_URL` | Yes | PostgreSQL DSN for audit records |
| `SOURCE_BASE_ABFSS_URI` | Yes | Base path of raw Delta tables to anonymize |
| `TARGET_BASE_ABFSS_URI` | Yes | Target Lakehouse Tables root where anonymized Delta tables are written. Must end with `<lakehouse>.Lakehouse/Tables`; do not use `Files/...` if the output should appear as Fabric tables. |
| `READ_LOOKBACK_DAYS` | No (default `365`) | Number of recent days to read from each source table. Delta reads are filtered through DuckDB before DataFrame materialization; SQL shortcut reads push the same cutoff into T-SQL. Tables with no temporal column are logged and read fully because the pipeline cannot infer row age. |
| `MAX_TABLE_WORKERS` | No (default `1`) | Number of tables to process concurrently. Increase only when the host has enough CPU and memory for concurrent NLP work. |
| `MAX_UPLOAD_WORKERS` | No (default `4`) | Bounded concurrent uploads for Delta data files; `_delta_log` files are uploaded after data files. |
| `K_ANONYMITY_MIN` | No (default `5`) | Minimum group size for quasi-identifier suppression |
| `ENABLE_KEY_VAULT` | No (default `1`) | Set to `0` to disable Azure Key Vault and use a local HMAC-SHA256 hash instead. Useful for development/testing when no Key Vault is available. When `0`, `KEY_VAULT_URL` and `KEY_VAULT_RSA_KEY_NAME` are ignored. |
| `HASH_SALT` | If `ENABLE_KEY_VAULT=0` | Secret salt for the local hash pseudonymizer. If omitted a fixed default is used — deterministic but not secret, so only suitable for local development. |
| `KEY_VAULT_URL` | If identifier columns exist and `ENABLE_KEY_VAULT=1` | Azure Key Vault URL (e.g. `https://my-vault.vault.azure.net/`) |
| `KEY_VAULT_RSA_KEY_NAME` | If identifier columns exist and `ENABLE_KEY_VAULT=1` | RSA key name used to derive pseudonyms; the latest enabled version is always used and never leaves the HSM |
| `GPS_PRECISION` | No (default `2`) | Decimal places for GPS rounding (2 is about 1 km) |
| `SQL_ENDPOINT_URL` | No | Fabric SQL Analytics Endpoint — enables shortcut discovery |
| `SQL_DATABASE` | No | Database name on the SQL endpoint |
| **Purview integration** | | |
| `PURVIEW_ACCOUNT_NAME` | No | Microsoft Purview account name (e.g. `my-purview`). When set, column classifications from the Purview catalog are fetched before Presidio runs and used as the authoritative Tier A classification source. Requires `PURVIEW_CLIENT_ID`, `PURVIEW_CLIENT_SECRET`, and `PURVIEW_MUST_ANONYMIZE_TYPE` to also be set. |
| `PURVIEW_CLIENT_ID` | If `PURVIEW_ACCOUNT_NAME` set | Client ID of the **dedicated Purview service principal**. Purview uses a separate service principal from the Fabric/OneLake pipeline (`AZURE_CLIENT_ID`) — do not reuse the same credentials. |
| `PURVIEW_CLIENT_SECRET` | If `PURVIEW_ACCOUNT_NAME` set | Client secret for the Purview service principal. |
| `PURVIEW_TENANT_ID` | No | Azure AD tenant for the Purview service principal. Defaults to `AZURE_TENANT_ID` when not set — only needed when the Purview account lives in a different tenant from the Fabric workspace. |
| `PURVIEW_MUST_ANONYMIZE_TYPE` | If `PURVIEW_ACCOUNT_NAME` set | Name of a **custom Purview classification type** that means "this column must be anonymized". Columns bearing this type are assigned `ACTION_REDACT` (all values replaced with `[REDACTED]`) regardless of what Presidio or spaCy would classify them as. This classification overrides any pre-computed policy, including Phase 1 sampling results. Configure the matching custom classification in your Purview account and set the same name here (e.g. `MUST_ANONYMIZE`). |
| `PURVIEW_NOT_PII_TYPE` | No | Name of a **custom Purview classification type** that means "this column has been verified as non-PII". Columns bearing this type are excluded from all analysis — both column-policy masking and row-level Presidio scanning — exactly like a `pii_column_exclusions` database entry. Use this for columns that pattern-match to PII (product codes, hex hashes, generic dates) but are known to contain no personal data. |
| `ENABLE_COLUMN_POLICY` | No (default `1`) | Master switch for the column-policy layer (Purview → presidio-structured → spaCy similarity). Set to `0` to fall back to row-by-row Presidio only. |
| `ENABLE_PRESIDIO_STRUCTURED` | No (default `1`) | Tier B1 toggle — value-sampling column classification via `presidio-structured`. Disable if your dataset has very small column heights and the votes become unreliable. |
| `COLUMN_SIMILARITY_THRESHOLD` | No (default `0.55`) | Tier B2 cosine threshold for spaCy embedding similarity between a column name and `CONCEPT_SEEDS`. Lower → more aggressive name-based classification; higher → more conservative. |
| `SEMANTIC_SIMILARITY_THRESHOLD` | No (default `0.55`) | Token-level cosine threshold for the semantic Art. 9 / Art. 10 recognizer. |
| `ENABLE_FUZZY_TYPO_MATCH` | No (default `0`) | Opt-in rapidfuzz fuzzy matcher for any remaining deny-list recognizers. The semantic recognizer has its own fuzzy fallback (always on) — this flag is only relevant if you re-introduce deny-list-based custom recognizers. |
| `ANONYMIZATION_REGIONS` | No (default `all`) | Comma-separated regions to enable national-ID detection for (`us`, `eu`, `uk`). `eu` drops US-only entities. |
| `SPACY_MODEL_EN` / `_FR` / `_DE` / `_LB` | No | Per-language spaCy model overrides. Default is `en_core_web_lg`, `fr_core_news_lg`, `de_core_news_lg`, `de_core_news_lg` (Luxembourgish uses the German model — no dedicated model exists). Overriding to `_md` saves ~800 MB peak RSS but fails 12 protected detection cases (semantic recognizer thresholds are tuned to `_lg` vectors) — only use in deployments that accept the recall loss. |

OneLake URI format:
```
abfss://<WorkspaceName>@onelake.dfs.fabric.microsoft.com/<LakehouseName>.Lakehouse/Tables/<path>
```
Copy from Fabric portal: open the Lakehouse → right-click the table → **Properties** → **ABFS path**.

### 2 — Run with Docker Compose

```bash
docker compose up --build
```

Compose starts a managed PostgreSQL instance alongside the pipeline — no extra database setup needed.

### 3 — Run as a standalone container

```bash
docker build -t fabric-pii-pipeline:latest .

docker run --rm \
  --env-file .env \
  -e DATABASE_URL="postgresql://user:pass@your-pg-host:5432/pii_audit" \
  fabric-pii-pipeline:latest
```

### 4 — Local development

```bash
uv sync                                       # create .venv from uv.lock
uv run python -m spacy download en_core_web_lg
uv run python -m spacy download fr_core_news_lg
uv run python -m spacy download de_core_news_lg
uv run pytest                                 # full test suite
uv run pytest -m "not requires_spacy"         # skip spaCy-dependent tests
```

`uv sync` recreates `.venv` deterministically from `uv.lock`. When you change a dependency in `pyproject.toml`, run `uv lock` (or `make lock`) to refresh the lock file and commit the result — the Dockerfile uses `uv sync --frozen` and will fail the build if `uv.lock` is stale.

---

## Learn more

GPS-trajectory aggregation, how to extend detection coverage for new entity categories, and the locked-test contract are documented separately — see [`docs/learn_more.md`](docs/learn_more.md).

---

## ⚠️ Known limitations — human verification is required

**Statistical / NLP-based anonymisation is not a guarantee.** This pipeline reduces residual-PII risk by stacking multiple imperfect detectors (Purview metadata, Presidio value sampling, spaCy embedding similarity, regex + Luhn/IBAN/IPv6 validators, rapidfuzz typo tolerance, a residual safety net) but it can — and will — let edge cases through. Categories of leak this codebase has observed at least once and that cannot be ruled out without human review:

- **Bare given names in short cells.** Single-token names like `Jimmy` or `Anna` in a `notes`-style column do not always trigger PERSON per-cell; the column-policy layer compensates for known-PII columns, but a free-text column that occasionally contains a name can still leak.
- **OOV / minority-language tokens.** Luxembourgish forms (`Lëtzebuerger`, `Kriibs`, `Prisongstrof`) and other rare surface variants are zero-vector in the spaCy models we load. The semantic recognizer falls back to rapidfuzz, but Levenshtein-distance-2 won't catch every misspelling, OCR artefact, or transliteration.
- **Language mis-routing.** `langdetect` is probabilistic on short cells. A French sentence misclassified as English (or vice-versa) routes to the wrong language's NLP pipeline; the union-seed strategy mitigates this for Art. 9 / Art. 10 detection but doesn't fix every case.
- **Custom recognizers and the `0.4` threshold.** Context-driven recognizers (NATIONAL_TAX_ID, BOOKING_REF, POSTAL_CODE, CONTRACT_NUMBER, SWIFT_BIC, …) only fire when a context keyword is nearby. A document that strips labels (e.g. a database export with header-less rows) will lose those boosts.
- **False positives.** Conversely, the embedding/fuzzy layers occasionally flag legitimate non-PII (a product description, a SKU, a hex colour). The pipeline errs on the side of masking, which can damage analytics value.
- **Categories not modelled.** Behavioural data, genetic information, location traces below `GPS_PRECISION`, and free-text quotations from real people are not exhaustively covered.

**Before publishing the anonymised output, share it with someone who knows the source data.** The `column_policy` audit entry in PostgreSQL is the right starting point — it lists every column the pipeline classified, the action taken (hash / tokenise / bin / scan), and the source tier. Any column tagged `fallback` (Tier C) deserves a manual eyeball pass; any column tagged `presidio_structured` or `embedding_similarity` should be sanity-checked against the source values for unexpected behaviour.

Human verification is the final layer of defence. The code below it is best-effort.
