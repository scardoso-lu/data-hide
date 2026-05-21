# Fabric PII Anonymization Pipeline

A containerized, **stateless** Python pipeline that discovers every Delta table under a OneLake base path, anonymizes all personal data found in text columns using Presidio + spaCy NLP, and writes clean Parquet files back to OneLake — with a full PostgreSQL audit trail on every run.

No PySpark. No files written inside the container at runtime.

---

## How it works

```mermaid
flowchart TD
    A["OneLake — Delta tables\n(SOURCE_BASE_ABFSS_URI)"]
    A2["SQL Analytics Endpoint\n(SQL_ENDPOINT_URL · optional)"]
    B["Pandas DataFrame\nper discovered table"]
    C["Dynamic Column Classification\ndetect_identifier_columns\ndetect_quasi_identifiers\nflag_free_text_columns"]
    D["Hash & k-Anonymity\nhash_identifier_columns\nenforce_k_anonymity"]
    E["Presidio + spaCy NLP\nanonymize_dataframe\n(EN · FR · DE · LB)"]
    F["Purview Sensitivity Check\nrun_purview_check\n(optional)"]
    G["Residual PII Validation\nvalidate_residual_pii"]
    H["OneLake — Parquet output\n(TARGET_BASE_ABFSS_URI)"]
    I["Pipeline aborts\ntarget not written"]
    J["PostgreSQL Audit DB\nAuditDB · open_run / close_run"]

    A -->|"discover_table_mappings\nread_mode=delta"| B
    A2 -.->|"shortcuts not in ADLS\nread_mode=sql"| B
    B --> C
    C --> D
    D --> E
    B -.->|optional| F
    F -.-> E
    E --> G
    G -->|clean| H
    G -->|PII found| I
    E --> J
```

---

## GDPR coverage

Each GDPR obligation is traced to the file and function that implements it.

| GDPR Article | Obligation | Implementation | File | Function |
|---|---|---|---|---|
| Art. 4(1) — Personal data | Identify and protect all natural-person identifiers | NLP entity detection across text columns | `app/anonymization.py` | `_analyze()` |
| Art. 5(1)(b) — Purpose limitation | Write anonymized data to a separate target; never overwrite source | Source ≠ target URI guard before any write | `app/service.py` | `run_table()` |
| Art. 5(1)(c) — Data minimisation | Remove direct identifiers | Auto-detect and SHA-256 hash identifier columns | `app/classification.py` · `app/anonymization.py` | `detect_identifier_columns()` · `hash_identifier_columns()` |
| Art. 5(1)(c) — Data minimisation | Reduce GPS precision | Round lat/lon coordinates to N decimal places (default 2 d.p. ≈ 1 km) | `app/classification.py` · `app/anonymization.py` | `detect_gps_columns()` · `anonymize_gps_columns()` |
| Art. 5(1)(c) — Data minimisation | Suppress rare quasi-identifier combinations | k-anonymity: groups smaller than `K_ANONYMITY_MIN` are dropped | `app/anonymization.py` | `enforce_k_anonymity()` |
| Art. 5(1)(d) — Accuracy | Guarantee no residual PII in output | Scan every text cell after anonymization; abort if anything remains | `app/anonymization.py` | `validate_residual_pii()` |
| Art. 5(1)(f) — Integrity & confidentiality | Cryptographically protect identifiers | Salted SHA-256 hash; salt stored only in env variable | `app/anonymization.py` | `hash_identifier_columns()` |
| Art. 25 — Data protection by design | Pseudonymise by default | Replace PII spans with stable tokens (`PERSON_0`, `EMAIL_ADDRESS_1`, …) | `app/anonymization.py` | `anonymize_dataframe()` |
| Art. 30 — Records of processing activities | Log every processing activity | One PostgreSQL row per run and per column with entity counts | `app/repository.py` | `AuditDB.open_run()` · `AuditDB.close_run()` · `AuditDB.record_columns()` |
| Art. 32 — Security of processing | Appropriate technical and organisational measures | Non-root container, no runtime file writes, ABFSS TLS, salted hashes | `Dockerfile` · `app/anonymization.py` | — |
| Art. 35 — Data protection impact assessment | Document processing risks | Per-run audit records include entity counts, suppressed rows, residual PII count, and Purview discrepancies | `app/repository.py` | `AuditDB.close_run()` |

---

## Dynamic field discovery and GDPR entity detection

Column classification and PII field discovery are **fully automatic** — no schema mapping, allow-lists, or per-table configuration is required.

### How columns are classified (`app/classification.py`)

| Classifier | Function | What it detects |
|---|---|---|
| Identifier columns | `detect_identifier_columns()` | High-cardinality columns whose names or values match known ID patterns (e.g. `user_id`, `ssn`, `nrn`) — hashed before NLP scanning |
| Quasi-identifier columns | `detect_quasi_identifiers()` | Columns that are individually innocuous but combinable to re-identify individuals (e.g. `age`, `gender`, `postcode`) — subject to k-anonymity |
| Free-text columns | `flag_free_text_columns()` | Long-form string columns sent to Presidio for full NLP scanning |
| Column name sanitization | `sanitize_column_names()` | Renames unsafe characters before processing; renames are logged in the audit record |

### What Presidio + spaCy detect (`app/anonymization.py`)

Language is detected automatically per cell using `langdetect`. Models loaded: **English** (`en_core_web_lg`), **French** (`fr_core_news_lg`), **German** (`de_core_news_lg`), **Luxembourgish** (via the German model — no dedicated spaCy model exists for `lb`).

| GDPR Personal Data Category | Presidio Entity | Detection method |
|---|---|---|
| Full name / natural person (Art. 4(1)) | `PERSON` | spaCy NER — all language models |
| Electronic contact — email | `EMAIL_ADDRESS` | Rule-based recognizer |
| Electronic contact — phone (incl. `+352`) | `PHONE_NUMBER` | Rule-based recognizer |
| Financial — payment card | `CREDIT_CARD` | Rule-based recognizer (Luhn validation) |
| Financial — bank account | `IBAN_CODE` | Rule-based recognizer |
| Financial — routing / account number | `US_BANK_NUMBER` | Rule-based recognizer |
| Geographic location | `LOCATION` | spaCy NER — all language models |
| Date that could identify an individual | `DATE_TIME` | spaCy NER — all language models |
| Nationality / religion / political opinion (Art. 9) | `NRP` | spaCy NER — all language models |
| Network identifier | `IP_ADDRESS` | Rule-based recognizer |

All detected spans are replaced with stable pseudonym tokens (`ENTITY_TYPE_N`) that are consistent within a run, so joins across anonymized tables remain valid.

---

## Table discovery: Delta + SQL shortcuts

`discover_table_mappings` uses two complementary strategies so every table — whether a native Delta table or a Fabric **shortcut** — is included.

| Strategy | Trigger | `read_mode` | How it reads |
|---|---|---|---|
| ADLS Delta scan | Always | `delta` | Scans `SOURCE_BASE_ABFSS_URI` via `DataLakeServiceClient`; only directories with a `_delta_log` subdirectory are included (`delta-rs`) |
| SQL Analytics Endpoint | `SQL_ENDPOINT_URL` + `SQL_DATABASE` set | `sql` | Queries `INFORMATION_SCHEMA.TABLES` on the Fabric SQL endpoint; tables already found by the ADLS scan are skipped to avoid duplicates |

Shortcuts in Fabric OneLake are exposed via the SQL Analytics Endpoint but have no `_delta_log` directory, so ADLS discovery misses them. Setting `SQL_ENDPOINT_URL` and `SQL_DATABASE` fills that gap automatically.

Authentication reuses `DefaultAzureCredential` with the `https://database.windows.net/.default` scope and passes the bearer token to **ODBC Driver 18 for SQL Server** via `SQL_COPT_SS_ACCESS_TOKEN` — no username/password required.

---

## Installation with Docker

### Prerequisites

| Tool | Version |
|---|---|
| Docker + Docker Compose | 24+ |
| Azure Service Principal or Managed Identity | — |
| PostgreSQL | 14+ (provided by Compose for local runs) |

The Service Principal needs:

| Resource | Required role |
|---|---|
| Source OneLake | Storage Blob Data Reader |
| Target OneLake | Storage Blob Data Contributor |
| Purview account (optional) | Purview Data Reader |

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
| `TARGET_BASE_ABFSS_URI` | Yes | Base path where anonymized Parquet files are written |
| `PURVIEW_ACCOUNT_NAME` | No | Purview account name for sensitivity-label cross-check |
| `K_ANONYMITY_MIN` | No (default `5`) | Minimum group size for quasi-identifier combinations |
| `HASH_SALT` | No | Salt mixed into SHA-256 identifier hashes |
| `SQL_ENDPOINT_URL` | No | Fabric SQL Analytics Endpoint hostname — enables shortcut discovery |
| `SQL_DATABASE` | No | Database name on the SQL endpoint (typically the Lakehouse name) |
| `GPS_PRECISION` | No (default `2`) | Decimal places for spatial rounding of GPS columns (2 ≈ 1 km, 3 ≈ 110 m) |

OneLake URI format:
```
abfss://<WorkspaceName>@onelake.dfs.fabric.microsoft.com/<LakehouseName>.Lakehouse/<Tables|Files>/<path>
```
Copy from Fabric portal: open the Lakehouse → right-click the table → **Properties** → **ABFS path**.

### 2 — Run with Docker Compose (recommended for local use)

Compose starts a managed Postgres instance alongside the pipeline. `DATABASE_URL` is pre-wired — no extra configuration needed.

```bash
docker compose up --build
```

Inspect audit records after the run:

```bash
docker compose exec db psql -U pipeline -d pii_audit

pii_audit=# SELECT run_id, table_name, status, entities_total, finished_at FROM pii_pipeline_runs;
pii_audit=# SELECT column_name, detections, entity_counts FROM pii_pipeline_column_events WHERE run_id = '<run_id>';
```

Stop and remove containers (Postgres data is kept in the `pg_data` volume):

```bash
docker compose down
```

### 3 — Run as a standalone container (against an external Postgres)

Build the image:

```bash
docker build -t fabric-pii-pipeline:latest .
```

Verify the non-root user:

```bash
docker run --rm --entrypoint whoami fabric-pii-pipeline:latest
# → appuser
```

Run against your environment:

```bash
docker run --rm \
  --env-file .env \
  -e DATABASE_URL="postgresql://user:pass@your-pg-host:5432/pii_audit" \
  fabric-pii-pipeline:latest
```

### 4 — Push to Docker Hub

```bash
docker login
export DOCKER_HUB_USERNAME=myusername

./push_to_dockerhub.sh          # tags as :latest
./push_to_dockerhub.sh 1.2.0   # also tags a versioned release
```

The script builds for `linux/amd64` so the image runs on both Apple Silicon and cloud-hosted amd64 runners.

---

## Run locally without Docker

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_lg
python -m spacy download fr_core_news_lg
python -m spacy download de_core_news_lg

export $(grep -v '^#' .env | xargs)
python main.py
```

---

## PostgreSQL audit schema

Tables are created automatically on first run (`CREATE TABLE IF NOT EXISTS`).

### `pii_pipeline_runs` — one row per execution

| Column | Type | Description |
|---|---|---|
| `run_id` | UUID PK | Unique execution identifier |
| `pipeline_version` | TEXT | Code version |
| `started_at` | TIMESTAMPTZ | Pipeline start time |
| `finished_at` | TIMESTAMPTZ | Pipeline end time (NULL while running) |
| `table_name` | TEXT | Logical table name from the mapping |
| `source_uri` | TEXT | Source ABFS path |
| `target_uri` | TEXT | Target ABFS path |
| `total_rows` | INTEGER | Rows read from source |
| `total_columns` | INTEGER | Total columns in source table |
| `columns_scanned` | INTEGER | String columns passed to Presidio |
| `columns_hit` | JSONB | Column names where entities were found |
| `entities_total` | INTEGER | Total entity detections across all columns |
| `entity_counts` | JSONB | `{"PERSON": 4, "EMAIL_ADDRESS": 2, …}` |
| `suppressed_rows` | INTEGER | Rows dropped by k-anonymity |
| `residual_pii` | INTEGER | PII findings after anonymization (0 on success) |
| `purview_ok` | BOOLEAN | Whether Purview was reachable |
| `purview_flagged` | JSONB | Columns Purview marked sensitive |
| `purview_diffs` | JSONB | Purview-flagged columns absent from DataFrame |
| `status` | TEXT | `running` → `success` or `failure` |
| `error_msg` | TEXT | Exception detail when status = failure |
| `created_at` | TIMESTAMPTZ | Row insert time |

### `pii_pipeline_column_events` — one row per column per execution

| Column | Type | Description |
|---|---|---|
| `id` | BIGSERIAL PK | — |
| `run_id` | UUID FK | References `pii_pipeline_runs.run_id` |
| `column_name` | TEXT | Column that was scanned |
| `detections` | INTEGER | Entity hits found in this column |
| `entity_counts` | JSONB | Per-entity breakdown for this column |
| `processed_at` | TIMESTAMPTZ | When the column was scanned |

---

## Microsoft Purview setup

When `PURVIEW_ACCOUNT_NAME` is set the pipeline queries the Purview Atlas API for column-level sensitivity classifications on the source table and logs any columns Purview flagged that Presidio did not detect. The check is **non-blocking** — a 404 or network error logs a warning and continues.

The service principal needs the **Purview Data Reader** role on the account.

---

## Running on Azure-managed compute

On ACI / AKS / Azure ML with an assigned Managed Identity, omit the three `AZURE_*` variables. `DefaultAzureCredential` acquires tokens automatically via the IMDS endpoint.

---

## Security notes

* The container runs as **non-root** (uid/gid 1001, `appuser`).
* **No files are written at runtime** — the container is fully stateless.
* Credentials arrive via environment variables, never baked into the image.
* `.env` is git-ignored; never commit it.
* Token scope is minimal: `https://storage.azure.com/.default` for OneLake, `https://purview.azure.net/.default` for Purview.
* The `en_core_web_lg`, `fr_core_news_lg`, and `de_core_news_lg` models are embedded at build time; no outbound NLP calls at runtime. Luxembourgish (`lb`) text is handled by the German model, as no dedicated spaCy model exists for that language.

---

## Project structure

```
.
├── app/
│   ├── main.py            # Entrypoint — wires injectable dependencies
│   ├── service.py         # Orchestration: discover → classify → anonymize → write
│   ├── anonymization.py   # Presidio engine, NLP anonymization, k-anonymity, hashing
│   ├── classification.py  # Dynamic column classifier (identifier / quasi / free-text)
│   └── repository.py      # ADLS/Delta I/O, Purview client, AuditDB, table discovery
├── tests/
│   ├── test_pipeline.py   # Orchestration integration tests
│   ├── test_anonymization.py
│   ├── test_audit_db.py
│   ├── test_alerts.py
│   └── test_helpers.py
├── requirements.txt
├── Dockerfile             # Stateless container (non-root, no volumes)
├── docker-compose.yml     # Local dev: pipeline + Postgres
└── .env.example           # Environment variable template
```
