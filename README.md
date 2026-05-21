# Fabric PII Anonymization Pipeline

A containerized, **stateless** Python pipeline that:

1. **Reads** a Delta table from a Microsoft Fabric Lakehouse (OneLake).
2. **Cross-checks** column sensitivity labels against Microsoft Purview (optional).
3. **Anonymizes** PII, GDPR, and financial entities using Microsoft Presidio + spaCy.
4. **Writes** the cleaned data as a Parquet file to OneLake for later Fabric processing.
5. **Audits** every run in a PostgreSQL database (run-level + per-column granularity).
6. **Records** pipeline failures in the PostgreSQL alert table.

No PySpark. No files written inside the container at runtime.

---

## Architecture

```
OneLake (source)                              OneLake (target)
abfss://…/raw_customers                       abfss://…/anonymized_customers
        │                                               │
        │  delta-rs (read)                              │  delta-rs (write)
        ▼                                               ▲
   Pandas DataFrame ──► Purview check ──► Presidio ──► Anonymized DataFrame
                                                │
                                         PostgreSQL audit DB
                                   pii_pipeline_runs  (1 row / run)
                             pii_pipeline_column_events  (1 row / column)
```

### Entities detected and masked

| Entity | Examples |
|---|---|
| `PERSON` | John Smith |
| `EMAIL_ADDRESS` | user@example.com |
| `PHONE_NUMBER` | +352 621 123 456 |
| `CREDIT_CARD` | 4111 1111 1111 1111 |
| `IBAN_CODE` | LU28 0019 4006 4475 0000 |
| `LOCATION`, `DATE_TIME`, `NRP`, and recognizer-specific entities | EU addresses, dates, nationalities, and local identifiers |

Presidio recognizers are not restricted to a US-only entity allow-list; the pipeline asks the analyzer for its available GDPR-relevant findings and replaces detections with stable pseudonym tokens such as `EMAIL_ADDRESS_0`.

---

## Prerequisites

| Tool | Version |
|---|---|
| Docker + Docker Compose | 24+ |
| Azure Service Principal | — |
| PostgreSQL | 14+ (provided by compose for local runs) |

The Service Principal (or Managed Identity) needs:

| Resource | Required role |
|---|---|
| Source OneLake | Storage Blob Data Reader |
| Target OneLake | Storage Blob Data Contributor |
| Purview account (optional) | Purview Data Reader |

---

## Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

### Required

| Variable | Description |
|---|---|
| `AZURE_TENANT_ID` | Azure AD tenant ID |
| `AZURE_CLIENT_ID` | Service principal application (client) ID |
| `AZURE_CLIENT_SECRET` | Service principal secret value |
| `DATABASE_URL` | PostgreSQL DSN for audit records and alerts (table mappings no longer required here) |

### Table mapping

Set both base URIs and the pipeline discovers all tables automatically:

```
SOURCE_BASE_ABFSS_URI=abfss://MyWorkspace@onelake.dfs.fabric.microsoft.com/Source.Lakehouse/Tables/raw
TARGET_BASE_ABFSS_URI=abfss://MyWorkspace@onelake.dfs.fabric.microsoft.com/Target.Lakehouse/Files/anonymized
```

Every Delta table directory found directly under `SOURCE_BASE_ABFSS_URI` gets an anonymized copy written under the same name at `TARGET_BASE_ABFSS_URI`. `Tables/raw/customers` → `Files/anonymized/customers`, and so on for every table, with no per-table configuration required.

### Optional

| Variable | Default | Description |
|---|---|---|
| `PURVIEW_ACCOUNT_NAME` | *(disabled)* | Purview account name for label cross-check |
| `K_ANONYMITY_MIN` | `5` | Minimum group size for quasi-identifier combinations |
| `HASH_SALT` | empty | Salt for deterministic identifier hashes |

### OneLake ABFS URI format

```
abfss://<WorkspaceName>@onelake.dfs.fabric.microsoft.com/<LakehouseName>.Lakehouse/<Tables|Files>/<path>
```

Copy from Fabric portal: open the Lakehouse → right-click the table → **Properties** → **ABFS path**.

---

## Running locally with Docker Compose

The compose file spins up a Postgres instance and the pipeline in one command.

```bash
cp .env.example .env
# Fill in AZURE_*, SOURCE_BASE_ABFSS_URI, and TARGET_BASE_ABFSS_URI

docker compose up --build
```

`DATABASE_URL` is automatically wired to the compose-managed Postgres — no
manual configuration needed for the audit database when running locally.

To inspect audit records after the run:

```bash
docker compose exec db psql -U pipeline -d pii_audit

pii_audit=# SELECT run_id, status, total_rows, entities_total, finished_at FROM pii_pipeline_runs;
pii_audit=# SELECT column_name, detections, entity_counts FROM pii_pipeline_column_events WHERE run_id = '<run_id>';
```

To stop and remove containers (Postgres data is kept in the `pg_data` volume):

```bash
docker compose down
```

---

## Running standalone (without compose)

Build the image:

```bash
docker build -t fabric-pii-pipeline:latest .
```

Verify it runs as a non-root user:

```bash
docker run --rm --entrypoint whoami fabric-pii-pipeline:latest
# → appuser
```

Run against an external Postgres:

```bash
docker run --rm \
  --env-file .env \
  -e DATABASE_URL="postgresql://user:pass@your-pg-host:5432/pii_audit" \
  fabric-pii-pipeline:latest
```

---

## PostgreSQL audit schema

Tables are created automatically on first run (`CREATE TABLE IF NOT EXISTS`).

### `pii_pipeline_runs`  — one row per execution

| Column | Type | Description |
|---|---|---|
| `run_id` | UUID PK | Unique execution identifier |
| `pipeline_version` | TEXT | Code version |
| `started_at` | TIMESTAMPTZ | Pipeline start time |
| `finished_at` | TIMESTAMPTZ | Pipeline end time (NULL while running) |
| `source_uri` | TEXT | Source ABFS path |
| `target_uri` | TEXT | Target ABFS path |
| `total_rows` | INTEGER | Rows read from source |
| `total_columns` | INTEGER | Total columns in source table |
| `columns_scanned` | INTEGER | String columns passed to Presidio |
| `columns_hit` | JSONB | Column names where entities were found |
| `entities_total` | INTEGER | Total entity detections across all columns |
| `entity_counts` | JSONB | `{"PERSON": 4, "EMAIL_ADDRESS": 2, …}` |
| `purview_ok` | BOOLEAN | Whether Purview was reachable |
| `purview_flagged` | JSONB | Columns Purview marked sensitive |
| `purview_diffs` | JSONB | Purview-flagged columns absent from DataFrame |
| `status` | TEXT | `running` → `success` or `failure` |
| `error_msg` | TEXT | Exception detail when status = failure |
| `created_at` | TIMESTAMPTZ | Row insert time |

### `pii_pipeline_column_events`  — one row per column per execution

| Column | Type | Description |
|---|---|---|
| `id` | BIGSERIAL PK | — |
| `run_id` | UUID FK | References `pii_pipeline_runs.run_id` |
| `column_name` | TEXT | Column that was scanned |
| `detections` | INTEGER | Entity hits found in this column |
| `entity_counts` | JSONB | Per-entity breakdown for this column |
| `processed_at` | TIMESTAMPTZ | When the column was scanned |

---

## Pushing to Docker Hub

```bash
# One-time login
docker login

export DOCKER_HUB_USERNAME=myusername

# Push as :latest
./push_to_dockerhub.sh

# Push a versioned tag (also updates :latest)
./push_to_dockerhub.sh 1.2.0
```

The script always builds for `linux/amd64` so the image works on both local
Apple Silicon machines and cloud-hosted amd64 runners.

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

## Microsoft Purview setup

When `PURVIEW_ACCOUNT_NAME` is set the pipeline queries the Purview Atlas API
for column-level sensitivity classifications on the source table and logs any
columns Purview flagged that Presidio didn't detect. The check is
**non-blocking** — a 404 or network error logs a warning and continues.

The service principal needs the **Purview Data Reader** role on the account.

---

## Alert records

Pipeline failures are persisted in PostgreSQL instead of sent to webhooks.
Read `pii_pipeline_alerts` for operational alerting:

```sql
SELECT run_id, table_name, subject, body, created_at
FROM pii_pipeline_alerts
ORDER BY created_at DESC;
```

---

## Running on Azure-managed compute

On ACI / AKS / Azure ML with an assigned Managed Identity, omit the three
`AZURE_*` variables. `DefaultAzureCredential` acquires tokens automatically
via the IMDS endpoint.

---

## Security notes

* The container runs as **non-root** (uid/gid 1001, `appuser`).
* **No files are written at runtime** — the container is fully stateless.
* Credentials arrive via environment variables, never baked into the image.
* `.env` is git-ignored; never commit it.
* Token scope is minimal: `https://storage.azure.com/.default` for OneLake,
  `https://purview.azure.net/.default` for Purview.
* The `en_core_web_lg`, `fr_core_news_lg`, and `de_core_news_lg` models are embedded at build time; no outbound NLP calls at runtime. Luxembourgish (lb) text is handled by the German model, as no dedicated spaCy model exists for that language.

---

## Project structure

```
.
├── main.py                # Pipeline orchestration
├── requirements.txt       # Python dependencies
├── Dockerfile             # Stateless container (non-root, no volumes)
├── docker-compose.yml     # Local dev: pipeline + Postgres
├── push_to_dockerhub.sh   # Build & push to Docker Hub
├── .env.example           # Environment variable template
└── README.md
```
