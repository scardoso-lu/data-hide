# Fabric PII Anonymization Pipeline

A containerized Python pipeline that:

1. **Reads** a Delta table from a Microsoft Fabric Lakehouse (OneLake).
2. **Cross-checks** column sensitivity labels against Microsoft Purview (optional).
3. **Anonymizes** PII, GDPR, and financial entities using Microsoft Presidio + spaCy.
4. **Writes** the cleaned data as a Delta table to a different Microsoft Fabric Lakehouse.
5. **Audits** every run in a central Delta audit table and local JSONL files.
6. **Alerts** on pipeline failures via a configurable Teams / Slack webhook.

No PySpark — everything runs on lightweight `delta-rs` + Pandas.

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
                                        Audit Delta table + JSONL logs
```

### Entities detected and masked

| Entity | Examples masked |
|---|---|
| `PERSON` | John Smith |
| `EMAIL_ADDRESS` | user@example.com |
| `PHONE_NUMBER` | +1-800-555-0100 |
| `CREDIT_CARD` | 4111 1111 1111 1111 |
| `IBAN_CODE` | GB29 NWBK 6016 1331 9268 19 |
| `US_BANK_NUMBER` | 123456789 |

All detections are replaced with `***`.

---

## Prerequisites

| Tool | Version |
|---|---|
| Docker | 20.10+ |
| Azure Service Principal | — |

The Service Principal (or Managed Identity) needs:

| Resource | Required role |
|---|---|
| Source OneLake | Storage Blob Data Reader |
| Target OneLake | Storage Blob Data Contributor |
| Audit OneLake | Storage Blob Data Contributor |
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
| `SOURCE_ABFSS_URI` | Full `abfss://` path to the source Delta table |
| `TARGET_ABFSS_URI` | Full `abfss://` path to the target Delta table |

### Optional

| Variable | Default | Description |
|---|---|---|
| `AUDIT_ABFSS_URI` | *(disabled)* | Delta table for the central run-history audit log |
| `PURVIEW_ACCOUNT_NAME` | *(disabled)* | Purview account name for sensitivity-label cross-check |
| `ALERT_WEBHOOK_URL` | *(disabled)* | Teams or Slack webhook URL for failure alerts |
| `LOG_DIR` | `/app/logs` | Directory for `pipeline.log` and per-run `audit_<id>.jsonl` |

### OneLake ABFS URI format

```
abfss://<WorkspaceName>@onelake.dfs.fabric.microsoft.com/<LakehouseName>.Lakehouse/Tables/<TableName>
```

You can copy the exact URI from the Fabric portal: open the Lakehouse → right-click the table → **Properties** → **ABFS path**.

---

## Audit and observability

### Local log files (always written)

| File | Content |
|---|---|
| `$LOG_DIR/pipeline.log` | Human-readable rolling log (all runs) |
| `$LOG_DIR/audit_<run_id>.jsonl` | Structured JSONL event stream for this run |

Each JSONL file contains events:

| Event | When |
|---|---|
| `pipeline_start` | Pipeline begins |
| `purview_check` | Purview result (if enabled) |
| `column_processed` | After each text column is scanned |
| `pipeline_success` | Final summary on success |
| `pipeline_failure` | Error detail on failure |

### Central audit Delta table (`AUDIT_ABFSS_URI`)

One row per run.  Schema:

| Column | Type | Description |
|---|---|---|
| `run_id` | STRING | UUID for this execution |
| `pipeline_version` | STRING | Code version |
| `pipeline_start_ts` | STRING | ISO 8601 start time |
| `pipeline_end_ts` | STRING | ISO 8601 end time |
| `source_uri` | STRING | Source ABFS path |
| `target_uri` | STRING | Target ABFS path |
| `total_rows_processed` | INT64 | Rows read |
| `total_columns_in_table` | INT64 | Total columns |
| `total_columns_scanned` | INT64 | String columns scanned |
| `columns_anonymized` | STRING (JSON) | Names of columns that had detections |
| `total_entities_detected` | INT64 | Sum of all entity hits |
| `entity_counts` | STRING (JSON) | `{"PERSON": 4, "EMAIL_ADDRESS": 2, …}` |
| `purview_available` | BOOL | Whether Purview was reachable |
| `purview_flagged_columns` | STRING (JSON) | Columns Purview marked sensitive |
| `purview_discrepancies` | STRING (JSON) | Purview-flagged cols absent from DataFrame |
| `status` | STRING | `"success"` or `"failure"` |
| `error_message` | STRING | Error detail if status=failure |

---

## Build & run with Docker

### Build the image

```bash
docker build -t fabric-pii-pipeline:latest .
```

> The `en_core_web_lg` spaCy model (~800 MB) is baked into the image at build time — no network required at runtime.

### Verify the image runs as a non-root user

```bash
docker run --rm --entrypoint whoami fabric-pii-pipeline:latest
# expected output: appuser
```

### Run — mount a log volume for persistence

```bash
docker run --rm \
  --env-file .env \
  -v "$(pwd)/logs:/app/logs" \
  fabric-pii-pipeline:latest
```

Passing variables individually:

```bash
docker run --rm \
  -e AZURE_TENANT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  -e AZURE_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  -e AZURE_CLIENT_SECRET=your-secret \
  -e SOURCE_ABFSS_URI="abfss://..." \
  -e TARGET_ABFSS_URI="abfss://..." \
  -e AUDIT_ABFSS_URI="abfss://..." \
  -e PURVIEW_ACCOUNT_NAME="my-purview-account" \
  -e ALERT_WEBHOOK_URL="https://outlook.office.com/webhook/..." \
  -v "$(pwd)/logs:/app/logs" \
  fabric-pii-pipeline:latest
```

---

## Run locally (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_lg

# Export env vars
export $(grep -v '^#' .env | xargs)

python main.py
```

---

## Microsoft Purview setup

When `PURVIEW_ACCOUNT_NAME` is set, the pipeline:

1. Translates the source ABFS URI to a Purview Atlas qualified name.
2. Queries the Purview Catalog API for column-level sensitivity classifications.
3. Logs any columns Purview flagged that are not also detected by Presidio (discrepancies).
4. Records results in the JSONL audit file and the central audit table.

The check is **non-blocking** — if Purview is unreachable or the asset is not yet catalogued the pipeline continues and logs a warning.

The service principal requires the **Purview Data Reader** (or **Collection Admin**) role on the Purview account.

---

## Alert webhook

Set `ALERT_WEBHOOK_URL` to receive a JSON `POST` on every pipeline failure.

**Microsoft Teams** — create an Incoming Webhook connector in a channel and paste the URL.

**Slack** — add the **Incoming Webhooks** app to your workspace and paste the webhook URL.

The payload shape `{"text": "..."}` is accepted by both platforms.

---

## Running on Azure-managed compute

When the container runs on Azure Container Instances, AKS, or Azure ML with an assigned **Managed Identity**, omit all three `AZURE_*` variables. `DefaultAzureCredential` automatically acquires tokens via the IMDS endpoint.

---

## Security notes

* The Docker container runs as **non-root** (uid/gid 1001, `appuser`).
* Credentials arrive at runtime via environment variables — never baked into the image.
* `.env` is git-ignored; never commit it.
* The OAuth2 scope is `https://storage.azure.com/.default` (OneLake only) and `https://purview.azure.net/.default` (Purview only) — minimal privilege.
* The `en_core_web_lg` model is embedded in the image; no outbound NLP API calls are made at runtime.

---

## Project structure

```
.
├── main.py            # Pipeline orchestration
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container definition (non-root, self-contained)
├── .env.example       # Environment variable template
└── README.md
```
