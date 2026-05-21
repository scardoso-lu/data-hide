# Fabric PII Anonymization Pipeline

A containerized, **stateless** Python pipeline that discovers every Delta table under a OneLake base path, anonymizes all personal data found in text columns using Presidio + spaCy NLP, and writes clean output back to OneLake — with a full PostgreSQL audit trail on every run.

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

| GDPR Article | Obligation | Implementation |
|---|---|---|
| Art. 4(1) — Personal data | Identify and protect all natural-person identifiers | NLP entity detection across text columns |
| Art. 5(1)(b) — Purpose limitation | Write anonymized data to a separate target; never overwrite source | Source ≠ target URI guard before any write |
| Art. 5(1)(c) — Data minimisation | Remove direct identifiers | Auto-detect and SHA-256 hash identifier columns |
| Art. 5(1)(c) — Data minimisation | Reduce GPS precision | Round lat/lon to N decimal places; floor co-located timestamps to day |
| Art. 5(1)(c) — Data minimisation | Suppress rare quasi-identifier combinations | k-anonymity: groups smaller than `K_ANONYMITY_MIN` are dropped |
| Art. 5(1)(d) — Accuracy | Guarantee no residual PII in output | Scan every text cell after anonymization; abort if anything remains |
| Art. 5(1)(f) — Integrity & confidentiality | Cryptographically protect identifiers | Salted SHA-256 hash; salt stored only in env variable |
| Art. 25 — Data protection by design | Pseudonymise by default | Replace PII spans with stable tokens (`PERSON_0`, `EMAIL_ADDRESS_1`, …) |
| Art. 30 — Records of processing activities | Log every processing activity | One PostgreSQL row per run and per column with entity counts |
| Art. 32 — Security of processing | Appropriate technical and organisational measures | Non-root container, no runtime file writes, ABFSS TLS, salted hashes |
| Art. 35 — Data protection impact assessment | Document processing risks | Per-run audit records include entity counts, suppressed rows, and residual PII count |

For full details on each GDPR article see [gdpr-info.eu](https://gdpr-info.eu).

---

## GPS trajectory data

Tables that contain GPS coordinates, a speed column, and a timestamp column are treated as **trajectory data** and follow a separate path:

1. Addresses are NLP-anonymized (names, locations stripped from free-text columns).
2. Individual rows are **aggregated** into `(grid cell × hour of day × day of week)` speed statistics — no vehicle identifiers or raw timestamps survive.
3. Cells with fewer than `K_ANONYMITY_MIN` pings are suppressed.

The resulting output contains only `avg_speed_kmh`, `p50_speed_kmh`, `p85_speed_kmh`, and `ping_count` per cell/time slot — safe for business analytics and external LLM consumption.

Non-trajectory GPS tables (no speed column) have coordinates rounded to `GPS_PRECISION` decimal places (default `1` ≈ 11 km for city data) and timestamps floored to day before row-level k-anonymity is applied.

---

## Installation

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
| `TARGET_BASE_ABFSS_URI` | Yes | Base path where anonymized output is written |
| `K_ANONYMITY_MIN` | No (default `5`) | Minimum group size for quasi-identifier suppression |
| `HASH_SALT` | No | Salt mixed into SHA-256 identifier hashes |
| `GPS_PRECISION` | No (default `1`) | Decimal places for GPS rounding (1 ≈ 11 km) |
| `SQL_ENDPOINT_URL` | No | Fabric SQL Analytics Endpoint — enables shortcut discovery |
| `SQL_DATABASE` | No | Database name on the SQL endpoint |

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
