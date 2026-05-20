# Fabric PII Anonymization Pipeline

A containerized Python pipeline that:

1. **Reads** a Delta table from a Microsoft Fabric Lakehouse (OneLake).
2. **Anonymizes** PII, GDPR, and financial entities using [Microsoft Presidio](https://microsoft.github.io/presidio/).
3. **Writes** the cleaned data as a Delta table to a different Microsoft Fabric Lakehouse.

No PySpark required — everything runs on lightweight `delta-rs` + Pandas.

---

## Architecture

```
OneLake (source)                     OneLake (target)
abfss://…/raw_customers              abfss://…/anonymized_customers
        │                                       │
        │  delta-rs (read)                      │  delta-rs (write)
        ▼                                       ▲
   Pandas DataFrame  ──►  Presidio  ──►  Anonymized DataFrame
```

### Detected & masked entities

| Entity | Examples |
|---|---|
| `PERSON` | John Smith, María García |
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
| Azure CLI / Service Principal | — |

The Service Principal needs the **Storage Blob Data Contributor** role (or equivalent) on both the source and target Fabric OneLake.

---

## Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `AZURE_TENANT_ID` | Azure AD tenant ID |
| `AZURE_CLIENT_ID` | Service principal application (client) ID |
| `AZURE_CLIENT_SECRET` | Service principal secret value |
| `SOURCE_ABFSS_URI` | Full `abfss://` path to the source Delta table |
| `TARGET_ABFSS_URI` | Full `abfss://` path to the target Delta table (created if absent) |

### OneLake ABFS URI format

```
abfss://<WorkspaceName>@onelake.dfs.fabric.microsoft.com/<LakehouseName>.Lakehouse/Tables/<TableName>
```

**Example:**

```
abfss://DataEngineering@onelake.dfs.fabric.microsoft.com/SalesLakehouse.Lakehouse/Tables/orders
```

You can copy the exact URI from Fabric portal: open your Lakehouse → right-click a table → **Properties** → **ABFS path**.

---

## Build & run with Docker

### Build the image

```bash
docker build -t fabric-pii-pipeline:latest .
```

> The build downloads the `en_core_web_lg` spaCy model (~800 MB) into the image layer so no network is needed at runtime.

### Run the container

```bash
docker run --rm --env-file .env fabric-pii-pipeline:latest
```

Or pass variables individually:

```bash
docker run --rm \
  -e AZURE_TENANT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  -e AZURE_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  -e AZURE_CLIENT_SECRET=your-secret \
  -e SOURCE_ABFSS_URI="abfss://..." \
  -e TARGET_ABFSS_URI="abfss://..." \
  fabric-pii-pipeline:latest
```

---

## Run locally (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_lg

# Export env vars (or use a tool like direnv / dotenv)
export $(grep -v '^#' .env | xargs)

python main.py
```

---

## Running on Azure-hosted compute

When the container runs inside an Azure resource that has a **Managed Identity** assigned (e.g. Azure Container Instances, AKS, Azure ML), you can omit the three service-principal variables entirely. `DefaultAzureCredential` will automatically use the Managed Identity.

---

## Security notes

* The Docker image runs as a non-root user (`appuser`).
* Credentials are consumed at runtime via environment variables — never baked into the image.
* The `.env` file is listed in `.gitignore`; never commit it.
* Token acquisition uses the minimal scope `https://storage.azure.com/.default` (read/write OneLake only).

---

## Project structure

```
.
├── main.py            # Pipeline logic
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container definition
├── .env.example       # Environment variable template
└── README.md
```
