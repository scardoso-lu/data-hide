import { getConfig } from "@/lib/queries/config"
import ConfigTable from "./config-table"

export const dynamic = "force-dynamic"

export default async function ConfigPage() {
  const entries = await getConfig()

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold">Pipeline Configuration</h1>
        <p className="text-base-content/60 text-sm mt-1">
          Runtime parameters stored in <code className="font-mono text-xs bg-base-200 px-1 rounded">pii_pipeline_config</code>.
          Changes take effect on the next pipeline run.
        </p>
      </div>

      <div className="card bg-base-100 shadow p-6">
        <ConfigTable entries={entries} />
      </div>

      {/* Reference for known parameters */}
      <div className="collapse collapse-arrow bg-base-100 shadow">
        <input type="checkbox" />
        <div className="collapse-title font-medium text-sm">
          Known parameter reference
        </div>
        <div className="collapse-content">
          <div className="overflow-x-auto">
            <table className="table table-xs">
              <thead>
                <tr>
                  <th>Key</th>
                  <th>Default</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {KNOWN_PARAMS.map((p) => (
                  <tr key={p.key}>
                    <td className="font-mono text-xs">{p.key}</td>
                    <td className="font-mono text-xs opacity-70">{p.default}</td>
                    <td className="text-xs">{p.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

const KNOWN_PARAMS = [
  { key: "K_ANONYMITY_MIN",          default: "5",   desc: "Minimum group size for k-anonymity suppression." },
  { key: "K_ANONYMITY_TABLES",       default: "",    desc: "Comma-separated list of tables where k-anonymity runs." },
  { key: "PURVIEW_ACCOUNT_NAME",     default: "",    desc: "Azure Purview account name for column classification." },
  { key: "ENABLE_KEY_VAULT",         default: "1",   desc: "Set to 0 to disable Key Vault and fall back to local HMAC hashing." },
  { key: "GPS_PRECISION",            default: "1",   desc: "Decimal places to keep after GPS coordinate rounding." },
  { key: "MAX_TABLE_WORKERS",        default: "1",   desc: "Parallel worker processes for multi-table runs." },
  { key: "IDENTIFIER_COLS",          default: "",    desc: "Comma-separated column names to pseudonymize across all tables." },
  { key: "SQL_ENDPOINT_URL",         default: "",    desc: "Fabric SQL endpoint for shortcut-based table discovery." },
  { key: "SQL_DATABASE",             default: "",    desc: "Fabric SQL database name for shortcut discovery." },
  { key: "ENABLE_PRESIDIO_STRUCTURED", default: "1", desc: "Enable Presidio structured-value sampling in column classification." },
  { key: "ENABLE_COLUMN_POLICY",     default: "1",   desc: "Enable the column-policy classification layer." },
  { key: "COLUMN_SIMILARITY_THRESHOLD", default: "0.55", desc: "spaCy embedding similarity threshold for column-name classification." },
]
