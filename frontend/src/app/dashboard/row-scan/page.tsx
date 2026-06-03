import { getRowScanColumns } from "@/lib/queries/apply-row-scan"
import RowScanTable from "./row-scan-table"

export const dynamic = "force-dynamic"

export default async function RowScanPage() {
  const columns = await getRowScanColumns()

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold">Row-Scan Columns</h1>
        <p className="text-base-content/60 text-sm mt-1">
          When the pipeline reaches its final tier, any table listed here has the{" "}
          <strong>row-by-row Presidio scan</strong> applied to exactly its listed
          columns — each non-null cell is scanned for PII. Only these columns are
          scanned, which avoids the false positives of scanning every text column.
          Stored in{" "}
          <code className="font-mono text-xs bg-base-200 px-1 rounded">
            apply_row_scan
          </code>
          . Changes take effect on the next pipeline run.
        </p>
      </div>

      <div className="card bg-base-100 shadow p-6">
        <RowScanTable columns={columns} />
      </div>
    </div>
  )
}
