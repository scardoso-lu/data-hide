import { notFound } from "next/navigation"
import Link from "next/link"
import { getRunById } from "@/lib/queries/runs"
import RunStatusBadge from "@/components/run-status-badge"

export const dynamic = "force-dynamic"

function fmtDate(iso: string | null): string {
  if (!iso) return "—"
  return new Date(iso).toLocaleString("en-GB", {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  })
}

function fmtNum(n: number | null | undefined): string {
  if (n == null) return "—"
  return n.toLocaleString()
}

interface Props {
  params: Promise<{ run_id: string }>
}

export default async function RunDetailPage({ params }: Props) {
  const { run_id } = await params
  const run = await getRunById(run_id)
  if (!run) notFound()

  const totalSec = run.stage_seconds
    ? Object.values(run.stage_seconds).reduce((a, b) => a + b, 0)
    : null

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Breadcrumb */}
      <div className="text-sm breadcrumbs">
        <ul>
          <li><Link href="/dashboard/audit">Audit Runs</Link></li>
          <li className="font-mono text-xs opacity-60">
            {run_id.slice(0, 8)}…
          </li>
        </ul>
      </div>

      {/* Header */}
      <div className="flex flex-wrap items-start gap-4">
        <div className="flex-1 min-w-0">
          <h1 className="text-2xl font-bold truncate">
            {run.table_name ?? "Unknown table"}
          </h1>
          <p className="font-mono text-xs text-base-content/50 mt-1 break-all">
            {run.source_uri}
          </p>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <RunStatusBadge status={run.status} />
          <span className="badge badge-outline badge-sm font-mono">
            v{run.pipeline_version}
          </span>
        </div>
      </div>

      {/* Error box — truncate at 500 chars to avoid leaking long stack traces */}
      {run.error_msg && (
        <div className="alert alert-error">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <div>
            <p className="font-semibold text-sm">Error</p>
            <pre className="text-xs whitespace-pre-wrap mt-1 opacity-90">
              {run.error_msg.length > 500
                ? run.error_msg.slice(0, 500) + "\n… (truncated — see pipeline logs for full detail)"
                : run.error_msg}
            </pre>
          </div>
        </div>
      )}

      {/* Summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          { label: "Rows in",       value: fmtNum(run.total_rows) },
          { label: "Suppressed",    value: fmtNum(run.suppressed_rows) },
          { label: "Residual PII",  value: fmtNum(run.residual_pii) },
          { label: "Total time",    value: totalSec != null ? `${totalSec.toFixed(2)} s` : "—" },
        ].map((s) => (
          <div key={s.label} className="stat bg-base-100 shadow rounded-box">
            <div className="stat-title text-xs">{s.label}</div>
            <div className="stat-value text-lg">{s.value}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Metadata */}
        <div className="card bg-base-100 shadow">
          <div className="card-body">
            <h2 className="card-title text-base">Metadata</h2>
            <table className="table table-xs">
              <tbody>
                <MetaRow label="Run ID"       value={<span className="font-mono text-xs">{run.run_id}</span>} />
                <MetaRow label="Started"      value={fmtDate(run.started_at)} />
                <MetaRow label="Finished"     value={fmtDate(run.finished_at)} />
                <MetaRow label="Output type"  value={run.output_type} />
                <MetaRow label="Columns"      value={`${fmtNum(run.total_columns)} total / ${fmtNum(run.columns_scanned)} scanned`} />
                <MetaRow label="Agg. cells"   value={fmtNum(run.aggregate_cells)} />
                <MetaRow label="KV key ver."  value={run.key_vault_key_version ?? "—"} />
                <MetaRow label="Purview"      value={run.purview_ok ? "✓ available" : "✗ unavailable"} />
              </tbody>
            </table>
          </div>
        </div>

        {/* Stage timings */}
        {run.stage_seconds && Object.keys(run.stage_seconds).length > 0 && (
          <div className="card bg-base-100 shadow">
            <div className="card-body">
              <h2 className="card-title text-base">Stage Timings</h2>
              <table className="table table-xs">
                <thead>
                  <tr><th>Stage</th><th className="text-right">Duration</th><th className="text-right">Share</th></tr>
                </thead>
                <tbody>
                  {Object.entries(run.stage_seconds)
                    .sort(([, a], [, b]) => b - a)
                    .map(([stage, sec]) => (
                      <tr key={stage}>
                        <td className="font-mono text-xs">{stage}</td>
                        <td className="text-right text-xs">{sec.toFixed(3)} s</td>
                        <td className="text-right text-xs opacity-60">
                          {totalSec ? `${((sec / totalSec) * 100).toFixed(1)}%` : "—"}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Entity counts */}
        {run.entity_counts && Object.keys(run.entity_counts).length > 0 && (
          <div className="card bg-base-100 shadow">
            <div className="card-body">
              <h2 className="card-title text-base">Entity Counts</h2>
              <table className="table table-xs">
                <thead>
                  <tr><th>Entity type</th><th className="text-right">Count</th></tr>
                </thead>
                <tbody>
                  {Object.entries(run.entity_counts)
                    .sort(([, a], [, b]) => b - a)
                    .map(([type, count]) => (
                      <tr key={type}>
                        <td className="font-mono text-xs">{type}</td>
                        <td className="text-right text-xs">{count.toLocaleString()}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Hashed columns */}
        {run.hashed_columns && run.hashed_columns.length > 0 && (
          <div className="card bg-base-100 shadow">
            <div className="card-body">
              <h2 className="card-title text-base">Pseudonymised Columns</h2>
              <ul className="space-y-1">
                {run.hashed_columns.map((col) => (
                  <li key={col} className="font-mono text-xs badge badge-ghost">{col}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>

      {/* Purview discrepancies */}
      {run.purview_flagged && run.purview_flagged.length > 0 && (
        <div className="card bg-base-100 shadow">
          <div className="card-body">
            <h2 className="card-title text-base text-warning">Purview Flagged Columns</h2>
            <div className="flex flex-wrap gap-2">
              {run.purview_flagged.map((col) => (
                <span key={col} className="badge badge-warning badge-sm font-mono">{col}</span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function MetaRow({
  label,
  value,
}: {
  label: string
  value: React.ReactNode
}) {
  return (
    <tr>
      <th className="text-xs font-medium opacity-60 w-36">{label}</th>
      <td className="text-xs">{value}</td>
    </tr>
  )
}
