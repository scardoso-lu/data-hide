import Link from "next/link"
import { getRunStats } from "@/lib/queries/runs"
import { getRecentAlertCount } from "@/lib/queries/alerts"
import RunStatusBadge from "@/components/run-status-badge"
import { getRuns } from "@/lib/queries/runs"

export const dynamic = "force-dynamic"

function fmtNumber(n: string | number | null): string {
  if (n === null || n === undefined) return "—"
  return Number(n).toLocaleString()
}

function fmtDate(iso: string | null): string {
  if (!iso) return "—"
  return new Date(iso).toLocaleString("en-GB", {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  })
}

export default async function DashboardPage() {
  const [stats, alertCount, recentRuns] = await Promise.all([
    getRunStats(),
    getRecentAlertCount(),
    getRuns(8),
  ])

  const total = parseInt(stats.total, 10)
  const successCount = parseInt(stats.success_count, 10)
  const successRate =
    total > 0 ? Math.round((successCount / total) * 100) : 0

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Overview</h1>
        <p className="text-base-content/60 text-sm mt-1">
          Pipeline health and activity summary
        </p>
      </div>

      {/* ── Stats ── */}
      <div className="stats stats-horizontal shadow w-full flex-wrap">
        <div className="stat">
          <div className="stat-title">Total Runs</div>
          <div className="stat-value text-primary">{fmtNumber(stats.total)}</div>
          <div className="stat-desc">
            {stats.running_count !== "0" && (
              <span className="text-info">{stats.running_count} running</span>
            )}
          </div>
        </div>

        <div className="stat">
          <div className="stat-title">Success Rate</div>
          <div
            className={`stat-value ${
              successRate >= 90
                ? "text-success"
                : successRate >= 70
                ? "text-warning"
                : "text-error"
            }`}
          >
            {successRate}%
          </div>
          <div className="stat-desc">
            {stats.success_count} ok / {stats.failure_count} failed
          </div>
        </div>

        <div className="stat">
          <div className="stat-title">Rows Processed</div>
          <div className="stat-value text-secondary">
            {fmtNumber(stats.total_rows_processed)}
          </div>
          <div className="stat-desc">
            {fmtNumber(stats.total_rows_suppressed)} suppressed by k-anon
          </div>
        </div>

        <div className="stat">
          <div className="stat-title">Alerts (24 h)</div>
          <div
            className={`stat-value ${
              alertCount > 0 ? "text-error" : "text-success"
            }`}
          >
            {alertCount}
          </div>
          <div className="stat-desc">
            Last run:{" "}
            <span className="font-medium">{fmtDate(stats.last_run_at)}</span>
          </div>
        </div>
      </div>

      {/* ── Recent Runs ── */}
      <div className="card bg-base-100 shadow">
        <div className="card-body p-0">
          <div className="flex items-center justify-between px-6 py-4 border-b border-base-200">
            <h2 className="font-semibold text-lg">Recent Runs</h2>
            <Link href="/dashboard/audit" className="btn btn-ghost btn-sm">
              View all →
            </Link>
          </div>
          <div className="overflow-x-auto">
            <table className="table table-sm">
              <thead>
                <tr>
                  <th>Table</th>
                  <th>Status</th>
                  <th>Rows in</th>
                  <th>Suppressed</th>
                  <th>Started</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {recentRuns.length === 0 && (
                  <tr>
                    <td colSpan={6} className="text-center py-8 text-base-content/50">
                      No runs yet.
                    </td>
                  </tr>
                )}
                {recentRuns.map((run) => (
                  <tr key={run.run_id} className="hover">
                    <td className="font-mono text-xs max-w-[200px] truncate">
                      {run.table_name ?? "—"}
                    </td>
                    <td>
                      <RunStatusBadge status={run.status} />
                    </td>
                    <td>{fmtNumber(run.total_rows)}</td>
                    <td>{fmtNumber(run.suppressed_rows)}</td>
                    <td className="text-xs text-base-content/60">
                      {fmtDate(run.started_at)}
                    </td>
                    <td>
                      <Link
                        href={`/dashboard/audit/${run.run_id}`}
                        className="btn btn-ghost btn-xs"
                      >
                        Detail
                      </Link>
                    </td>
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
