import Link from "next/link"
import { getRuns, getRunsCount } from "@/lib/queries/runs"
import RunStatusBadge from "@/components/run-status-badge"

export const dynamic = "force-dynamic"

const PAGE_SIZE = 25

function fmtDate(iso: string | null): string {
  if (!iso) return "—"
  return new Date(iso).toLocaleString("en-GB", {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  })
}

function elapsedSec(started: string, finished: string | null): string {
  if (!finished) return "—"
  const sec = (new Date(finished).getTime() - new Date(started).getTime()) / 1000
  if (sec < 60) return `${sec.toFixed(1)} s`
  return `${(sec / 60).toFixed(1)} min`
}

interface Props {
  searchParams: { page?: string }
}

export default async function AuditPage({ searchParams }: Props) {
  const page = Math.max(1, parseInt(searchParams.page ?? "1", 10))
  const offset = (page - 1) * PAGE_SIZE

  const [runs, total] = await Promise.all([
    getRuns(PAGE_SIZE, offset),
    getRunsCount(),
  ])

  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Audit Runs</h1>
          <p className="text-base-content/60 text-sm mt-1">
            {total} total run{total !== 1 ? "s" : ""}
          </p>
        </div>
      </div>

      <div className="card bg-base-100 shadow p-0 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="table table-sm">
            <thead>
              <tr>
                <th>Table</th>
                <th>Status</th>
                <th>Rows in</th>
                <th>Suppressed</th>
                <th>Elapsed</th>
                <th>Version</th>
                <th>Started</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {runs.length === 0 && (
                <tr>
                  <td colSpan={8} className="text-center py-12 text-base-content/40">
                    No runs found.
                  </td>
                </tr>
              )}
              {runs.map((run) => (
                <tr key={run.run_id} className="hover">
                  <td className="font-mono text-xs max-w-[180px] truncate">
                    {run.table_name ?? "—"}
                  </td>
                  <td><RunStatusBadge status={run.status} /></td>
                  <td className="text-xs">
                    {run.total_rows != null ? run.total_rows.toLocaleString() : "—"}
                  </td>
                  <td className="text-xs">
                    {run.suppressed_rows > 0
                      ? <span className="text-warning">{run.suppressed_rows.toLocaleString()}</span>
                      : "0"
                    }
                  </td>
                  <td className="text-xs">
                    {elapsedSec(run.started_at, run.finished_at)}
                  </td>
                  <td className="font-mono text-xs opacity-60">
                    {run.pipeline_version}
                  </td>
                  <td className="text-xs text-base-content/50 whitespace-nowrap">
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

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex justify-center py-4 border-t border-base-200">
            <div className="join">
              {page > 1 && (
                <Link
                  href={`/dashboard/audit?page=${page - 1}`}
                  className="join-item btn btn-sm"
                >
                  «
                </Link>
              )}
              {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
                const p = i + 1
                return (
                  <Link
                    key={p}
                    href={`/dashboard/audit?page=${p}`}
                    className={`join-item btn btn-sm ${p === page ? "btn-active" : ""}`}
                  >
                    {p}
                  </Link>
                )
              })}
              {page < totalPages && (
                <Link
                  href={`/dashboard/audit?page=${page + 1}`}
                  className="join-item btn btn-sm"
                >
                  »
                </Link>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
