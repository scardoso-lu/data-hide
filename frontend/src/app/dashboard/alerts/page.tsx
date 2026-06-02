import Link from "next/link"
import { getAlerts } from "@/lib/queries/alerts"

export const dynamic = "force-dynamic"

const SEVERITY_BADGE: Record<string, string> = {
  error:   "badge-error",
  warning: "badge-warning",
  info:    "badge-info",
}

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleString("en-GB", {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  })
}

export default async function AlertsPage() {
  const alerts = await getAlerts(200)

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold">Alerts</h1>
        <p className="text-base-content/60 text-sm mt-1">
          Pipeline failure and anomaly notifications (last 200).
        </p>
      </div>

      {alerts.length === 0 ? (
        <div className="card bg-base-100 shadow p-12 text-center text-base-content/40">
          No alerts — the pipeline has been running cleanly.
        </div>
      ) : (
        <div className="space-y-3">
          {alerts.map((alert) => (
            <div
              key={alert.id}
              className={`alert ${
                alert.severity === "error"
                  ? "alert-error"
                  : alert.severity === "warning"
                  ? "alert-warning"
                  : "bg-base-100 shadow"
              } flex-col items-start gap-2`}
            >
              <div className="flex items-center justify-between w-full flex-wrap gap-2">
                <div className="flex items-center gap-2">
                  <span
                    className={`badge badge-sm ${
                      SEVERITY_BADGE[alert.severity] ?? "badge-ghost"
                    }`}
                  >
                    {alert.severity}
                  </span>
                  <span className="font-semibold text-sm">{alert.subject}</span>
                  {alert.table_name && (
                    <span className="badge badge-ghost badge-sm font-mono">
                      {alert.table_name}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  {alert.run_id && (
                    <Link
                      href={`/dashboard/audit/${alert.run_id}`}
                      className="btn btn-ghost btn-xs"
                    >
                      View run
                    </Link>
                  )}
                  <span className="text-xs opacity-60 whitespace-nowrap">
                    {fmtDate(alert.created_at)}
                  </span>
                </div>
              </div>
              {alert.body && (
                <pre className="text-xs whitespace-pre-wrap opacity-80 w-full font-sans">
                  {alert.body}
                </pre>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
