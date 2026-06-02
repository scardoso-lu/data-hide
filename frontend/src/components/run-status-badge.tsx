interface Props {
  status: string
}

const STATUS_CLASSES: Record<string, string> = {
  success: "badge-success",
  failure: "badge-error",
  running: "badge-info",
}

export default function RunStatusBadge({ status }: Props) {
  const cls = STATUS_CLASSES[status] ?? "badge-ghost"
  return (
    <span className={`badge badge-sm font-medium ${cls}`}>
      {status === "running" && (
        <span className="loading loading-ring loading-xs mr-1" />
      )}
      {status}
    </span>
  )
}
