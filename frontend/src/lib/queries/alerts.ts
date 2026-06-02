import { pool } from "@/lib/db"

export interface PipelineAlert {
  id: number
  severity: string
  subject: string
  body: string | null
  table_name: string | null
  run_id: string | null
  created_at: string
}

export async function getAlerts(limit: number): Promise<PipelineAlert[]> {
  const { rows } = await pool.query<PipelineAlert>(
    `SELECT id, severity, subject, body, table_name, run_id, created_at
     FROM pii_pipeline_alerts
     ORDER BY created_at DESC
     LIMIT $1`,
    [Math.min(limit, 500)],
  )
  return rows
}

export async function getRecentAlertCount(): Promise<number> {
  const { rows } = await pool.query<{ count: string }>(
    `SELECT COUNT(*)::text AS count
     FROM pii_pipeline_alerts
     WHERE created_at > NOW() - INTERVAL '24 hours'`,
  )
  return parseInt(rows[0].count, 10)
}
