import { pool } from "@/lib/db"

export interface PipelineRun {
  run_id: string
  table_name: string | null
  source_uri: string
  status: string
  pipeline_version: string
  error_msg: string | null
  total_rows: number | null
  suppressed_rows: number
  residual_pii: number | null
  stage_seconds: Record<string, number> | null
  started_at: string
  finished_at: string | null
  output_type: string | null
  total_columns: number | null
  columns_scanned: number | null
  aggregate_cells: number | null
  key_vault_key_version: string | null
  purview_ok: boolean
  entity_counts: Record<string, number> | null
  hashed_columns: string[] | null
  purview_flagged: string[] | null
}

export interface RunStats {
  total: string
  success_count: string
  failure_count: string
  running_count: string
  total_rows_processed: string | null
  total_rows_suppressed: string | null
  last_run_at: string | null
}

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

const MAX_PAGE_OFFSET = 10_000

export async function getRuns(
  limit: number,
  offset = 0,
): Promise<PipelineRun[]> {
  const { rows } = await pool.query<PipelineRun>(
    `SELECT run_id, table_name, source_uri, status, pipeline_version,
            total_rows, suppressed_rows, started_at, finished_at,
            error_msg, residual_pii, stage_seconds, output_type,
            total_columns, columns_scanned, aggregate_cells,
            key_vault_key_version, purview_ok, entity_counts,
            hashed_columns, purview_flagged
     FROM pii_pipeline_runs
     ORDER BY started_at DESC
     LIMIT $1 OFFSET $2`,
    [Math.min(limit, 200), Math.min(offset, MAX_PAGE_OFFSET)],
  )
  return rows
}

export async function getRunsCount(): Promise<number> {
  const { rows } = await pool.query<{ count: string }>(
    "SELECT COUNT(*)::text AS count FROM pii_pipeline_runs",
  )
  return parseInt(rows[0].count, 10)
}

export async function getRunById(
  runId: string,
): Promise<PipelineRun | null> {
  if (!UUID_RE.test(runId)) return null
  const { rows } = await pool.query<PipelineRun>(
    `SELECT run_id, table_name, source_uri, status, pipeline_version,
            total_rows, suppressed_rows, started_at, finished_at,
            error_msg, residual_pii, stage_seconds, output_type,
            total_columns, columns_scanned, aggregate_cells,
            key_vault_key_version, purview_ok, entity_counts,
            hashed_columns, purview_flagged
     FROM pii_pipeline_runs
     WHERE run_id = $1`,
    [runId],
  )
  return rows[0] ?? null
}

export async function getRunStats(): Promise<RunStats> {
  const { rows } = await pool.query<RunStats>(`
    SELECT
      COUNT(*)::text                                        AS total,
      COUNT(*) FILTER (WHERE status = 'success')::text     AS success_count,
      COUNT(*) FILTER (WHERE status = 'failed')::text      AS failure_count,
      COUNT(*) FILTER (WHERE status = 'running')::text     AS running_count,
      SUM(total_rows)::text                                AS total_rows_processed,
      SUM(suppressed_rows)::text                           AS total_rows_suppressed,
      MAX(started_at)::text                                AS last_run_at
    FROM pii_pipeline_runs
  `)
  return rows[0]
}
