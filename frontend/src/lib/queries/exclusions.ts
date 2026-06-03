import { pool } from "@/lib/db"

export interface ColumnExclusion {
  id: number
  table_name: string
  column_name: string
  reason: string | null
  created_at: string
}

export async function getExclusions(): Promise<ColumnExclusion[]> {
  const { rows } = await pool.query<ColumnExclusion>(
    `SELECT id, table_name, column_name, reason, created_at
     FROM pii_column_exclusions
     ORDER BY table_name, column_name`,
  )
  return rows
}

export async function addExclusion(
  tableName: string,
  columnName: string,
  reason: string | null,
): Promise<void> {
  await pool.query(
    `INSERT INTO pii_column_exclusions (table_name, column_name, reason)
     VALUES ($1, $2, $3)
     ON CONFLICT (table_name, column_name) DO UPDATE
       SET reason = EXCLUDED.reason`,
    [tableName, columnName, reason],
  )
}

export async function deleteExclusion(
  tableName: string,
  columnName: string,
): Promise<void> {
  await pool.query(
    "DELETE FROM pii_column_exclusions WHERE table_name = $1 AND column_name = $2",
    [tableName, columnName],
  )
}
