import { pool } from "@/lib/db"

export interface RowScanColumn {
  table_name: string
  column_name: string
  created_at: string
}

// (table, column) pairs opted into the pipeline's targeted row-by-row Presidio
// scan (backend table: apply_row_scan).
export async function getRowScanColumns(): Promise<RowScanColumn[]> {
  const { rows } = await pool.query<RowScanColumn>(
    `SELECT table_name, column_name, created_at
     FROM apply_row_scan
     ORDER BY table_name, column_name`,
  )
  return rows
}

export async function addRowScanColumn(
  tableName: string,
  columnName: string,
): Promise<void> {
  await pool.query(
    `INSERT INTO apply_row_scan (table_name, column_name)
     VALUES ($1, $2)
     ON CONFLICT (table_name, column_name) DO NOTHING`,
    [tableName, columnName],
  )
}

export async function deleteRowScanColumn(
  tableName: string,
  columnName: string,
): Promise<void> {
  await pool.query(
    "DELETE FROM apply_row_scan WHERE table_name = $1 AND column_name = $2",
    [tableName, columnName],
  )
}
