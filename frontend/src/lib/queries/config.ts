import { pool } from "@/lib/db"

export interface ConfigEntry {
  key: string
  value: string
  description: string | null
  updated_at: string
}

export async function getConfig(): Promise<ConfigEntry[]> {
  const { rows } = await pool.query<ConfigEntry>(
    `SELECT key, value, description, updated_at
     FROM pii_pipeline_config
     ORDER BY key`,
  )
  return rows
}

export async function upsertConfig(
  key: string,
  value: string,
  description: string | null,
): Promise<void> {
  await pool.query(
    `INSERT INTO pii_pipeline_config (key, value, description, updated_at)
     VALUES ($1, $2, $3, NOW())
     ON CONFLICT (key) DO UPDATE
       SET value       = EXCLUDED.value,
           description = EXCLUDED.description,
           updated_at  = NOW()`,
    [key, value, description],
  )
}

export async function deleteConfig(key: string): Promise<void> {
  await pool.query(
    "DELETE FROM pii_pipeline_config WHERE key = $1",
    [key],
  )
}
