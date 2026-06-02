import { Pool } from "pg"

declare global {
  // eslint-disable-next-line no-var
  var _pgPool: Pool | undefined
}

function createPool(): Pool {
  const connectionString = process.env.DATABASE_URL
  if (!connectionString) throw new Error("DATABASE_URL is not set")
  return new Pool({ connectionString, max: 5 })
}

// Reuse pool across hot-reloads in development.
export const pool: Pool =
  globalThis._pgPool ?? (globalThis._pgPool = createPool())
