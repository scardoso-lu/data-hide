import { Pool } from "pg"

declare global {
  // eslint-disable-next-line no-var
  var _pgPool: Pool | undefined
}

function createPool(): Pool {
  const connectionString = process.env.DATABASE_URL
  if (!connectionString) throw new Error("DATABASE_URL is not set")
  // Enforce SSL in production. Set DATABASE_DISABLE_SSL=true to opt out
  // (e.g. plain docker-compose with a local Postgres that has no TLS cert).
  const ssl =
    process.env.NODE_ENV === "production" &&
    process.env.DATABASE_DISABLE_SSL !== "true"
      ? { rejectUnauthorized: true }
      : false
  return new Pool({ connectionString, max: 5, ssl })
}

// Reuse pool across hot-reloads in development.
export const pool: Pool =
  globalThis._pgPool ?? (globalThis._pgPool = createPool())
