"use server"

import { revalidatePath } from "next/cache"
import { auth } from "@/auth"
import { addRowScanColumn, deleteRowScanColumn } from "@/lib/queries/apply-row-scan"

async function requireAccess(): Promise<string> {
  const session = await auth()
  const allowed = (session as typeof session & { allowedAccess?: boolean })?.allowedAccess
  if (!session?.user || !allowed) throw new Error("Unauthorized")
  return session.user.email ?? "unknown"
}

// Allow letters, digits, underscores, and dots (schema.table notation).
const IDENTIFIER_RE = /^[a-zA-Z_][a-zA-Z0-9_.]*$/
const MAX_NAME_LEN = 128

function validateIdentifier(name: string, label: string): void {
  if (!name)                      throw new Error(`${label} is required`)
  if (name.length > MAX_NAME_LEN) throw new Error(`${label} exceeds 128-character limit`)
  if (!IDENTIFIER_RE.test(name))  throw new Error(`${label} contains invalid characters`)
}

export async function addRowScanColumnAction(formData: FormData): Promise<void> {
  const actor = await requireAccess()
  const tableName  = (formData.get("table_name")  as string | null)?.trim() ?? ""
  const columnName = (formData.get("column_name") as string | null)?.trim() ?? ""

  validateIdentifier(tableName,  "table_name")
  validateIdentifier(columnName, "column_name")

  await addRowScanColumn(tableName, columnName)
  console.log(`[audit] apply_row_scan.add table=${tableName} column=${columnName} actor=${actor}`)
  revalidatePath("/dashboard/row-scan")
}

export async function deleteRowScanColumnAction(
  tableName: string,
  columnName: string,
): Promise<void> {
  const actor = await requireAccess()
  validateIdentifier(tableName,  "table_name")
  validateIdentifier(columnName, "column_name")
  await deleteRowScanColumn(tableName, columnName)
  console.log(`[audit] apply_row_scan.delete table=${tableName} column=${columnName} actor=${actor}`)
  revalidatePath("/dashboard/row-scan")
}
