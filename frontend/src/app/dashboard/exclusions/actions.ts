"use server"

import { revalidatePath } from "next/cache"
import { auth } from "@/auth"
import { addExclusion, deleteExclusion } from "@/lib/queries/exclusions"

async function requireAccess(): Promise<void> {
  const session = await auth()
  const allowed = (session as typeof session & { allowedAccess?: boolean })?.allowedAccess
  if (!session?.user || !allowed) throw new Error("Unauthorized")
}

// Allow letters, digits, underscores, and dots (schema.table notation).
// Rejects anything that has no place in a column or table name.
const IDENTIFIER_RE = /^[a-zA-Z_][a-zA-Z0-9_.]*$/
const MAX_NAME_LEN   = 128
const MAX_REASON_LEN = 500

function validateIdentifier(name: string, label: string): void {
  if (!name)                        throw new Error(`${label} is required`)
  if (name.length > MAX_NAME_LEN)   throw new Error(`${label} exceeds 128-character limit`)
  if (!IDENTIFIER_RE.test(name))    throw new Error(`${label} contains invalid characters`)
}

export async function addExclusionAction(formData: FormData): Promise<void> {
  await requireAccess()
  const tableName  = (formData.get("table_name")  as string | null)?.trim() ?? ""
  const columnName = (formData.get("column_name") as string | null)?.trim() ?? ""
  const reason     = (formData.get("reason")      as string | null)?.trim() || null

  validateIdentifier(tableName,  "table_name")
  validateIdentifier(columnName, "column_name")
  if (reason && reason.length > MAX_REASON_LEN)
    throw new Error("Reason exceeds 500-character limit")

  await addExclusion(tableName, columnName, reason)
  revalidatePath("/dashboard/exclusions")
}

export async function deleteExclusionAction(
  tableName: string,
  columnName: string,
): Promise<void> {
  await requireAccess()
  validateIdentifier(tableName,  "table_name")
  validateIdentifier(columnName, "column_name")
  await deleteExclusion(tableName, columnName)
  revalidatePath("/dashboard/exclusions")
}
