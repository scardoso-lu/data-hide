"use server"

import { revalidatePath } from "next/cache"
import { auth } from "@/auth"
import { addExclusion, deleteExclusion } from "@/lib/queries/exclusions"

async function requireAccess(): Promise<void> {
  const session = await auth()
  const allowed = (session as typeof session & { allowedAccess?: boolean })?.allowedAccess
  if (!session?.user || !allowed) throw new Error("Unauthorized")
}

export async function addExclusionAction(formData: FormData): Promise<void> {
  await requireAccess()
  const tableName  = (formData.get("table_name")  as string | null)?.trim()
  const columnName = (formData.get("column_name") as string | null)?.trim()
  const reason     = (formData.get("reason")      as string | null)?.trim() || null

  if (!tableName || !columnName) throw new Error("table_name and column_name are required")
  await addExclusion(tableName, columnName, reason)
  revalidatePath("/dashboard/exclusions")
}

export async function deleteExclusionAction(
  tableName: string,
  columnName: string,
): Promise<void> {
  await requireAccess()
  await deleteExclusion(tableName, columnName)
  revalidatePath("/dashboard/exclusions")
}
