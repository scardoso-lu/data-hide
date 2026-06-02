"use server"

import { revalidatePath } from "next/cache"
import { addExclusion, deleteExclusion } from "@/lib/queries/exclusions"

export async function addExclusionAction(formData: FormData): Promise<void> {
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
  await deleteExclusion(tableName, columnName)
  revalidatePath("/dashboard/exclusions")
}
