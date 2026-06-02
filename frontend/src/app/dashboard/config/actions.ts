"use server"

import { revalidatePath } from "next/cache"
import { upsertConfig, deleteConfig } from "@/lib/queries/config"

export async function upsertConfigAction(formData: FormData): Promise<void> {
  const key = (formData.get("key") as string | null)?.trim()
  const value = (formData.get("value") as string | null) ?? ""
  const description = (formData.get("description") as string | null)?.trim() || null

  if (!key) throw new Error("Key is required")
  await upsertConfig(key, value, description)
  revalidatePath("/dashboard/config")
}

export async function deleteConfigAction(key: string): Promise<void> {
  if (!key) throw new Error("Key is required")
  await deleteConfig(key)
  revalidatePath("/dashboard/config")
}
