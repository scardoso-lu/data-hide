"use server"

import { revalidatePath } from "next/cache"
import { auth } from "@/auth"
import { upsertConfig, deleteConfig } from "@/lib/queries/config"

async function requireAccess(): Promise<void> {
  const session = await auth()
  const allowed = (session as typeof session & { allowedAccess?: boolean })?.allowedAccess
  if (!session?.user || !allowed) throw new Error("Unauthorized")
}

export async function upsertConfigAction(formData: FormData): Promise<void> {
  await requireAccess()
  const key = (formData.get("key") as string | null)?.trim()
  const value = (formData.get("value") as string | null) ?? ""
  const description = (formData.get("description") as string | null)?.trim() || null

  if (!key) throw new Error("Key is required")
  await upsertConfig(key, value, description)
  revalidatePath("/dashboard/config")
}

export async function deleteConfigAction(key: string): Promise<void> {
  await requireAccess()
  if (!key) throw new Error("Key is required")
  await deleteConfig(key)
  revalidatePath("/dashboard/config")
}
