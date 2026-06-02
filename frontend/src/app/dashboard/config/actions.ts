"use server"

import { revalidatePath } from "next/cache"
import { auth } from "@/auth"
import { upsertConfig, deleteConfig } from "@/lib/queries/config"

async function requireAccess(): Promise<string> {
  const session = await auth()
  const allowed = (session as typeof session & { allowedAccess?: boolean })?.allowedAccess
  if (!session?.user || !allowed) throw new Error("Unauthorized")
  return session.user.email ?? "unknown"
}

const MAX_KEY_LEN   = 128
const MAX_VALUE_LEN = 10_000
const MAX_DESC_LEN  = 500

export async function upsertConfigAction(formData: FormData): Promise<void> {
  const actor = await requireAccess()
  const key         = (formData.get("key")         as string | null)?.trim()
  const value       = (formData.get("value")        as string | null) ?? ""
  const description = (formData.get("description")  as string | null)?.trim() || null

  if (!key)                              throw new Error("Key is required")
  if (key.length > MAX_KEY_LEN)          throw new Error("Key exceeds 128-character limit")
  if (value.length > MAX_VALUE_LEN)      throw new Error("Value exceeds 10 000-character limit")
  if (description && description.length > MAX_DESC_LEN)
                                         throw new Error("Description exceeds 500-character limit")

  await upsertConfig(key, value, description)
  console.log(`[audit] config.upsert key=${key} actor=${actor}`)
  revalidatePath("/dashboard/config")
}

export async function deleteConfigAction(key: string): Promise<void> {
  const actor = await requireAccess()
  if (!key || typeof key !== "string") throw new Error("Key is required")
  if (key.length > MAX_KEY_LEN)        throw new Error("Key exceeds 128-character limit")
  await deleteConfig(key)
  console.log(`[audit] config.delete key=${key} actor=${actor}`)
  revalidatePath("/dashboard/config")
}
