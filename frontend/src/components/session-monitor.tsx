"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"

// Poll the NextAuth session endpoint every 5 minutes. When the 8-hour admin
// session expires the response will have no user object, and we redirect to
// /login so the user gets a clear prompt rather than a cryptic server error.
const POLL_INTERVAL_MS = 5 * 60 * 1_000

export default function SessionMonitor() {
  const router = useRouter()

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch("/api/auth/session", { credentials: "include" })
        if (!res.ok) return
        const data = await res.json()
        if (!data?.user) {
          router.push("/login")
        }
      } catch {
        // Network error — don't force-redirect; the next successful check will catch it.
      }
    }

    const id = setInterval(check, POLL_INTERVAL_MS)
    return () => clearInterval(id)
  }, [router])

  return null
}
