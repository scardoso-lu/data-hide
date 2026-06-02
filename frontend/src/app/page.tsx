import { redirect } from "next/navigation"
import { auth } from "@/auth"

/**
 * Root page — bounce to /dashboard if authenticated, /login otherwise.
 * The middleware also handles this redirect, but this ensures the root URL
 * always resolves to something meaningful.
 */
export default async function Home() {
  const session = await auth()
  redirect(session ? "/dashboard" : "/login")
}
