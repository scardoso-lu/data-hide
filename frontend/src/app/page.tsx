import { redirect } from "next/navigation"
import { auth } from "@/auth"

/**
 * Root page — bounce to /admin if authenticated, /login otherwise.
 */
export default async function Home() {
  const session = await auth()
  redirect(session ? "/admin" : "/login")
}
