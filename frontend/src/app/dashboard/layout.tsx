import { auth } from "@/auth"
import { redirect } from "next/navigation"
import Navbar from "@/components/navbar"
import Sidebar from "@/components/sidebar"
import SessionMonitor from "@/components/session-monitor"

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const session = await auth()
  if (!session) redirect("/login")
  const hasAccess =
    (session as typeof session & { allowedAccess?: boolean })?.allowedAccess ??
    false
  if (!hasAccess) redirect("/unauthorized")

  return (
    <div className="min-h-screen bg-base-200">
      <SessionMonitor />
      <Navbar user={session.user} />
      <div className="flex">
        <Sidebar />
        <main className="flex-1 min-w-0 p-6">{children}</main>
      </div>
    </div>
  )
}
