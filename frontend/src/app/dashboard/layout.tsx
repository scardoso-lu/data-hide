import { auth } from "@/auth"
import { redirect } from "next/navigation"
import Navbar from "@/components/navbar"
import Sidebar from "@/components/sidebar"

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const session = await auth()
  // Middleware handles unauthenticated redirects, but double-check here so
  // TypeScript knows session is non-null for the rest of this component.
  if (!session) redirect("/login")

  return (
    <div className="min-h-screen bg-base-200">
      <Navbar user={session.user} />
      <div className="flex">
        <Sidebar />
        <main className="flex-1 min-w-0 p-6">{children}</main>
      </div>
    </div>
  )
}
