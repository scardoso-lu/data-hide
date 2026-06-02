import { signOut } from "@/auth"
import type { Session } from "next-auth"

interface Props {
  user: Session["user"]
}

/**
 * Top navigation bar — Server Component.
 * Sign-out is an inline server action; no client JS needed.
 */
export default function Navbar({ user }: Props) {
  const displayName = user?.name ?? user?.email ?? "User"
  const initials = displayName
    .split(" ")
    .slice(0, 2)
    .map((w: string) => w[0]?.toUpperCase() ?? "")
    .join("")

  return (
    <nav className="navbar bg-base-100 border-b border-base-300 h-16 px-6 sticky top-0 z-30">
      {/* Brand */}
      <div className="navbar-start">
        <span className="font-bold text-lg tracking-tight">
          <span className="text-primary">data</span>
          <span className="text-base-content">-hide</span>
        </span>
        <span className="badge badge-ghost badge-sm ml-2">admin</span>
      </div>

      {/* User menu */}
      <div className="navbar-end gap-3">
        <div className="dropdown dropdown-end">
          <label
            tabIndex={0}
            className="btn btn-ghost btn-sm gap-2 normal-case"
          >
            <div className="avatar placeholder">
              <div className="bg-neutral text-neutral-content rounded-full w-7 text-xs">
                <span>{initials || "?"}</span>
              </div>
            </div>
            <span className="hidden sm:inline max-w-[160px] truncate text-sm">
              {displayName}
            </span>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </label>
          <ul
            tabIndex={0}
            className="dropdown-content menu p-2 shadow bg-base-100 rounded-box w-48 border border-base-300 z-40"
          >
            <li className="menu-title px-4 py-1 text-xs opacity-60 truncate">
              {user?.email}
            </li>
            <li>
              <form
                action={async () => {
                  "use server"
                  await signOut({ redirectTo: "/login" })
                }}
              >
                <button
                  type="submit"
                  className="w-full text-left flex items-center gap-2 text-error"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                  </svg>
                  Sign out
                </button>
              </form>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  )
}
