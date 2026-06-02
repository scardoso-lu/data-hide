import { signOut } from "@/auth"

/**
 * Shown when the user successfully authenticated with Microsoft but is not
 * a member of any group listed in AZURE_AD_ALLOWED_GROUPS.
 */
export default function UnauthorizedPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-base-200">
      <div className="card w-full max-w-md bg-base-100 shadow-xl">
        <div className="card-body items-center text-center gap-4">
          {/* Icon */}
          <div className="w-14 h-14 rounded-full bg-error/10 flex items-center justify-center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-7 w-7 text-error"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>

          <div>
            <h1 className="card-title text-xl justify-center">Access Denied</h1>
            <p className="text-base-content/60 text-sm mt-2 leading-relaxed">
              Your Microsoft account is not authorised to access this
              application. Contact your administrator and ask to be added to the
              required Azure AD group.
            </p>
          </div>

          <div className="divider my-0" />

          {/* Sign out — inline server action, no client JS needed */}
          <form
            className="w-full"
            action={async () => {
              "use server"
              await signOut({ redirectTo: "/login" })
            }}
          >
            <button type="submit" className="btn btn-outline w-full">
              Sign out and try a different account
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
