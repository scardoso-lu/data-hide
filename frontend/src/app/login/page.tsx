import { signIn } from "@/auth"

export default function LoginPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-base-200">
      <div className="card w-96 bg-base-100 shadow-xl">
        <div className="card-body items-center text-center gap-4">
          {/* Logo / brand */}
          <div className="w-14 h-14 rounded-full bg-primary flex items-center justify-center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-7 w-7 text-primary-content"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
              />
            </svg>
          </div>

          <div>
            <h1 className="card-title text-2xl font-bold">data-hide</h1>
            <p className="text-base-content/60 text-sm mt-1">
              PII Pipeline Admin
            </p>
          </div>

          <div className="divider my-0" />

          <p className="text-sm text-base-content/70">
            Sign in with your organisational Microsoft account to manage
            pipeline settings and view audit records.
          </p>

          <form
            className="w-full"
            action={async () => {
              "use server"
              await signIn("microsoft-entra-id", { redirectTo: "/dashboard" })
            }}
          >
            <button type="submit" className="btn btn-primary w-full gap-2">
              {/* Microsoft logo (simplified) */}
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 23 23"
                className="w-5 h-5"
                aria-hidden="true"
              >
                <path fill="#f35325" d="M1 1h10v10H1z" />
                <path fill="#81bc06" d="M12 1h10v10H12z" />
                <path fill="#05a6f0" d="M1 12h10v10H1z" />
                <path fill="#ffba08" d="M12 12h10v10H12z" />
              </svg>
              Sign in with Microsoft
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
