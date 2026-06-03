import Link from "next/link"

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-base-200">
      <div className="card w-96 bg-base-100 shadow-xl">
        <div className="card-body items-center text-center gap-4">
          <div>
            <h1 className="card-title text-xl justify-center">Page not found</h1>
            <p className="text-base-content/60 text-sm mt-2 leading-relaxed">
              The page you requested does not exist.
            </p>
          </div>
          <div className="divider my-0" />
          <Link href="/login" className="btn btn-primary w-full">
            Return to sign in
          </Link>
        </div>
      </div>
    </div>
  )
}
