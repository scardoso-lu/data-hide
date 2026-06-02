/**
 * Auth.js v5 middleware.
 *
 * Runs the `authorized` callback (defined in src/auth.ts) before every
 * matched request.  Unauthenticated users are redirected to /login;
 * authenticated users visiting /login are redirected to /dashboard.
 *
 * The matcher intentionally excludes:
 *   /api/auth/*   — NextAuth's own callback / CSRF endpoints
 *   /_next/*      — Next.js static assets and image optimiser
 *   /favicon.ico  — browser default favicon request
 */
export { auth as middleware } from "@/auth"

export const config = {
  matcher: [
    "/((?!api/auth|_next/static|_next/image|favicon\\.ico).*)",
  ],
}
