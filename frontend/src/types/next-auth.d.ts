import type { DefaultSession } from "next-auth"

/**
 * Extend the built-in NextAuth Session type so that
 * `session.allowedAccess` is typed throughout the app.
 */
declare module "next-auth" {
  interface Session extends DefaultSession {
    /** True when the user belongs to at least one AZURE_AD_ALLOWED_GROUPS group. */
    allowedAccess: boolean
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    allowedAccess?: boolean
  }
}
