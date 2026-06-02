import NextAuth from "next-auth"
import MicrosoftEntraID from "next-auth/providers/microsoft-entra-id"

// Fail fast at module load time so a misconfigured container never serves traffic.
// `openssl rand -base64 32` produces 44 characters; enforce >= 32 as the floor.
if (process.env.NODE_ENV === "production") {
  const s = process.env.AUTH_SECRET
  if (!s || s === "build-placeholder" || s.length < 32) {
    throw new Error(
      "AUTH_SECRET must be a cryptographically random string of ≥ 32 characters. " +
        "Generate one with: openssl rand -base64 32",
    )
  }
}

// ── Allowed groups ────────────────────────────────────────────────────────────
// Comma-separated Azure AD group Object IDs.
// Leave empty (or unset) to allow every authenticated user in.
// Find an ID in: Azure Portal → Groups → <group> → Overview → Object ID.
const ALLOWED_GROUPS = (process.env.AZURE_AD_ALLOWED_GROUPS ?? "")
  .split(",")
  .map((g) => g.trim())
  .filter(Boolean)

// ── Graph API fallback ────────────────────────────────────────────────────────
/**
 * Fetch all group IDs the user is a direct member of via Microsoft Graph.
 *
 * Only called when Azure is not embedding a `groups` claim in the ID token
 * (the default for new app registrations).  Requires the
 * `GroupMember.Read.All` *delegated* permission — an admin must grant consent
 * once via:
 *   Azure Portal → App registrations → <app> → API permissions → Grant admin consent
 *
 * Returns [] on any error; access is then denied when ALLOWED_GROUPS is set.
 */
async function fetchGraphGroups(accessToken: string): Promise<string[]> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 5_000)
  try {
    const res = await fetch(
      "https://graph.microsoft.com/v1.0/me/memberOf?$select=id&$top=999",
      {
        headers: { Authorization: `Bearer ${accessToken}` },
        signal: controller.signal,
      },
    )
    if (!res.ok) {
      console.error("[auth] Graph /memberOf returned HTTP", res.status)
      return []
    }
    const data = (await res.json()) as { value: { id: string }[] }
    return data.value.map((g) => g.id)
  } catch {
    console.error("[auth] fetchGraphGroups failed — check Graph API permissions")
    return []
  } finally {
    clearTimeout(timer)
  }
}

// ── Access resolution ─────────────────────────────────────────────────────────
/**
 * Decide whether this user should be granted access.
 *
 * Resolution order:
 *  1. AZURE_AD_ALLOWED_GROUPS is empty           → allow everyone (dev default)
 *  2. ID token contains a `groups` claim         → check against the list
 *     (requires: Azure Portal → App registration → Token configuration →
 *      Add groups claim → Security groups)
 *  3. No groups claim in token                   → call Graph /me/memberOf
 *     (requires: GroupMember.Read.All, admin consent)
 */
async function resolveAccess(
  profileGroups: string[] | undefined,
  accessToken: string | undefined,
): Promise<boolean> {
  if (ALLOWED_GROUPS.length === 0) return true

  if (profileGroups !== undefined) {
    // Fast path — groups were embedded in the ID token, no Graph call needed.
    return profileGroups.some((g) => ALLOWED_GROUPS.includes(g))
  }

  if (accessToken) {
    const graphGroups = await fetchGraphGroups(accessToken)
    return graphGroups.some((g) => ALLOWED_GROUPS.includes(g))
  }

  // No groups claim and no access token — deny by default.
  return false
}

// ── NextAuth ──────────────────────────────────────────────────────────────────
export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [
    MicrosoftEntraID({
      clientId: process.env.AZURE_AD_CLIENT_ID!,
      clientSecret: process.env.AZURE_AD_CLIENT_SECRET!,
      // Leave undefined for multi-tenant; set AZURE_AD_TENANT_ID for
      // single-tenant deployments (recommended for corporate setups).
      tenantId: process.env.AZURE_AD_TENANT_ID,
    }),
  ],

  pages: {
    signIn: "/login",
    // Redirect all OAuth/callback errors to the login page so the default
    // NextAuth error page (which exposes error-type names) is never shown.
    error: "/login",
  },

  session: {
    // Limit admin sessions to 8 hours (A07 — authentication failures)
    maxAge: 8 * 60 * 60,
  },

  callbacks: {
    /**
     * jwt — fires on every sign-in and token refresh.
     *
     * `account` is only present on the *initial* sign-in, so group membership
     * is resolved once and the boolean is cached in the JWT.  All subsequent
     * requests read the cached value — no Graph API call per request.
     */
    async jwt({ token, account, profile }) {
      if (account) {
        const profileGroups = (
          profile as Record<string, unknown> | undefined
        )?.groups as string[] | undefined

        token.allowedAccess = await resolveAccess(
          profileGroups,
          account.access_token,
        )
      } else {
        // Refresh path — preserve existing value; deny if somehow missing.
        token.allowedAccess = (token.allowedAccess as boolean | undefined) ?? false
      }
      return token
    },

    /** Expose allowedAccess on the session so Server Components can read it. */
    session({ session, token }) {
      return {
        ...session,
        allowedAccess: (token.allowedAccess as boolean | undefined) ?? false,
      }
    },

    /**
     * authorized — runs in middleware before every matched request.
     *
     * Route behaviour:
     *  /unauthorized  → always accessible (shows the access-denied page)
     *  /login         → accessible when not signed in; redirect away if signed in
     *  everything else → require a valid, authorised session
     */
    authorized({ auth: session, request: { nextUrl } }) {
      const isLoggedIn = !!session?.user
      const hasAccess =
        (session as (typeof session & { allowedAccess?: boolean }))
          ?.allowedAccess ?? false
      const { pathname } = nextUrl

      if (pathname.startsWith("/unauthorized")) return true

      if (pathname.startsWith("/login")) {
        if (!isLoggedIn) return true
        return Response.redirect(
          new URL(hasAccess ? "/dashboard" : "/unauthorized", nextUrl),
        )
      }

      if (!isLoggedIn) {
        console.warn(`[security] unauthenticated access attempt path=${pathname}`)
        return false
      }
      if (!hasAccess) {
        const email = (session as typeof session & { user?: { email?: string } })?.user?.email ?? "unknown"
        console.warn(`[security] access denied path=${pathname} actor=${email}`)
        return Response.redirect(new URL("/unauthorized", nextUrl))
      }
      return true
    },
  },
})
