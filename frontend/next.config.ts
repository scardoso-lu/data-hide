import type { NextConfig } from "next"

// CSP note: Next.js App Router requires 'unsafe-inline' for its runtime
// script chunks and inline styles, but NOT 'unsafe-eval'. A nonce-based
// strict CSP would need additional build tooling.
// frame-ancestors 'none' is the critical clickjacking control.
const CSP = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data:",
  "font-src 'self'",
  "connect-src 'self'",
  "frame-src 'none'",
  "frame-ancestors 'none'",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
].join("; ")

const securityHeaders = [
  // Prevent clickjacking (redundant with CSP frame-ancestors but belt-and-suspenders)
  { key: "X-Frame-Options", value: "DENY" },
  // Prevent MIME-type sniffing
  { key: "X-Content-Type-Options", value: "nosniff" },
  // Force HTTPS for 2 years on this origin and subdomains
  { key: "Strict-Transport-Security", value: "max-age=63072000; includeSubDomains; preload" },
  // Limit referrer leakage
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  // Disable unnecessary browser features
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=(), payment=()" },
  { key: "Content-Security-Policy", value: CSP },
  // Disable legacy XSS auditor (causes information leakage in older browsers)
  { key: "X-XSS-Protection", value: "0" },
  // Prevent this page from being opened in a cross-origin context
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  // Prevent cross-origin resources from being read by this document
  { key: "Cross-Origin-Resource-Policy", value: "same-origin" },
]

const nextConfig: NextConfig = {
  output: "standalone",
  poweredByHeader: false,

  async headers() {
    return [
      {
        source: "/(.*)",
        headers: securityHeaders,
      },
      {
        // Prevent authenticated pages from being cached by proxies or shared caches
        source: "/dashboard/(.*)",
        headers: [
          { key: "Cache-Control", value: "no-store" },
        ],
      },
      {
        // Prevent login/unauthorized pages from being cached — back-button after
        // logout must not show a stale authenticated or access-denied state.
        source: "/(login|unauthorized)",
        headers: [
          { key: "Cache-Control", value: "no-store" },
        ],
      },
    ]
  },
}

export default nextConfig
