/**
 * Transparent reverse proxy to the Python Flask admin app.
 *
 * Every request reaching this handler has already passed the NextAuth
 * middleware (auth.ts → authorized callback), so the user is authenticated
 * and authorised.  This route adds two headers before forwarding:
 *
 *   X-Admin-Token  — shared secret checked by Flask (rejects direct access)
 *   X-Forwarded-User / X-Forwarded-Name — identity carried from the session
 *
 * Flask is intentionally NOT exposed on any external port in docker-compose;
 * only this proxy can reach it on the internal Docker network.
 */

import { auth } from "@/auth"
import { NextRequest, NextResponse } from "next/server"

const FLASK_URL = (process.env.FLASK_ADMIN_URL ?? "http://admin:5001").replace(/\/$/, "")
const ADMIN_TOKEN = process.env.ADMIN_TOKEN ?? ""

const SKIP_REQUEST_HEADERS = new Set(["host", "x-forwarded-host"])
const SKIP_RESPONSE_HEADERS = new Set(["transfer-encoding", "connection", "keep-alive"])

async function proxy(req: NextRequest): Promise<NextResponse> {
  const session = await auth()
  if (!session?.user) {
    return NextResponse.redirect(new URL("/login", req.url))
  }

  if (!ADMIN_TOKEN) {
    console.error("[admin-proxy] ADMIN_TOKEN is not set — refusing to proxy")
    return new NextResponse("Admin token not configured.", { status: 503 })
  }

  const target = `${FLASK_URL}${req.nextUrl.pathname}${req.nextUrl.search}`

  // Build forwarded headers
  const fwd = new Headers()
  for (const [k, v] of req.headers.entries()) {
    if (!SKIP_REQUEST_HEADERS.has(k.toLowerCase())) fwd.set(k, v)
  }
  fwd.set("x-admin-token", ADMIN_TOKEN)
  fwd.set("x-forwarded-user", session.user.email ?? "")
  fwd.set("x-forwarded-name", session.user.name ?? "")

  const hasBody = req.method !== "GET" && req.method !== "HEAD"
  const body = hasBody ? await req.arrayBuffer() : undefined

  let flaskRes: Response
  try {
    flaskRes = await fetch(target, {
      method: req.method,
      headers: fwd,
      body,
      redirect: "manual",
    })
  } catch (err) {
    console.error("[admin-proxy] Flask unreachable:", err)
    return new NextResponse("Admin service unavailable.", { status: 502 })
  }

  const resHeaders = new Headers()
  for (const [k, v] of flaskRes.headers.entries()) {
    if (!SKIP_RESPONSE_HEADERS.has(k.toLowerCase())) resHeaders.set(k, v)
  }

  const resBody = await flaskRes.arrayBuffer()
  return new NextResponse(resBody, { status: flaskRes.status, headers: resHeaders })
}

export const GET = proxy
export const POST = proxy
