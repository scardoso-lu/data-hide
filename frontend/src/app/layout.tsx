import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "Admin",
  description: "Secure admin application",
  // Prevent all search engine indexing globally; robots.ts adds the HTTP header.
  robots: { index: false, follow: false },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" data-theme="corporate">
      <body className="antialiased">{children}</body>
    </html>
  )
}
