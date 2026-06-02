import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "data-hide | PII Pipeline Admin",
  description: "Manage and monitor the Fabric PII anonymization pipeline",
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
