import type { MetadataRoute } from "next"

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: "*",
      // Disallow all crawling — this is an internal admin panel, not a public site.
      disallow: "/",
    },
  }
}
